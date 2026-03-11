from __future__ import annotations

import asyncio
import logging
import os
import wave
from typing import Any

import numpy as np
import ulid

from audio.preprocess import preprocess_chunk
from clients.asr_client import ASRClient
from clients.diarization_client import (
    DiarizationClient,
    DiarEpochResult,
    match_speakers_by_embedding,
)
from core.config import get_settings
from core.matching import align_final_with_chunks, map_speakers
from storage.redis_store import RedisStore

logger = logging.getLogger("qwen3-asr.session")


class SessionService:
    _diar_semaphore: asyncio.Semaphore | None = None

    def __init__(self, store: RedisStore):
        self.store = store
        self.settings = get_settings()
        self.asr = ASRClient()
        self.diar = DiarizationClient()
        # Lazy-init class-level semaphore (must be created inside running loop)
        if SessionService._diar_semaphore is None:
            SessionService._diar_semaphore = asyncio.Semaphore(self.settings.max_concurrent_diar)

    async def close(self):
        await self.asr.close()

    async def start_session(self, sample_rate: int, channels: int, chunk_sec: int = 2, ssid: str | None = None):
        ssid = ssid or str(ulid.new())
        await self.store.create_or_touch_session(
            ssid,
            {
                "sample_rate": sample_rate,
                "channels": channels,
                "chunk_sec": chunk_sec,
                "last_accepted_seq": -1,
                "is_stopped": False,
                "chunk_count": 0,
            },
        )
        return ssid

    async def ingest_chunk(
        self, ssid: str, seq: int, raw: bytes, t0: float | None = None, realtime: bool = True,
    ) -> dict[str, Any]:
        meta = await self.store.get_session_meta(ssid)
        sample_rate = int(meta.get("sample_rate", 16000))
        channels = int(meta.get("channels", 1))

        if meta.get("is_stopped"):
            return {"error": "SESSION_STOPPED", "backlog_hint": "paused"}

        # Audio duration limit
        pcm_total = await self.store.append_pcm(ssid, raw)
        new_duration = (pcm_total / 2) / sample_rate
        if new_duration > self.settings.max_session_audio_sec:
            return {"error": "AUDIO_LIMIT_EXCEEDED", "backlog_hint": "paused"}

        unique = await self.store.record_chunk(ssid, seq, raw)
        if not unique:
            status = await self.store.get_status(ssid)
            return {
                "accepted_seq": status.get("last_accepted_seq", seq),
                "backlog_hint": "ok",
                "duplicate": True,
            }

        audio, metrics = preprocess_chunk(raw, channels=channels)

        # Compute chunk position in ms
        pcm_before = pcm_total - len(raw)
        start_sample = pcm_before // 2
        end_sample = pcm_total // 2
        start_ms = int(start_sample * 1000 / sample_rate)
        end_ms = int(end_sample * 1000 / sample_rate)

        chunk_count = int(meta.get("chunk_count", 0)) + 1
        await self.store.set_status(
            ssid,
            {
                "last_accepted_seq": seq,
                "backlog_hint": "ok",
                "audio_metrics": metrics.__dict__,
                "chunk_count": chunk_count,
            },
        )

        # ── 인크리멘탈 diarization 트리거 체크 ──
        interval = self.settings.diar_chunk_interval_sec
        if interval > 0:
            prev_duration = (pcm_before / 2) / sample_rate
            # 이전 에폭 수 vs 현재 에폭 수 비교
            prev_epoch_count = int(prev_duration // interval)
            curr_epoch_count = int(new_duration // interval)
            if curr_epoch_count > prev_epoch_count:
                # 새 에폭 경계를 넘음 → 인크리멘탈 diar 트리거
                epoch_idx = curr_epoch_count - 1
                epoch_start_byte = epoch_idx * interval * sample_rate * 2
                epoch_end_byte = (epoch_idx + 1) * interval * sample_rate * 2
                asyncio.create_task(
                    self._run_incremental_diar_epoch(
                        ssid, epoch_idx, epoch_start_byte, epoch_end_byte, sample_rate,
                    )
                )

        # realtime=False: 오디오 저장만, STT 스킵
        if not realtime:
            return {
                "accepted_seq": seq,
                "backlog_hint": "ok",
                "partial_text": "",
                "audio_metrics": metrics.__dict__,
                "start_ms": start_ms,
                "end_ms": end_ms,
            }

        # realtime=True: 실시간 STT 수행 (vLLM이 자체 큐잉/배칭 처리)
        asr_error = None
        if self.settings.partial_mode == "on":
            segs, asr_error = await self.asr.transcribe_partial(audio, sample_rate)
            partial_text = " ".join(s.text for s in segs).strip()
            await self.store.append_partial(
                ssid, {
                    "seq": seq,
                    "text": partial_text,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "segments": [s.__dict__ for s in segs],
                }
            )
        else:
            partial_text = ""

        response = {
            "accepted_seq": seq,
            "backlog_hint": "ok",
            "partial_text": partial_text,
            "audio_metrics": metrics.__dict__,
            "start_ms": start_ms,
            "end_ms": end_ms,
        }
        if self.settings.partial_mode == "on" and asr_error:
            response["asr_error"] = asr_error
        return response

    async def status(self, ssid: str) -> dict[str, Any]:
        st = await self.store.get_status(ssid)
        meta = await self.store.get_session_meta(ssid)
        pcm_data = await self.store.get_pcm(ssid)
        sample_rate = int(meta.get("sample_rate", 16000))
        audio_duration_sec = round(len(pcm_data) / 2 / sample_rate, 2) if pcm_data else 0.0
        return {
            "ssid": ssid,
            "is_stopped": meta.get("is_stopped", False),
            "stt_final_status": st.get("stt_final_status", "idle"),
            "stt_final_error": st.get("stt_final_error", ""),
            "diar_status": st.get("diar_status", "idle"),
            "diar_error": st.get("diar_error", ""),
            "audio_duration_sec": audio_duration_sec,
            **st,
        }

    # ── Incremental diarization (에폭 단위) ────────────────────────

    async def _run_incremental_diar_epoch(
        self, ssid: str, epoch_idx: int, start_byte: int, end_byte: int, sample_rate: int,
    ) -> None:
        """10분 에폭에 대해 diarization + 임베딩 추출을 수행한다."""
        try:
            # diarization 소스 확인
            diarization_source = self._get_diar_source()
            if not diarization_source:
                return

            assert self._diar_semaphore is not None
            interval = self.settings.diar_chunk_interval_sec
            offset_sec = epoch_idx * interval
            logger.info("Incremental diar epoch %d queued for %s (offset=%.0fs)",
                        epoch_idx, ssid[:8], offset_sec)

            async with self._diar_semaphore:
                logger.info("Incremental diar epoch %d started for %s", epoch_idx, ssid[:8])

                # PCM 데이터에서 해당 에폭 구간 추출
                pcm_data = await self.store.get_pcm(ssid)
                epoch_pcm = pcm_data[start_byte:min(end_byte, len(pcm_data))]

                if len(epoch_pcm) < sample_rate * 2:  # 최소 1초
                    return

                # diar 모델 로드 + 에폭 diarize
                token = self._get_diar_token(diarization_source)
                turns, embeddings = await asyncio.to_thread(
                    self._diarize_epoch_sync,
                    diarization_source, token, epoch_pcm, sample_rate, offset_sec,
                )

                # 이전 에폭들의 임베딩과 매칭하여 글로벌 화자 ID 결정
                prev_epochs = await self.store.get_diar_epochs(ssid)
                speaker_map = self._resolve_speaker_map(
                    prev_epochs, turns, embeddings, epoch_idx,
                )

                # 글로벌 시간 + 글로벌 화자 ID로 변환
                global_turns = []
                for t in turns:
                    global_speaker = speaker_map.get(t.speaker, t.speaker)
                    global_turns.append({
                        "speaker": global_speaker,
                        "start": t.start + offset_sec,
                        "end": t.end + offset_sec,
                    })

                # 글로벌 임베딩 (매핑 적용)
                global_embeddings = {}
                for local_spk, emb in embeddings.items():
                    global_spk = speaker_map.get(local_spk, local_spk)
                    global_embeddings[global_spk] = emb.tolist()

                epoch_data = {
                    "epoch_idx": epoch_idx,
                    "offset_sec": offset_sec,
                    "duration_sec": len(epoch_pcm) / 2 / sample_rate,
                    "turns": global_turns,
                    "speaker_embeddings": global_embeddings,
                    "speaker_map": speaker_map,
                }
                await self.store.append_diar_epoch(ssid, epoch_data)

                logger.info(
                    "Incremental diar epoch %d done for %s: %d turns, %d speakers",
                    epoch_idx, ssid[:8], len(global_turns), len(global_embeddings),
                )

        except Exception as e:
            logger.warning("Incremental diar epoch %d error for %s: %s", epoch_idx, ssid[:8], e)

    def _get_diar_source(self) -> str | None:
        if os.path.isdir(self.settings.pyannote_local_path):
            return self.settings.pyannote_local_path
        if self.settings.pyannote_token:
            return self.settings.pyannote_model
        return None

    def _get_diar_token(self, source: str) -> str | None:
        return self.settings.pyannote_token if source == self.settings.pyannote_model else None

    def _diarize_epoch_sync(
        self, source: str, token: str | None, pcm_data: bytes, sample_rate: int, offset_sec: float,
    ) -> tuple:
        """동기 에폭 diarization (thread pool에서 실행)."""
        self.diar.load(source, token, device=self.settings.diar_device)
        turns, embeddings = self.diar.diarize_epoch(
            pcm_data, sample_rate, offset_sec, device=self.settings.diar_device,
        )
        return turns, embeddings

    def _resolve_speaker_map(
        self,
        prev_epochs: list[dict],
        curr_turns: list,
        curr_embeddings: dict,
        epoch_idx: int,
    ) -> dict[str, str]:
        """이전 에폭의 임베딩과 비교하여 화자 매핑을 결정한다."""
        # 이전 에폭들의 글로벌 임베딩 수집 (가장 최근 것 우선)
        all_prev_embeddings: dict[str, np.ndarray] = {}
        for ep in reversed(prev_epochs):
            for spk, emb_list in ep.get("speaker_embeddings", {}).items():
                if spk not in all_prev_embeddings:
                    all_prev_embeddings[spk] = np.array(emb_list)

        if not all_prev_embeddings or not curr_embeddings:
            # 첫 에폭이거나 임베딩이 없으면 순차 할당
            speaker_map: dict[str, str] = {}
            existing_count = len(all_prev_embeddings)
            counter = existing_count
            for t in curr_turns:
                if t.speaker not in speaker_map:
                    speaker_map[t.speaker] = f"SPEAKER_{counter:02d}"
                    counter += 1
            return speaker_map

        # 임베딩 기반 매칭
        embedding_map = match_speakers_by_embedding(
            all_prev_embeddings,
            curr_embeddings,
            threshold=self.settings.diar_embedding_threshold,
        )

        # 매칭되지 않은 화자에 새 글로벌 ID 할당
        existing_globals = set(all_prev_embeddings.keys()) | set(embedding_map.values())
        max_idx = 0
        for g in existing_globals:
            if g.startswith("SPEAKER_"):
                try:
                    max_idx = max(max_idx, int(g.split("_")[1]) + 1)
                except (ValueError, IndexError):
                    pass

        speaker_map = dict(embedding_map)
        for t in curr_turns:
            if t.speaker not in speaker_map:
                speaker_map[t.speaker] = f"SPEAKER_{max_idx:02d}"
                max_idx += 1

        return speaker_map

    # ── Stop: WAV 저장 + 잔여분 diar & stt_final 시작 ──────────

    async def stop_session(self, ssid: str) -> dict[str, Any]:
        meta = await self.store.get_session_meta(ssid)
        sample_rate = int(meta.get("sample_rate", 16000))

        # Mark stopped
        await self.store.create_or_touch_session(ssid, {**meta, "is_stopped": True})

        # Save WAV
        pcm_data = await self.store.get_pcm(ssid)
        os.makedirs("data/audio", exist_ok=True)
        wav_path = f"data/audio/{ssid}.wav"
        self._write_wav(wav_path, pcm_data, sample_rate)

        await self.store.set_status(ssid, {
            "audio_path": wav_path,
            "stt_final_status": "running",
            "diar_status": "running",
        })

        # Background tasks
        asyncio.create_task(self._run_stt_final_background(ssid, pcm_data, sample_rate))
        asyncio.create_task(self._run_diarization_remaining(ssid, pcm_data, sample_rate))

        return {
            "ssid": ssid,
            "status": "stopped",
            "audio_path": wav_path,
            "stt_final_status": "running",
            "diar_status": "running",
        }

    def _write_wav(self, path: str, pcm_data: bytes, sample_rate: int) -> None:
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm_data)

    # ── Background STT re-processing (larger segments) ─────────────

    async def _run_stt_final_background(self, ssid: str, pcm_data: bytes, sample_rate: int) -> None:
        try:
            segment_bytes = self.settings.stt_final_chunk_sec * sample_rate * 2
            total_bytes = len(pcm_data)
            if total_bytes == 0:
                await self.store.set_stt_final(ssid, [])
                await self.store.set_status(ssid, {"stt_final_status": "done"})
                return

            final_items: list[dict[str, Any]] = []
            offset = 0
            while offset < total_bytes:
                end = min(offset + segment_bytes, total_bytes)
                segment_pcm = pcm_data[offset:end]

                start_sample = offset // 2
                end_sample = end // 2
                start_ms = int(start_sample * 1000 / sample_rate)
                end_ms = int(end_sample * 1000 / sample_rate)

                # Convert to float32 for ASR
                audio = np.frombuffer(segment_pcm, dtype="<i2").astype(np.float32) / 32768.0
                segs, _err = await self.asr.transcribe_partial(audio, sample_rate)
                text = " ".join(s.text for s in segs).strip()

                final_items.append({
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "text": text,
                })
                offset = end

            await self.store.set_stt_final(ssid, final_items)
            await self.store.set_status(ssid, {"stt_final_status": "done"})
            logger.info(
                "STT final done for %s: %d segments (%ds each)",
                ssid[:8], len(final_items), self.settings.stt_final_chunk_sec,
            )
        except Exception as e:
            logger.warning("STT final error for %s: %s", ssid[:8], e)
            await self.store.set_status(ssid, {
                "stt_final_status": "error",
                "stt_final_error": repr(e),
            })

    # ── Background diarization: 잔여분만 처리 ────────────────────

    async def _run_diarization_remaining(self, ssid: str, pcm_data: bytes, sample_rate: int) -> None:
        """stop 시 호출: 이미 처리된 에폭 이후의 잔여 오디오만 diarize한다."""
        try:
            diarization_source = self._get_diar_source()
            if not diarization_source:
                await self.store.set_status(ssid, {
                    "diar_status": "error",
                    "diar_error": "no diarization model available",
                })
                return

            assert self._diar_semaphore is not None
            await self.store.set_status(ssid, {"diar_status": "queued"})

            async with self._diar_semaphore:
                await self.store.set_status(ssid, {"diar_status": "running"})

                # 이미 처리된 에폭 확인
                prev_epochs = await self.store.get_diar_epochs(ssid)
                interval = self.settings.diar_chunk_interval_sec
                total_duration = len(pcm_data) / 2 / sample_rate

                if interval > 0 and prev_epochs:
                    # 마지막 에폭 이후의 잔여분만 처리
                    completed_epochs = len(prev_epochs)
                    remaining_start_sec = completed_epochs * interval
                    remaining_start_byte = int(remaining_start_sec * sample_rate * 2)

                    if remaining_start_byte < len(pcm_data):
                        remaining_pcm = pcm_data[remaining_start_byte:]
                        remaining_duration = len(remaining_pcm) / 2 / sample_rate

                        logger.info(
                            "Diar remaining for %s: %.1fs (epochs done: %d, remaining: %.1fs)",
                            ssid[:8], total_duration, completed_epochs, remaining_duration,
                        )

                        token = self._get_diar_token(diarization_source)
                        turns, embeddings = await asyncio.to_thread(
                            self._diarize_epoch_sync,
                            diarization_source, token, remaining_pcm,
                            sample_rate, remaining_start_sec,
                        )

                        # 화자 매핑
                        speaker_map = self._resolve_speaker_map(
                            prev_epochs, turns, embeddings, completed_epochs,
                        )

                        # 잔여분 에폭 저장
                        global_turns = []
                        for t in turns:
                            global_spk = speaker_map.get(t.speaker, t.speaker)
                            global_turns.append({
                                "speaker": global_spk,
                                "start": t.start + remaining_start_sec,
                                "end": t.end + remaining_start_sec,
                            })

                        global_embeddings = {}
                        for local_spk, emb in embeddings.items():
                            global_spk = speaker_map.get(local_spk, local_spk)
                            global_embeddings[global_spk] = emb.tolist()

                        epoch_data = {
                            "epoch_idx": completed_epochs,
                            "offset_sec": remaining_start_sec,
                            "duration_sec": remaining_duration,
                            "turns": global_turns,
                            "speaker_embeddings": global_embeddings,
                            "speaker_map": speaker_map,
                        }
                        await self.store.append_diar_epoch(ssid, epoch_data)
                    else:
                        logger.info("Diar remaining for %s: no remaining audio", ssid[:8])

                    # 모든 에폭의 turns를 합쳐서 최종 결과 생성
                    all_epochs = await self.store.get_diar_epochs(ssid)
                    all_segments = []
                    for ep in all_epochs:
                        all_segments.extend(ep.get("turns", []))

                    await self.store.set_status(ssid, {
                        "diar_status": "done",
                        "diar_segments": all_segments,
                    })
                    logger.info(
                        "Diarization done for %s: %d total segments from %d epochs",
                        ssid[:8], len(all_segments), len(all_epochs),
                    )
                else:
                    # 인크리멘탈 에폭이 없음 → 전체 fallback
                    logger.info("No incremental epochs for %s, full diarization fallback", ssid[:8])
                    wav_path = f"data/audio/{ssid}.wav"
                    token = self._get_diar_token(diarization_source)
                    diar_result = await asyncio.to_thread(
                        self._diarize_sync, diarization_source, token, wav_path,
                    )
                    await self.store.set_status(ssid, {
                        "diar_status": "done",
                        "diar_segments": diar_result,
                    })
                    logger.info("Diarization done for %s: %d segments (full)", ssid[:8], len(diar_result))

        except Exception as e:
            logger.warning("Diarization error for %s: %s", ssid[:8], e)
            await self.store.set_status(ssid, {
                "diar_status": "error",
                "diar_error": repr(e),
            })

    def _diarize_sync(self, source: str, token: str | None, wav_path: str) -> list[dict[str, Any]]:
        self.diar.load(source, token, device=self.settings.diar_device)
        return [d.__dict__ for d in self.diar.diarize(wav_path)]

    # ── Finalize ───────────────────────────────────────────────────

    async def finalize(self, ssid: str) -> dict[str, Any]:
        meta = await self.store.get_session_meta(ssid)
        sample_rate = int(meta.get("sample_rate", 16000))
        st = await self.store.get_status(ssid)

        # STT source: stt_final (재처리) > partials (per-chunk) fallback
        stt_final_items = await self.store.get_stt_final(ssid)
        partials = await self.store.get_partials(ssid)

        from core.models import ASRSegment, DiarTurn

        # chunk partials → ASRSegment (시간 경계용)
        chunk_segments: list[ASRSegment] = []
        for p in partials:
            text = (p.get("text") or "").strip()
            if text:
                chunk_segments.append(ASRSegment(
                    start=p.get("start_ms", 0) / 1000.0,
                    end=p.get("end_ms", 0) / 1000.0,
                    text=text,
                ))

        if stt_final_items:
            # stt_final → ASRSegment (텍스트 품질용)
            final_segments: list[ASRSegment] = []
            for item in stt_final_items:
                text = (item.get("text") or "").strip()
                if text:
                    final_segments.append(ASRSegment(
                        start=item.get("start_ms", 0) / 1000.0,
                        end=item.get("end_ms", 0) / 1000.0,
                        text=text,
                    ))

            # final 텍스트를 chunk 시간 경계에 맞춰 재분할
            if chunk_segments:
                asr_segments = align_final_with_chunks(final_segments, chunk_segments)
                stt_source = "final+chunk_aligned"
            else:
                asr_segments = final_segments
                stt_source = "final"
        else:
            asr_segments = chunk_segments
            stt_source = "chunk"

        # full_text는 항상 stt_final 기준 (있으면)
        text_source = stt_final_items if stt_final_items else [
            {"start_ms": p.get("start_ms", 0), "end_ms": p.get("end_ms", 0), "text": p.get("text", "")}
            for p in partials if p.get("text")
        ]
        full_text = " ".join(
            (item.get("text") or "").strip()
            for item in sorted(text_source, key=lambda x: (x.get("start_ms", 0), x.get("end_ms", 0)))
            if (item.get("text") or "").strip()
        ).strip()

        # Diarization
        diar_segments = st.get("diar_segments", [])
        diar_status = st.get("diar_status", "skipped")

        # If diarization was not run via stop, try inline
        if diar_status not in ("done", "error"):
            wav_path = st.get("audio_path", "")
            if not wav_path:
                # Build WAV from PCM
                pcm_data = await self.store.get_pcm(ssid)
                if pcm_data:
                    os.makedirs("data/audio", exist_ok=True)
                    wav_path = f"data/audio/{ssid}.wav"
                    self._write_wav(wav_path, pcm_data, sample_rate)

            diarization_source = self._get_diar_source()

            if diarization_source and wav_path:
                try:
                    token = self._get_diar_token(diarization_source)
                    self.diar.load(diarization_source, token, device=self.settings.diar_device)
                    diar_segments = [d.__dict__ for d in self.diar.diarize(wav_path)]
                    diar_status = f"ok: {diarization_source}"
                except Exception as exc:
                    diar_status = f"failed: {exc}"
                    diar_segments = []
            else:
                diar_status = "skipped: no diarization model"

        diar_turns = [DiarTurn(**d) for d in diar_segments] if diar_segments else []
        timeline = [t.__dict__ for t in map_speakers(asr_segments, diar_turns)]

        # Build raw data for partials (per-chunk STT items)
        raw_stt_items = []
        for p in partials:
            if p.get("text"):
                raw_stt_items.append({
                    "start_ms": p.get("start_ms", 0),
                    "end_ms": p.get("end_ms", 0),
                    "text": p["text"],
                })

        pcm_data = await self.store.get_pcm(ssid)
        audio_duration_sec = round(len(pcm_data) / 2 / sample_rate, 2) if pcm_data else 0.0

        return {
            "ssid": ssid,
            "session_id": ssid,
            "timeline": timeline,
            "full_text": full_text,
            "diarization": diar_segments,
            "diarization_status": diar_status,
            "meta": {
                "matching_fallback": self.settings.matching_fallback,
                "diarization_status": diar_status,
                "stt_final_chunk_sec": self.settings.stt_final_chunk_sec,
                "stt_source": stt_source,
                "audio_duration_sec": audio_duration_sec,
                "chunk_count": int(meta.get("chunk_count", 0)),
            },
            "raw_stt_items": raw_stt_items,
            "raw_diar_segments": diar_segments,
            "raw_stt_final_items": stt_final_items,
        }
