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
        self._background_tasks: dict[str, set[asyncio.Task]] = {}  # ssid → tasks
        # Lazy-init class-level semaphore (must be created inside running loop)
        if SessionService._diar_semaphore is None:
            SessionService._diar_semaphore = asyncio.Semaphore(self.settings.max_concurrent_diar)

    async def close(self):
        # Cancel all tracked background tasks
        for ssid, tasks in self._background_tasks.items():
            for t in tasks:
                t.cancel()
        self._background_tasks.clear()
        await self.asr.close()

    def _track_task(self, ssid: str, task: asyncio.Task) -> None:
        """Background task를 세션별로 추적한다."""
        if ssid not in self._background_tasks:
            self._background_tasks[ssid] = set()
        self._background_tasks[ssid].add(task)
        task.add_done_callback(lambda t: self._background_tasks.get(ssid, set()).discard(t))

    async def cancel_session_tasks(self, ssid: str) -> None:
        """세션의 모든 background task를 취소한다."""
        tasks = self._background_tasks.pop(ssid, set())
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info("Cancelled %d background tasks for %s", len(tasks), ssid[:8])

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

    async def resume_session(self, ssid: str) -> dict[str, Any]:
        """중단된 세션을 재개한다.

        세션이 존재하고 TTL 내에 있으면 is_stopped를 해제하여 다시 청크를 받을 수 있게 한다.
        세션이 없으면 에러를 반환한다.
        """
        meta = await self.store.get_session_meta(ssid)
        if not meta:
            return {"error": "SESSION_NOT_FOUND", "ssid": ssid}

        sample_rate = int(meta.get("sample_rate", 16000))
        pcm_len = await self.store.get_pcm_length(ssid)
        audio_duration_sec = round(pcm_len / 2 / sample_rate, 2) if pcm_len else 0.0

        # is_stopped 해제하여 청크 수신 재개
        meta["is_stopped"] = False
        await self.store.create_or_touch_session(ssid, meta)

        partials = await self.store.get_partials(ssid)

        return {
            "ssid": ssid,
            "status": "resumed",
            "audio_duration_sec": audio_duration_sec,
            "chunk_count": int(meta.get("chunk_count", 0)),
            "partial_count": len(partials),
            "last_accepted_seq": int(meta.get("last_accepted_seq", -1)),
        }

    async def get_partial_results(self, ssid: str) -> dict[str, Any]:
        """세션의 지금까지 누적된 partial 결과를 반환한다 (finalize 없이).

        연결이 끊겼을 때 지금까지 인식된 텍스트만이라도 받는 fallback용.
        """
        meta = await self.store.get_session_meta(ssid)
        if not meta:
            return {"error": "SESSION_NOT_FOUND", "ssid": ssid}

        sample_rate = int(meta.get("sample_rate", 16000))
        partials = await self.store.get_partials(ssid)
        pcm_len = await self.store.get_pcm_length(ssid)
        audio_duration_sec = round(pcm_len / 2 / sample_rate, 2) if pcm_len else 0.0

        stt_final_items = await self.store.get_stt_final(ssid)

        partial_texts = []
        for p in sorted(partials, key=lambda x: x.get("start_ms", 0)):
            text = (p.get("text") or "").strip()
            if text:
                partial_texts.append({
                    "start_ms": p.get("start_ms", 0),
                    "end_ms": p.get("end_ms", 0),
                    "text": text,
                })

        full_text = " ".join(t["text"] for t in partial_texts).strip()

        if stt_final_items:
            full_text = " ".join(
                (item.get("text") or "").strip()
                for item in sorted(stt_final_items, key=lambda x: x.get("start_ms", 0))
                if (item.get("text") or "").strip()
            ).strip()

        diar_epochs = await self.store.get_diar_epochs(ssid)
        diar_segments = []
        for ep in diar_epochs:
            diar_segments.extend(ep.get("turns", []))

        st = await self.store.get_status(ssid)

        return {
            "ssid": ssid,
            "status": "partial",
            "is_stopped": meta.get("is_stopped", False),
            "audio_duration_sec": audio_duration_sec,
            "chunk_count": int(meta.get("chunk_count", 0)),
            "partial_count": len(partial_texts),
            "full_text": full_text,
            "partial_items": partial_texts,
            "stt_final_items": stt_final_items,
            "diar_segments": diar_segments,
            "diar_status": st.get("diar_status", "idle"),
            "stt_final_status": st.get("stt_final_status", "idle"),
        }

    async def ingest_chunk(
        self, ssid: str, seq: int, raw: bytes, t0: float | None = None, realtime: bool = True,
    ) -> dict[str, Any]:
        meta = await self.store.get_session_meta(ssid)
        sample_rate = int(meta.get("sample_rate", 16000))
        channels = int(meta.get("channels", 1))

        if meta.get("is_stopped"):
            return {"error": "SESSION_STOPPED", "backlog_hint": "paused"}

        # Audio duration limit — append to disk, get total length
        pcm_total = await self.store.append_pcm(ssid, raw)
        new_duration = (pcm_total / 2) / sample_rate
        if new_duration > self.settings.max_session_audio_sec:
            return {"error": "AUDIO_LIMIT_EXCEEDED", "backlog_hint": "paused"}

        unique = await self.store.record_chunk(ssid, seq)
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
            prev_epoch_count = int(prev_duration // interval)
            curr_epoch_count = int(new_duration // interval)
            if curr_epoch_count > prev_epoch_count:
                epoch_idx = curr_epoch_count - 1
                epoch_start_byte = int(epoch_idx * interval * sample_rate * 2)
                epoch_end_byte = int((epoch_idx + 1) * interval * sample_rate * 2)
                task = asyncio.create_task(
                    self._run_incremental_diar_epoch(
                        ssid, epoch_idx, epoch_start_byte, epoch_end_byte, sample_rate,
                    )
                )
                self._track_task(ssid, task)

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
        sample_rate = int(meta.get("sample_rate", 16000))
        pcm_len = await self.store.get_pcm_length(ssid)
        audio_duration_sec = round(pcm_len / 2 / sample_rate, 2) if pcm_len else 0.0
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

                # 디스크에서 해당 에폭 구간만 읽기
                epoch_pcm = await self.store.get_pcm_slice(ssid, start_byte, end_byte)

                if len(epoch_pcm) < sample_rate * 2:  # 최소 1초
                    return

                token = self._get_diar_token(diarization_source)
                turns, embeddings = await asyncio.to_thread(
                    self._diarize_epoch_sync,
                    diarization_source, token, epoch_pcm, sample_rate, offset_sec,
                )

                prev_epochs = await self.store.get_diar_epochs(ssid)
                speaker_map = self._resolve_speaker_map(
                    prev_epochs, turns, embeddings, epoch_idx,
                )

                global_turns = []
                for t in turns:
                    global_speaker = speaker_map.get(t.speaker, t.speaker)
                    global_turns.append({
                        "speaker": global_speaker,
                        "start": t.start + offset_sec,
                        "end": t.end + offset_sec,
                    })

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
        all_prev_embeddings: dict[str, np.ndarray] = {}
        for ep in reversed(prev_epochs):
            for spk, emb_list in ep.get("speaker_embeddings", {}).items():
                if spk not in all_prev_embeddings:
                    all_prev_embeddings[spk] = np.array(emb_list)

        if not all_prev_embeddings or not curr_embeddings:
            speaker_map: dict[str, str] = {}
            existing_count = len(all_prev_embeddings)
            counter = existing_count
            for t in curr_turns:
                if t.speaker not in speaker_map:
                    speaker_map[t.speaker] = f"SPEAKER_{counter:02d}"
                    counter += 1
            return speaker_map

        embedding_map = match_speakers_by_embedding(
            all_prev_embeddings,
            curr_embeddings,
            threshold=self.settings.diar_embedding_threshold,
        )

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

        # WAV 저장 — PCM을 스트리밍으로 변환 (전체를 메모리에 올리지 않음)
        audio_dir = self.settings.audio_data_dir
        os.makedirs(audio_dir, exist_ok=True)
        wav_path = os.path.join(audio_dir, f"{ssid}.wav")
        pcm_len = await self.store.get_pcm_length(ssid)
        await self._write_wav_from_disk(wav_path, ssid, pcm_len, sample_rate)

        await self.store.set_status(ssid, {
            "audio_path": wav_path,
            "stt_final_status": "running",
            "diar_status": "running",
        })

        # Background tasks — 디스크 경로 기반, 메모리에 전체 PCM 로드 안 함
        t1 = asyncio.create_task(self._run_stt_final_background(ssid, sample_rate))
        t2 = asyncio.create_task(self._run_diarization_remaining(ssid, sample_rate))
        self._track_task(ssid, t1)
        self._track_task(ssid, t2)

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

    async def _write_wav_from_disk(
        self, wav_path: str, ssid: str, pcm_len: int, sample_rate: int,
    ) -> None:
        """PCM을 전체 메모리에 올리지 않고 청크 단위로 WAV로 변환한다."""
        CHUNK = 1024 * 1024  # 1MB씩 읽기
        with wave.open(wav_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            offset = 0
            while offset < pcm_len:
                end = min(offset + CHUNK, pcm_len)
                chunk = await self.store.get_pcm_slice(ssid, offset, end)
                if not chunk:
                    break
                w.writeframes(chunk)
                offset = end

    # ── Background STT re-processing (larger segments) ─────────────

    async def _run_stt_final_background(self, ssid: str, sample_rate: int) -> None:
        try:
            segment_bytes = self.settings.stt_final_chunk_sec * sample_rate * 2
            total_bytes = await self.store.get_pcm_length(ssid)
            if total_bytes == 0:
                await self.store.set_stt_final(ssid, [])
                await self.store.set_status(ssid, {"stt_final_status": "done"})
                return

            final_items: list[dict[str, Any]] = []
            offset = 0
            while offset < total_bytes:
                end = min(offset + segment_bytes, total_bytes)
                segment_pcm = await self.store.get_pcm_slice(ssid, offset, end)

                start_sample = offset // 2
                end_sample = end // 2
                start_ms = int(start_sample * 1000 / sample_rate)
                end_ms = int(end_sample * 1000 / sample_rate)

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

    async def _run_diarization_remaining(self, ssid: str, sample_rate: int) -> None:
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

                prev_epochs = await self.store.get_diar_epochs(ssid)
                interval = self.settings.diar_chunk_interval_sec
                total_pcm_bytes = await self.store.get_pcm_length(ssid)
                total_duration = total_pcm_bytes / 2 / sample_rate

                if interval > 0 and prev_epochs:
                    completed_epochs = len(prev_epochs)
                    remaining_start_sec = completed_epochs * interval
                    remaining_start_byte = int(remaining_start_sec * sample_rate * 2)

                    if remaining_start_byte < total_pcm_bytes:
                        remaining_pcm = await self.store.get_pcm_slice(
                            ssid, remaining_start_byte, total_pcm_bytes,
                        )
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
                        del remaining_pcm  # 즉시 해제

                        speaker_map = self._resolve_speaker_map(
                            prev_epochs, turns, embeddings, completed_epochs,
                        )

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
                    wav_path = os.path.join(self.settings.audio_data_dir, f"{ssid}.wav")
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
        # stop에서 시작된 background task(stt_final, diar) 완료 대기
        pending_tasks = self._background_tasks.get(ssid, set())
        if pending_tasks:
            logger.info("Finalize waiting for %d background tasks for %s",
                        len(pending_tasks), ssid[:8])
            done, _ = await asyncio.wait(
                pending_tasks,
                timeout=self.settings.finalize_async_threshold_sec,
            )
            if len(done) < len(pending_tasks):
                logger.warning("Finalize: %d tasks timed out for %s",
                               len(pending_tasks) - len(done), ssid[:8])

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
            final_segments: list[ASRSegment] = []
            for item in stt_final_items:
                text = (item.get("text") or "").strip()
                if text:
                    final_segments.append(ASRSegment(
                        start=item.get("start_ms", 0) / 1000.0,
                        end=item.get("end_ms", 0) / 1000.0,
                        text=text,
                    ))

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
                # Build WAV from disk PCM (스트리밍)
                pcm_len = await self.store.get_pcm_length(ssid)
                if pcm_len > 0:
                    audio_dir = self.settings.audio_data_dir
                    os.makedirs(audio_dir, exist_ok=True)
                    wav_path = os.path.join(audio_dir, f"{ssid}.wav")
                    await self._write_wav_from_disk(wav_path, ssid, pcm_len, sample_rate)

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

        raw_stt_items = []
        for p in partials:
            if p.get("text"):
                raw_stt_items.append({
                    "start_ms": p.get("start_ms", 0),
                    "end_ms": p.get("end_ms", 0),
                    "text": p["text"],
                })

        pcm_len_final = await self.store.get_pcm_length(ssid)
        audio_duration_sec = round(pcm_len_final / 2 / sample_rate, 2) if pcm_len_final else 0.0

        # finalize 완료 후 세션 오디오 파일 정리
        await self.store.delete_session_files(ssid)
        # background task도 정리
        self._background_tasks.pop(ssid, None)

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
