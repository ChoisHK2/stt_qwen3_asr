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
from clients.diarization_client import DiarizationClient
from core.config import get_settings
from core.matching import map_speakers
from storage.redis_store import RedisStore

logger = logging.getLogger("qwen3-asr.session")


class SessionService:
    def __init__(self, store: RedisStore):
        self.store = store
        self.settings = get_settings()
        self.asr = ASRClient()
        self.diar = DiarizationClient()

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

    # ── Stop: WAV 저장 + 백그라운드 diar & stt_final 시작 ──────────

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
        asyncio.create_task(self._run_diarization_background(ssid, wav_path))

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

    # ── Background diarization ─────────────────────────────────────

    async def _run_diarization_background(self, ssid: str, wav_path: str) -> None:
        try:
            diarization_source = None
            if os.path.isdir(self.settings.pyannote_local_path):
                diarization_source = self.settings.pyannote_local_path
            elif self.settings.pyannote_token:
                diarization_source = self.settings.pyannote_model

            if not diarization_source:
                await self.store.set_status(ssid, {
                    "diar_status": "error",
                    "diar_error": "no diarization model available",
                })
                return

            token = self.settings.pyannote_token if diarization_source == self.settings.pyannote_model else None
            diar_result = await asyncio.to_thread(
                self._diarize_sync, diarization_source, token, wav_path,
            )
            await self.store.set_status(ssid, {
                "diar_status": "done",
                "diar_segments": diar_result,
            })
            logger.info("Diarization done for %s: %d segments", ssid[:8], len(diar_result))
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

        if stt_final_items:
            effective_items = stt_final_items
            stt_source = "final"
        else:
            # Fallback: build items from partials
            effective_items = []
            for p in partials:
                if p.get("text"):
                    effective_items.append({
                        "start_ms": p.get("start_ms", 0),
                        "end_ms": p.get("end_ms", 0),
                        "text": p["text"],
                    })
            stt_source = "chunk"

        full_text = " ".join(
            (item.get("text") or "").strip()
            for item in sorted(effective_items, key=lambda x: (x.get("start_ms", 0), x.get("end_ms", 0)))
            if (item.get("text") or "").strip()
        ).strip()

        # Build ASR segments from effective items
        from core.models import ASRSegment, DiarTurn

        asr_segments = []
        for item in effective_items:
            start_sec = item.get("start_ms", 0) / 1000.0
            end_sec = item.get("end_ms", 0) / 1000.0
            text = (item.get("text") or "").strip()
            if text:
                asr_segments.append(ASRSegment(start=start_sec, end=end_sec, text=text))

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

            diarization_source = None
            if os.path.isdir(self.settings.pyannote_local_path):
                diarization_source = self.settings.pyannote_local_path
            elif self.settings.pyannote_token:
                diarization_source = self.settings.pyannote_model

            if diarization_source and wav_path:
                try:
                    token = self.settings.pyannote_token if diarization_source == self.settings.pyannote_model else None
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
