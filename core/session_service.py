from __future__ import annotations

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


class SessionService:
    def __init__(self, store: RedisStore):
        self.store = store
        self.settings = get_settings()
        self.asr = ASRClient()
        self.diar = DiarizationClient()

    async def start_session(self, sample_rate: int, channels: int, chunk_sec: int = 2, ssid: str | None = None):
        ssid = ssid or str(ulid.new())
        await self.store.create_or_touch_session(
            ssid,
            {
                "sample_rate": sample_rate,
                "channels": channels,
                "chunk_sec": chunk_sec,
                "last_accepted_seq": -1,
            },
        )
        return ssid

    async def ingest_chunk(self, ssid: str, seq: int, raw: bytes, t0: float | None = None) -> dict[str, Any]:
        meta = await self.store.get_session_meta(ssid)
        sample_rate = int(meta.get("sample_rate", 16000))
        channels = int(meta.get("channels", 1))

        if await self.store.queue_length() >= self.settings.global_queue_limit:
            return {"error": "GLOBAL_QUEUE_FULL", "backlog_hint": "paused"}

        unique = await self.store.record_chunk(ssid, seq, raw)
        if not unique:
            status = await self.store.get_status(ssid)
            return {
                "accepted_seq": status.get("last_accepted_seq", seq),
                "backlog_hint": status.get("backlog_hint", "ok"),
                "duplicate": True,
            }

        # Track cumulative sample position for start_ms / end_ms
        total_samples = int(meta.get("total_samples", 0))
        num_samples = len(raw) // 2  # 16-bit PCM
        start_ms = int(total_samples * 1000 / sample_rate)
        end_ms = int((total_samples + num_samples) * 1000 / sample_rate)
        await self.store.update_session_field(ssid, "total_samples", total_samples + num_samples)

        audio, metrics = preprocess_chunk(raw, channels=channels)
        qsize = await self.store.enqueue_chunk(
            {
                "ssid": ssid,
                "seq": seq,
                "sample_rate": sample_rate,
            }
        )
        backlog_hint = "ok"
        if qsize >= int(self.settings.global_queue_limit * self.settings.backpressure_pause_ratio):
            backlog_hint = "paused"
        elif qsize >= int(self.settings.global_queue_limit * self.settings.backpressure_slow_ratio):
            backlog_hint = "slow_down"

        await self.store.set_status(
            ssid,
            {
                "last_accepted_seq": seq,
                "backlog_hint": backlog_hint,
                "queue_length": qsize,
                "audio_metrics": metrics.__dict__,
            },
        )

        asr_error = None
        if self.settings.partial_mode == "on":
            segs, asr_error = await self.asr.transcribe_partial(audio, sample_rate)
            partial_text = " ".join(s.text for s in segs).strip()
            await self.store.append_partial(
                ssid, {
                    "seq": seq,
                    "text": partial_text,
                    "segments": [s.__dict__ for s in segs],
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                }
            )
        else:
            partial_text = ""

        response = {
            "accepted_seq": seq,
            "backlog_hint": backlog_hint,
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
        return {"ssid": ssid, **st}

    async def finalize(self, ssid: str) -> dict[str, Any]:
        meta = await self.store.get_session_meta(ssid)
        sample_rate = int(meta.get("sample_rate", 16000))
        partials = await self.store.get_partials(ssid)
        segments = []
        chunks = []
        offset = 0.0
        for p in partials:
            raw = await self.store.get_chunk(ssid, p["seq"])
            chunk_duration = 0.0
            if raw:
                chunk_audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
                chunks.append(chunk_audio)
                chunk_duration = len(chunk_audio) / sample_rate
            for seg in p.get("segments", []):
                seg = dict(seg)
                seg["start"] = seg.get("start", 0.0) + offset
                seg["end"] = seg.get("end", 0.0) + offset
                if seg.get("words"):
                    seg["words"] = [
                        {**w, "start": w.get("start", 0.0) + offset, "end": w.get("end", 0.0) + offset}
                        for w in seg["words"]
                    ]
                segments.append(seg)
            offset += chunk_duration
        full_audio = np.concatenate(chunks) if chunks else np.zeros(sample_rate, dtype=np.float32)

        wav_path = f"/tmp/{ssid}.wav"
        with wave.open(wav_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes((full_audio * 32767).astype("<i2").tobytes())

        diarization = []
        diarization_status = "skipped"
        diarization_source = None

        if os.path.isdir(self.settings.pyannote_local_path):
            diarization_source = self.settings.pyannote_local_path
        elif self.settings.pyannote_token:
            diarization_source = self.settings.pyannote_model

        if diarization_source:
            try:
                token = self.settings.pyannote_token if diarization_source == self.settings.pyannote_model else None
                self.diar.load(diarization_source, token, device=self.settings.diar_device)
                diarization = [d.__dict__ for d in self.diar.diarize(wav_path)]
                diarization_status = f"ok: {diarization_source}"
            except Exception as exc:
                diarization_status = f"failed: {exc}"
                diarization = []
        else:
            diarization_status = (
                "skipped: no local diarization model and PYANNOTE_TOKEN not set "
                f"(expected local path: {self.settings.pyannote_local_path})"
            )

        from core.models import ASRSegment, DiarTurn

        asr_segments = [ASRSegment(**s) for s in segments]
        diar_turns = [DiarTurn(**d) for d in diarization]
        timeline = [t.__dict__ for t in map_speakers(asr_segments, diar_turns)]

        # Build raw_stt_items with start_ms/end_ms from partials
        raw_stt_items = []
        for p in partials:
            raw_stt_items.append({
                "seq": p["seq"],
                "text": p.get("text", ""),
                "start_ms": p.get("start_ms", 0),
                "end_ms": p.get("end_ms", 0),
            })

        # Build raw_diar_segments
        raw_diar_segments = [
            {"speaker": d["speaker"], "start": d["start"], "end": d["end"]}
            for d in diarization
        ]

        return {
            "ssid": ssid,
            "timeline": timeline,
            "full_text": " ".join(s["text"] for s in segments).strip(),
            "diarization": diarization,
            "raw_stt_items": raw_stt_items,
            "raw_diar_segments": raw_diar_segments,
            "meta": {
                "matching_fallback": self.settings.matching_fallback,
                "diarization_status": diarization_status,
            },
        }
