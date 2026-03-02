from __future__ import annotations

import io
import wave

import httpx
import numpy as np

from core.config import get_settings
from core.models import ASRSegment


class ASRClient:
    def __init__(self):
        self.settings = get_settings()

    def _to_wav_bytes(self, audio: np.ndarray, sample_rate: int) -> bytes:
        audio_i16 = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
        buff = io.BytesIO()
        with wave.open(buff, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(audio_i16.tobytes())
        return buff.getvalue()

    def _parse_segments(self, data: dict, audio: np.ndarray, sample_rate: int) -> list[ASRSegment]:
        if isinstance(data.get("segments"), list) and data["segments"]:
            return [
                ASRSegment(
                    start=float(s.get("start", 0.0)),
                    end=float(s.get("end", 0.0)),
                    text=str(s.get("text", "")).strip(),
                    words=s.get("words", []),
                )
                for s in data["segments"]
            ]

        text = str(data.get("text", "")).strip()
        if text:
            return [ASRSegment(start=0.0, end=len(audio) / sample_rate, text=text, words=[])]

        return [ASRSegment(start=0.0, end=len(audio) / sample_rate, text="", words=[])]

    async def transcribe_partial(self, audio: np.ndarray, sample_rate: int) -> tuple[list[ASRSegment], str | None]:
        wav_bytes = self._to_wav_bytes(audio, sample_rate)
        files = {"file": ("chunk.wav", wav_bytes, "audio/wav")}
        data = {
            "model": self.settings.vllm_model,
            "response_format": "verbose_json",
            "timestamp_granularities[]": "segment",
        }

        try:
            async with httpx.AsyncClient(timeout=self.settings.asr_timeout_sec) as cli:
                resp = await cli.post(f"{self.settings.vllm_base_url}/v1/audio/transcriptions", data=data, files=files)
                resp.raise_for_status()
                payload = resp.json()
            return self._parse_segments(payload, audio, sample_rate), None
        except Exception as exc:
            fallback = [ASRSegment(start=0.0, end=len(audio) / sample_rate, text="", words=[])]
            return fallback, str(exc)

    async def transcribe_full(self, audio: np.ndarray, sample_rate: int) -> tuple[list[ASRSegment], str | None]:
        return await self.transcribe_partial(audio, sample_rate)
