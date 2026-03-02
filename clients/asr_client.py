from __future__ import annotations

import base64

import httpx
import numpy as np

from core.config import get_settings
from core.models import ASRSegment


class ASRClient:
    def __init__(self):
        self.settings = get_settings()

    async def transcribe_partial(self, audio: np.ndarray, sample_rate: int) -> list[ASRSegment]:
        audio_b = (audio * 32767).astype("<i2").tobytes()
        payload = {
            "model": self.settings.vllm_model,
            "audio": base64.b64encode(audio_b).decode(),
            "sample_rate": sample_rate,
            "timestamps": True,
            "mode": "partial",
        }
        try:
            async with httpx.AsyncClient(timeout=self.settings.asr_timeout_sec) as cli:
                resp = await cli.post(f"{self.settings.vllm_base_url}/v1/audio/transcriptions", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            data = {"segments": [{"start": 0.0, "end": len(audio) / sample_rate, "text": ""}]}
        return [
            ASRSegment(
                start=s.get("start", 0.0),
                end=s.get("end", 0.0),
                text=s.get("text", "").strip(),
                words=s.get("words", []),
            )
            for s in data.get("segments", [])
        ]

    async def transcribe_full(self, audio: np.ndarray, sample_rate: int) -> list[ASRSegment]:
        return await self.transcribe_partial(audio, sample_rate)
