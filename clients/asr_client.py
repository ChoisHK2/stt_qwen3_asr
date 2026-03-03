from __future__ import annotations

import base64
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

    def _parse_chat_response(self, data: dict, audio: np.ndarray, sample_rate: int) -> list[ASRSegment]:
        text = ""
        choices = data.get("choices")
        if choices and len(choices) > 0:
            text = choices[0].get("message", {}).get("content", "").strip()

        duration = len(audio) / sample_rate
        if text:
            return [ASRSegment(start=0.0, end=duration, text=text, words=[])]
        return [ASRSegment(start=0.0, end=duration, text="", words=[])]

    async def transcribe_partial(self, audio: np.ndarray, sample_rate: int) -> tuple[list[ASRSegment], str | None]:
        wav_bytes = self._to_wav_bytes(audio, sample_rate)
        audio_b64 = base64.b64encode(wav_bytes).decode()

        payload = {
            "model": self.settings.vllm_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {
                                "url": f"data:audio/wav;base64,{audio_b64}",
                            },
                        },
                    ],
                },
            ],
        }

        try:
            async with httpx.AsyncClient(timeout=self.settings.asr_timeout_sec) as cli:
                resp = await cli.post(
                    f"{self.settings.vllm_base_url}/v1/chat/completions",
                    json=payload,
                )
                resp.raise_for_status()
                result = resp.json()
            return self._parse_chat_response(result, audio, sample_rate), None
        except Exception as exc:
            fallback = [ASRSegment(start=0.0, end=len(audio) / sample_rate, text="", words=[])]
            return fallback, str(exc)

    async def transcribe_full(self, audio: np.ndarray, sample_rate: int) -> tuple[list[ASRSegment], str | None]:
        return await self.transcribe_partial(audio, sample_rate)
