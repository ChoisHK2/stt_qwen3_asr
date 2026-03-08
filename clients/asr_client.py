from __future__ import annotations

import asyncio
import io
import logging
import wave

import httpx
import numpy as np

from core.config import get_settings
from core.models import ASRSegment

logger = logging.getLogger("qwen3-asr.asr_client")


class ASRClient:
    """vLLM 기반 ASR 클라이언트.

    - /v1/audio/transcriptions 엔드포인트 사용 (OpenAI 호환)
    - httpx 커넥션 풀을 재사용하여 연결 오버헤드 제거
    - asyncio.Semaphore로 동시 요청 수 제한
    """

    def __init__(self):
        self.settings = get_settings()
        self._semaphore = asyncio.Semaphore(self.settings.max_concurrent_asr)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.settings.asr_timeout_sec, connect=10.0),
                limits=httpx.Limits(
                    max_connections=self.settings.max_concurrent_asr + 4,
                    max_keepalive_connections=self.settings.max_concurrent_asr,
                ),
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _to_wav_bytes(self, audio: np.ndarray, sample_rate: int) -> bytes:
        audio_i16 = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
        buff = io.BytesIO()
        with wave.open(buff, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(audio_i16.tobytes())
        return buff.getvalue()

    def _parse_transcription_response(self, data: dict, audio: np.ndarray, sample_rate: int) -> list[ASRSegment]:
        text = data.get("text", "").strip()
        duration = len(audio) / sample_rate
        return [ASRSegment(start=0.0, end=duration, text=text, words=[])]

    async def transcribe_partial(self, audio: np.ndarray, sample_rate: int) -> tuple[list[ASRSegment], str | None]:
        wav_bytes = self._to_wav_bytes(audio, sample_rate)

        async with self._semaphore:
            try:
                client = await self._get_client()
                resp = await client.post(
                    f"{self.settings.vllm_base_url}/v1/audio/transcriptions",
                    files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                    data={"model": self.settings.vllm_model},
                )
                resp.raise_for_status()
                result = resp.json()
                return self._parse_transcription_response(result, audio, sample_rate), None
            except Exception as exc:
                logger.warning("ASR transcribe error: %s", exc)
                fallback = [ASRSegment(start=0.0, end=len(audio) / sample_rate, text="", words=[])]
                return fallback, str(exc)

    async def transcribe_full(self, audio: np.ndarray, sample_rate: int) -> tuple[list[ASRSegment], str | None]:
        return await self.transcribe_partial(audio, sample_rate)
