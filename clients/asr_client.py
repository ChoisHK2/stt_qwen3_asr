from __future__ import annotations

import asyncio
import base64
import io
import logging
import re
import wave

import httpx
import numpy as np

from core.config import get_settings
from core.models import ASRSegment

logger = logging.getLogger("qwen3-asr.asr_client")

# Qwen3-ASR 모델 출력에서 언어와 텍스트를 추출하는 정규식
_LANG_RE = re.compile(r"<\|([^|]+)\|>")
_TEXT_RE = re.compile(r"<\|transcription\|>(.*?)(?:<\|/?|$)", re.DOTALL)


def parse_asr_output(content: str) -> tuple[str | None, str]:
    """Qwen3-ASR chat completion 응답에서 텍스트를 추출한다.

    모델 출력 형식 예시:
        <|ko|><|transcription|>안녕하세요<|endoftext|>
        <|en|><|transcription|>hello world<|endoftext|>

    단순 텍스트도 허용한다 (fallback).
    """
    text_m = _TEXT_RE.search(content)
    if text_m:
        text = text_m.group(1).strip()
    else:
        # fallback: 태그 없이 plain text로 온 경우
        text = re.sub(r"<\|[^|]*\|>", "", content).strip()

    lang_m = _LANG_RE.search(content)
    lang = lang_m.group(1) if lang_m else None

    return lang, text


class ASRClient:
    """vLLM 기반 ASR 클라이언트 (qwen-asr-serve 호환).

    - /v1/chat/completions 엔드포인트 사용 (audio_url + base64)
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

    def _build_chat_payload(self, wav_bytes: bytes) -> dict:
        """base64 오디오를 포함하는 chat completions 요청 payload를 생성한다."""
        audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
        return {
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
                        }
                    ],
                }
            ],
        }

    def _parse_chat_response(self, data: dict, audio: np.ndarray, sample_rate: int) -> list[ASRSegment]:
        """chat completions 응답에서 ASRSegment를 추출한다."""
        content = data["choices"][0]["message"]["content"]
        _lang, text = parse_asr_output(content)
        duration = len(audio) / sample_rate
        return [ASRSegment(start=0.0, end=duration, text=text, words=[])]

    async def transcribe_partial(self, audio: np.ndarray, sample_rate: int) -> tuple[list[ASRSegment], str | None]:
        wav_bytes = self._to_wav_bytes(audio, sample_rate)
        payload = self._build_chat_payload(wav_bytes)

        async with self._semaphore:
            try:
                client = await self._get_client()
                resp = await client.post(
                    f"{self.settings.vllm_base_url}/v1/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                result = resp.json()
                return self._parse_chat_response(result, audio, sample_rate), None
            except Exception as exc:
                logger.warning("ASR transcribe error: %s", exc)
                fallback = [ASRSegment(start=0.0, end=len(audio) / sample_rate, text="", words=[])]
                return fallback, str(exc)

    async def transcribe_full(self, audio: np.ndarray, sample_rate: int) -> tuple[list[ASRSegment], str | None]:
        return await self.transcribe_partial(audio, sample_rate)
