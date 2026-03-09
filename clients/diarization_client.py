from __future__ import annotations

import logging
import os
import wave

import numpy as np

from core.models import DiarTurn

logger = logging.getLogger("qwen3-asr.diarization")

# 싱글턴 파이프라인 (기존 diar.py 패턴: 한 번 로드 후 재사용)
_pipeline = None


def _resolve_diar_device(requested: str) -> str:
    """auto → cuda(있으면) / cpu(없으면)."""
    req = (requested or "auto").strip().lower()
    if req and req != "auto":
        return req
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def get_pipeline(model_path: str, token: str | None = None, device: str = "cpu"):
    """pyannote 파이프라인을 싱글턴으로 로드한다.

    기존 프로젝트(diar.py)에서 검증된 패턴:
    - model_path 디렉토리에 config.yaml이 있으면 로컬 오프라인 로드
    - 없으면 token으로 HuggingFace 온라인 로드
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    from pyannote.audio import Pipeline
    import torch

    config_path = os.path.join(model_path, "config.yaml")

    if os.path.isfile(config_path):
        logger.info("Loading pyannote pipeline from local: %s", model_path)
        _pipeline = Pipeline.from_pretrained(model_path)
    elif token:
        logger.info("Loading pyannote pipeline from HuggingFace with token")
        _pipeline = Pipeline.from_pretrained(model_path, token=token)
    else:
        raise RuntimeError(
            f"config.yaml not found at {config_path} and no PYANNOTE_TOKEN set. "
            f"Run scripts/download_models.sh to download pyannote models."
        )

    _pipeline.to(torch.device(device))
    return _pipeline


class DiarizationClient:
    """화자분리 클라이언트 (pyannote.audio).

    싱글턴 파이프라인 + 동기 diarize 메서드.
    session_service에서 asyncio.to_thread로 호출한다.
    """

    def __init__(self):
        self._loaded = False

    def load(self, model_path: str, token: str | None = None, device: str = "cpu"):
        """파이프라인을 로드한다 (이미 로드되었으면 스킵)."""
        if self._loaded:
            return
        get_pipeline(model_path, token, device)
        self._loaded = True

    def diarize(self, wav_path: str) -> list[DiarTurn]:
        import torch

        pipeline = _pipeline
        if not pipeline:
            raise RuntimeError("Diarization pipeline not loaded. Call load() first.")

        with wave.open(wav_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            pcm = wf.readframes(wf.getnframes())

        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)

        waveform = torch.from_numpy(audio).unsqueeze(0)
        result = pipeline({"waveform": waveform, "sample_rate": sample_rate})

        # pyannote 3.x: result가 직접 Annotation이거나 .speaker_diarization 속성
        annotation = getattr(result, "speaker_diarization", result)

        turns: list[DiarTurn] = []
        for seg, _, speaker in annotation.itertracks(yield_label=True):
            turns.append(DiarTurn(speaker=str(speaker), start=float(seg.start), end=float(seg.end)))
        return turns
