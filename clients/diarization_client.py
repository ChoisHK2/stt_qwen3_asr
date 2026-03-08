from __future__ import annotations

import logging
import os
import wave

import numpy as np

from core.models import DiarTurn

logger = logging.getLogger("qwen3-asr.diarization")


class DiarizationClient:
    def __init__(self):
        self.pipeline = None

    def load(self, model_path: str, token: str | None = None, device: str = "cpu"):
        from pyannote.audio import Pipeline
        import torch

        config_path = os.path.join(model_path, "config.yaml")

        if os.path.isfile(config_path):
            # 오프라인: 로컬 config.yaml에서 로드 (하위 모델 경로 포함)
            logger.info("Loading pyannote pipeline from local config: %s", config_path)
            self.pipeline = Pipeline.from_pretrained(config_path)
        elif token:
            # 온라인: HuggingFace에서 직접 로드
            logger.info("Loading pyannote pipeline from HuggingFace: %s", model_path)
            self.pipeline = Pipeline.from_pretrained(model_path, use_auth_token=token)
        else:
            raise RuntimeError(
                f"No config.yaml found at {config_path} and no PYANNOTE_TOKEN set. "
                f"Run scripts/download_models.sh to download pyannote models."
            )

        self.pipeline.to(torch.device(device))

    def diarize(self, wav_path: str) -> list[DiarTurn]:
        import torch

        if not self.pipeline:
            raise RuntimeError("Diarization pipeline not loaded")

        with wave.open(wav_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            pcm = wf.readframes(wf.getnframes())

        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)

        waveform = torch.from_numpy(audio).unsqueeze(0)
        result = self.pipeline({"waveform": waveform, "sample_rate": sample_rate})

        annotation = getattr(result, "speaker_diarization", result)

        turns: list[DiarTurn] = []
        for seg, _, speaker in annotation.itertracks(yield_label=True):
            turns.append(DiarTurn(speaker=str(speaker), start=float(seg.start), end=float(seg.end)))
        return turns
