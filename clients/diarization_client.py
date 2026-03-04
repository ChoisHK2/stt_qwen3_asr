from __future__ import annotations

import wave

import numpy as np

from core.models import DiarTurn


class DiarizationClient:
    def __init__(self):
        self.pipeline = None

    def load(self, model_name: str, token: str | None = None, device: str = "cpu"):
        from pyannote.audio import Pipeline  # delayed import for graceful selftest
        import torch

        kwargs = {"use_auth_token": token} if token else {}
        self.pipeline = Pipeline.from_pretrained(model_name, **kwargs)
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
