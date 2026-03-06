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
        return merge_same_speaker_segments(turns)


def merge_same_speaker_segments(turns: list[DiarTurn], gap_sec: float = 0.3) -> list[DiarTurn]:
    """Merge adjacent diarization turns from the same speaker within gap_sec."""
    if not turns:
        return turns
    sorted_turns = sorted(turns, key=lambda t: t.start)
    merged: list[DiarTurn] = [DiarTurn(speaker=sorted_turns[0].speaker, start=sorted_turns[0].start, end=sorted_turns[0].end)]
    for t in sorted_turns[1:]:
        prev = merged[-1]
        if t.speaker == prev.speaker and (t.start - prev.end) <= gap_sec:
            prev.end = max(prev.end, t.end)
        else:
            merged.append(DiarTurn(speaker=t.speaker, start=t.start, end=t.end))
    return merged
