from __future__ import annotations

from core.models import DiarTurn


class DiarizationClient:
    def __init__(self):
        self.pipeline = None

    def load(self, model_name: str, token: str | None = None):
        from pyannote.audio import Pipeline  # delayed import for graceful selftest

        kwargs = {"use_auth_token": token} if token else {}
        self.pipeline = Pipeline.from_pretrained(model_name, **kwargs)

    def diarize(self, wav_path: str) -> list[DiarTurn]:
        if not self.pipeline:
            raise RuntimeError("Diarization pipeline not loaded")
        diarization = self.pipeline(wav_path)
        turns = []
        for seg, _, speaker in diarization.itertracks(yield_label=True):
            turns.append(DiarTurn(speaker=speaker, start=float(seg.start), end=float(seg.end)))
        return turns
