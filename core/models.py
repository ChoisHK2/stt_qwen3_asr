from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChunkTask:
    ssid: str
    seq: int
    sample_rate: int
    channels: int
    t0: float | None
    pcm16_bytes: bytes


@dataclass
class AudioMetrics:
    rms_dbfs: float
    peak_dbfs: float
    clipping_ratio: float
    noise_floor_db: float
    snr_estimate: float
    vad_ratio: float | None = None
    speech_prob: float | None = None
    suggestions: list[str] = field(default_factory=list)


@dataclass
class ASRSegment:
    start: float
    end: float
    text: str
    words: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DiarTurn:
    speaker: str
    start: float
    end: float


@dataclass
class TimelineTurn:
    speaker: str
    start: float
    end: float
    text: str
