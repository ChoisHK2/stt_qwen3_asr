from __future__ import annotations

import numpy as np

from core.config import get_settings
from core.models import AudioMetrics


def pcm16_to_float32(raw: bytes, channels: int) -> np.ndarray:
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio


def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
    return audio - np.mean(audio)


def apply_gain_and_limiter(audio: np.ndarray, target_rms_dbfs: float, limiter_peak_dbfs: float) -> np.ndarray:
    rms = np.sqrt(np.mean(np.square(audio)) + 1e-9)
    target = 10 ** (target_rms_dbfs / 20)
    gain = target / max(rms, 1e-6)
    out = audio * gain
    peak_limit = 10 ** (limiter_peak_dbfs / 20)
    return np.clip(out, -peak_limit, peak_limit)


def noise_reduce(audio: np.ndarray, mode: str) -> np.ndarray:
    if mode.upper() == "QUALITY":
        kernel = np.ones(9) / 9
    else:
        kernel = np.ones(3) / 3
    smooth = np.convolve(audio, kernel, mode="same")
    return audio - (smooth * 0.08)


def estimate_metrics(audio: np.ndarray) -> AudioMetrics:
    eps = 1e-9
    rms = np.sqrt(np.mean(np.square(audio)) + eps)
    peak = np.max(np.abs(audio)) + eps
    rms_db = 20 * np.log10(rms)
    peak_db = 20 * np.log10(peak)
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.999))
    noise_floor_db = float(np.percentile(20 * np.log10(np.abs(audio) + eps), 20))
    snr = float(rms_db - noise_floor_db)
    suggestions = []
    if rms_db < -35:
        suggestions.append("INPUT_TOO_QUIET")
    if clipping_ratio > 0.01:
        suggestions.append("CLIPPING_RISK")
    if snr < 8:
        suggestions.append("HIGH_NOISE")
    return AudioMetrics(
        rms_dbfs=float(rms_db),
        peak_dbfs=float(peak_db),
        clipping_ratio=clipping_ratio,
        noise_floor_db=noise_floor_db,
        snr_estimate=snr,
        suggestions=suggestions,
    )


def preprocess_chunk(raw: bytes, channels: int) -> tuple[np.ndarray, AudioMetrics]:
    settings = get_settings()
    audio = pcm16_to_float32(raw, channels)
    if settings.preprocess_enabled:
        audio = remove_dc_offset(audio)
        audio = apply_gain_and_limiter(audio, settings.target_rms_dbfs, settings.limiter_peak_dbfs)
        audio = noise_reduce(audio, settings.noise_reduction_mode)
    metrics = estimate_metrics(audio)
    return audio, metrics
