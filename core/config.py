from dataclasses import dataclass
from functools import lru_cache
import os


@dataclass
class Settings:
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    redis_url: str = "redis://localhost:6379/0"
    session_ttl_sec: int = 10800
    partial_mode: str = "on"
    overload_http_code: int = 429
    max_concurrent_asr: int = 32
    preprocess_enabled: bool = True
    noise_reduction_mode: str = "FAST"
    enable_vad: bool = False
    target_rms_dbfs: float = -20.0
    limiter_peak_dbfs: float = -1.0
    vllm_base_url: str = "http://localhost:8001"
    vllm_model: str = "Qwen/Qwen3-ASR-0.6B"
    asr_timeout_sec: int = 30
    pyannote_model: str = "pyannote/speaker-diarization-community-1"
    pyannote_local_path: str = "/models/pyannote/speaker-diarization-community-1"
    pyannote_token: str | None = None
    diar_device: str = "cpu"
    matching_fallback: str = "segment_majority"
    overlap_policy: str = "dominant"
    merge_mode: str = "gap"
    merge_gap_sec: float = 0.35
    min_turn_sec: float = 0.4
    max_turn_sec: float = 15.0
    min_words_per_turn: int = 1
    stt_final_chunk_sec: int = 60  # stop 후 재처리 세그먼트 크기(초)
    max_session_audio_sec: int = 7200  # 세션당 최대 오디오 길이(초)
    finalize_async_threshold_sec: float = 480
    model_dir: str = "/models"
    ssl_keyfile: str = ""
    ssl_certfile: str = ""


def _env_bool(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).lower() in {"1", "true", "yes", "on"}


@lru_cache
def get_settings() -> Settings:
    return Settings(
        app_host=os.getenv("APP_HOST", "0.0.0.0"),
        app_port=int(os.getenv("APP_PORT", "8000")),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        session_ttl_sec=int(os.getenv("SESSION_TTL_SEC", "10800")),
        partial_mode=os.getenv("PARTIAL_MODE", "on"),
        overload_http_code=int(os.getenv("OVERLOAD_HTTP_CODE", "429")),
        max_concurrent_asr=int(os.getenv("MAX_CONCURRENT_ASR", "32")),
        preprocess_enabled=_env_bool("PREPROCESS_ENABLED", True),
        noise_reduction_mode=os.getenv("NOISE_REDUCTION_MODE", "FAST"),
        enable_vad=_env_bool("ENABLE_VAD", False),
        target_rms_dbfs=float(os.getenv("TARGET_RMS_DBFS", "-20")),
        limiter_peak_dbfs=float(os.getenv("LIMITER_PEAK_DBFS", "-1")),
        vllm_base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8001"),
        vllm_model=os.getenv("VLLM_MODEL", "Qwen/Qwen3-ASR-0.6B"),
        asr_timeout_sec=int(os.getenv("ASR_TIMEOUT_SEC", "30")),
        pyannote_model=os.getenv("PYANNOTE_MODEL", "pyannote/speaker-diarization-community-1"),
        pyannote_local_path=os.getenv("PYANNOTE_LOCAL_PATH", "/models/pyannote/speaker-diarization-community-1"),
        pyannote_token=os.getenv("PYANNOTE_TOKEN") or None,
        diar_device=os.getenv("DIAR_DEVICE", "cpu"),
        matching_fallback=os.getenv("MATCHING_FALLBACK", "segment_majority"),
        overlap_policy=os.getenv("OVERLAP_POLICY", "dominant"),
        merge_mode=os.getenv("MERGE_MODE", "gap"),
        merge_gap_sec=float(os.getenv("MERGE_GAP_SEC", "0.35")),
        min_turn_sec=float(os.getenv("MIN_TURN_SEC", "0.4")),
        max_turn_sec=float(os.getenv("MAX_TURN_SEC", "15")),
        min_words_per_turn=int(os.getenv("MIN_WORDS_PER_TURN", "1")),
        stt_final_chunk_sec=int(os.getenv("STT_FINAL_CHUNK_SEC", "60")),
        max_session_audio_sec=int(os.getenv("MAX_SESSION_AUDIO_SEC", "7200")),
        finalize_async_threshold_sec=float(os.getenv("FINALIZE_ASYNC_THRESHOLD_SEC", "480")),
        model_dir=os.getenv("MODEL_DIR", "/models"),
        ssl_keyfile=os.getenv("SSL_KEYFILE", ""),
        ssl_certfile=os.getenv("SSL_CERTFILE", ""),
    )
