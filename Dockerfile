# ============================================================
# 올인원 이미지: qwenllm/qwen3-asr (vLLM + 오디오 처리) 기반
# Redis + vLLM(qwen-asr-serve) + FastAPI를 하나의 컨테이너에서 실행
#
# 빌드:  docker build -t qwen3-stt .
# HTTP:  docker run --gpus all --env-file .env -v ./data:/app/data -p 8000:8000 qwen3-stt
# HTTPS: docker run --gpus all --env-file .env -v ./data:/app/data -v ./cert:/cert \
#          -e SSL_KEYFILE=/cert/key.pem -e SSL_CERTFILE=/cert/cert.pem -p 8000:8000 qwen3-stt
# ============================================================
FROM qwenllm/qwen3-asr:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# ── 프록시 환경변수 완전 해제 (오프라인 환경 안전) ─────────
ENV http_proxy= \
    https_proxy= \
    HTTP_PROXY= \
    HTTPS_PROXY= \
    no_proxy= \
    NO_PROXY= \
    ALL_PROXY=
RUN sed -i '/proxy=/Id' /etc/environment || true

# Redis 설치 (인메모리 세션 스토어)
RUN apt-get update && apt-get install -y --no-install-recommends \
    redis-server dos2unix \
    && rm -rf /var/lib/apt/lists/*

# API 서비스 의존성 설치
# (qwenllm/qwen3-asr 이미지에 torch, soundfile, vllm 등 이미 포함)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# CRLF → LF 변환
RUN dos2unix /app/entrypoint.sh && chmod +x /app/entrypoint.sh \
 && find /app -type f -name "*.sh" -print0 | xargs -0 -r dos2unix

RUN mkdir -p /app/data/audio
VOLUME ["/app/data"]

# ── 환경변수 기본값 ──────────────────────────────────────────
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV DIAR_DEVICE=auto

# vLLM 서버 설정 (entrypoint.sh에서 사용)
ENV VLLM_MODEL_PATH=/app/models/Qwen3-ASR-0.6B
ENV VLLM_MODEL_NAME=Qwen/Qwen3-ASR-0.6B
ENV VLLM_PORT=8001
ENV VLLM_GPU_UTIL=0.85
ENV VLLM_MAX_MODEL_LEN=4096
ENV VLLM_MAX_NUM_SEQS=4

# API 서버 설정
ENV APP_HOST=0.0.0.0
ENV APP_PORT=8000
ENV REDIS_URL=redis://localhost:6379/0
ENV VLLM_BASE_URL=http://localhost:8001

# HTTPS (선택사항 - 비어있으면 HTTP)
ENV SSL_KEYFILE=""
ENV SSL_CERTFILE=""

EXPOSE 8000 8001

ENTRYPOINT ["/app/entrypoint.sh"]
