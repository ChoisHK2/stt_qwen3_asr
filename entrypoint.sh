#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 올인원 엔트리포인트: Redis → vLLM(qwen-asr-serve) → FastAPI
#
# - 로그 기반 vLLM readiness 체크 (curl 불필요)
# - vLLM 감시 프로세스 (죽으면 컨테이너 종료)
# - HTTPS 옵션 (SSL_KEYFILE/SSL_CERTFILE 설정 시 활성화)
# ============================================================

########################################
# 1) Redis 시작
########################################
echo "[entrypoint] Starting Redis..."
redis-server --daemonize yes --save "" --appendonly no --loglevel warning
echo "[entrypoint] Redis started on port 6379"

########################################
# 2) 모델 경로 검증
########################################
VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-/models/Qwen3-ASR-0.6B}"
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-Qwen/Qwen3-ASR-0.6B}"
VLLM_PORT="${VLLM_PORT:-8001}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-4}"

if [ ! -d "${VLLM_MODEL_PATH}" ]; then
  echo "[entrypoint] ERROR: Model not found at ${VLLM_MODEL_PATH}"
  echo "[entrypoint] Did you forget -v /path/to/models:/models ?"
  echo "[entrypoint] Example: docker run --gpus all -v ./models:/models ..."
  exit 1
fi

########################################
# 3) vLLM ASR 서버 시작 (백그라운드)
########################################
echo "[entrypoint] Starting vLLM..."
echo "  ├─ model path : ${VLLM_MODEL_PATH}"
echo "  ├─ model name : ${VLLM_MODEL_NAME}"
echo "  └─ port       : ${VLLM_PORT}"

qwen-asr-serve "${VLLM_MODEL_PATH}" \
  --served-model-name "${VLLM_MODEL_NAME}" \
  --host 0.0.0.0 \
  --port "${VLLM_PORT}" \
  --gpu-memory-utilization "${VLLM_GPU_UTIL}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
  --enforce-eager \
  > /tmp/vllm.log 2>&1 &
VLLM_PID=$!
echo "[entrypoint] vLLM PID=${VLLM_PID}"

########################################
# 4) vLLM 안정화 대기 (로그 기반)
########################################
echo "[entrypoint] Waiting for vLLM process to stabilize..."

# 최소 기동 시간 보장 (CUDA / 모델 로딩)
sleep 10

# 프로세스 생존 확인
if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
  echo "[entrypoint] ERROR: vLLM process exited during startup"
  echo "──────── vLLM log ────────"
  tail -100 /tmp/vllm.log
  exit 1
fi

# 로그 기반 readiness
MAX_WAIT=600
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
  if grep -q "Application startup complete" /tmp/vllm.log; then
    echo "[entrypoint] vLLM startup confirmed via logs (${WAITED}s)"
    break
  fi
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[entrypoint] ERROR: vLLM process died during startup"
    echo "──────── vLLM log ────────"
    tail -100 /tmp/vllm.log
    exit 1
  fi
  sleep 2
  WAITED=$((WAITED + 2))
done

if [ $WAITED -ge $MAX_WAIT ]; then
  echo "[entrypoint] WARNING: vLLM log readiness not detected in ${MAX_WAIT}s, proceeding anyway"
fi

echo "[entrypoint] vLLM process running"

########################################
# 5) vLLM 감시 프로세스
########################################
(
  while true; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      echo "[entrypoint] ERROR: vLLM process died. Shutting down container."
      tail -100 /tmp/vllm.log
      kill $$ 2>/dev/null || true
      exit 1
    fi
    sleep 5
  done
) &

########################################
# 6) FastAPI 시작 (포그라운드)
########################################
APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-8000}"

UVICORN_ARGS="api.app:app --host ${APP_HOST} --port ${APP_PORT}"

# HTTPS: SSL_KEYFILE + SSL_CERTFILE이 설정되면 활성화
if [ -n "${SSL_KEYFILE:-}" ] && [ -n "${SSL_CERTFILE:-}" ]; then
  if [ ! -f "${SSL_KEYFILE}" ]; then
    echo "[entrypoint] ERROR: SSL_KEYFILE not found: ${SSL_KEYFILE}"
    exit 1
  fi
  if [ ! -f "${SSL_CERTFILE}" ]; then
    echo "[entrypoint] ERROR: SSL_CERTFILE not found: ${SSL_CERTFILE}"
    exit 1
  fi
  UVICORN_ARGS="${UVICORN_ARGS} --ssl-keyfile ${SSL_KEYFILE} --ssl-certfile ${SSL_CERTFILE}"
  echo "[entrypoint] Starting FastAPI (HTTPS) on ${APP_HOST}:${APP_PORT}"
else
  echo "[entrypoint] Starting FastAPI (HTTP) on ${APP_HOST}:${APP_PORT}"
fi

exec uvicorn ${UVICORN_ARGS}
