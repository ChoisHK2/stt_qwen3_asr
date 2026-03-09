#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 올인원 엔트리포인트: Redis → vLLM(qwen-asr-serve) → FastAPI
# ============================================================

# ── 1) Redis 시작 ──────────────────────────────────────────────
echo "[entrypoint] Starting Redis..."
redis-server --daemonize yes --save "" --appendonly no --loglevel warning
echo "[entrypoint] Redis started on port 6379"

# ── 2) vLLM ASR 서버 시작 (백그라운드) ─────────────────────────
VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-/models/Qwen3-ASR-0.6B}"
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-Qwen/Qwen3-ASR-0.6B}"
VLLM_PORT="${VLLM_PORT:-8001}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-4}"

echo "[entrypoint] Starting vLLM: model=${VLLM_MODEL_PATH} port=${VLLM_PORT}"
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

# ── 3) vLLM 헬스체크 대기 ─────────────────────────────────────
echo "[entrypoint] Waiting for vLLM to be ready..."
MAX_WAIT=300
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
  if curl -sf "http://127.0.0.1:${VLLM_PORT}/health" > /dev/null 2>&1; then
    echo "[entrypoint] vLLM ready! (${WAITED}s)"
    break
  fi
  # vLLM 프로세스 크래시 감지
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "[entrypoint] ERROR: vLLM process died. Logs:"
    tail -50 /tmp/vllm.log
    exit 1
  fi
  sleep 2
  WAITED=$((WAITED + 2))
done

if [ $WAITED -ge $MAX_WAIT ]; then
  echo "[entrypoint] ERROR: vLLM did not become ready in ${MAX_WAIT}s"
  tail -50 /tmp/vllm.log
  exit 1
fi

# ── 4) FastAPI 시작 (포그라운드) ──────────────────────────────
echo "[entrypoint] Starting FastAPI on port ${APP_PORT:-8000}..."
exec uvicorn api.app:app --host "${APP_HOST:-0.0.0.0}" --port "${APP_PORT:-8000}"
