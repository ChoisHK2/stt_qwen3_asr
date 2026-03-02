#!/usr/bin/env bash
set -euo pipefail

role="${1:-api}"

if [[ "$role" == "worker" ]]; then
  exec python -m workers.chunk_worker
fi

exec uvicorn api.app:app --host "${APP_HOST:-0.0.0.0}" --port "${APP_PORT:-8000}"
