#!/usr/bin/env bash
set -euo pipefail

exec uvicorn api.app:app --host "${APP_HOST:-0.0.0.0}" --port "${APP_PORT:-8000}"
