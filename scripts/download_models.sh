#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-./models}"
mkdir -p "$OUT_DIR"

python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download("Qwen/Qwen3-ASR-1.7B", local_dir="./models/Qwen3-ASR-1.7B", local_dir_use_symlinks=False)
snapshot_download("pyannote/speaker-diarization-community-1", local_dir="./models/pyannote-speaker-diarization-community-1", local_dir_use_symlinks=False)
print("Model snapshots downloaded into ./models")
PY
