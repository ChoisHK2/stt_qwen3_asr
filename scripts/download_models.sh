#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-./models}"
mkdir -p "$OUT_DIR"

OUT_DIR="$OUT_DIR" python - <<'PY'
import os
from huggingface_hub import snapshot_download

out_dir = os.getenv("OUT_DIR", "./models")
pyannote_token = os.getenv("PYANNOTE_TOKEN")

snapshot_download(
    "Qwen/Qwen3-ASR-1.7B",
    local_dir=f"{out_dir}/Qwen3-ASR-1.7B",
    local_dir_use_symlinks=False,
)

if pyannote_token:
    snapshot_download(
        "pyannote/speaker-diarization-community-1",
        local_dir=f"{out_dir}/pyannote-speaker-diarization-community-1",
        local_dir_use_symlinks=False,
        token=pyannote_token,
    )
    print("Downloaded Qwen3-ASR + pyannote diarization models")
else:
    print("Downloaded Qwen3-ASR model only (set PYANNOTE_TOKEN to also download pyannote diarization model)")
PY
