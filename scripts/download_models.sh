#!/usr/bin/env bash
set -euo pipefail

# 사용법:
#   ./scripts/download_models.sh ./models
#   PYANNOTE_TOKEN=hf_xxx ./scripts/download_models.sh ./models
#
# Docker 내부 경로: /models
# 로컬 개발 경로:   ./models
#
# pyannote config.yaml은 상대경로로 생성되므로
# Docker에서도 로컬에서도 동일하게 동작합니다.

OUT_DIR="${1:-./models}"
mkdir -p "$OUT_DIR"

OUT_DIR="$OUT_DIR" python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

out_dir = os.getenv("OUT_DIR", "./models")
pyannote_token = os.getenv("PYANNOTE_TOKEN")

# ── Qwen3-ASR ──────────────────────────────────────────────────
snapshot_download(
    "Qwen/Qwen3-ASR-1.7B",
    local_dir=f"{out_dir}/Qwen3-ASR-1.7B",
    local_dir_use_symlinks=False,
)
print("✓ Downloaded Qwen3-ASR-1.7B")

snapshot_download(
    "Qwen/Qwen3-ASR-0.6B",
    local_dir=f"{out_dir}/Qwen3-ASR-0.6B",
    local_dir_use_symlinks=False,
)
print("✓ Downloaded Qwen3-ASR-0.6B")

# ── pyannote (오프라인용 하위 모델 포함) ───────────────────────
# 기존 프로젝트 구조에 맞춤: models/pyannote/speaker-diarization-community-1/
pyannote_dir = f"{out_dir}/pyannote/speaker-diarization-community-1"
os.makedirs(pyannote_dir, exist_ok=True)

# 1) segmentation model
seg_dir = f"{pyannote_dir}/segmentation-3.0"
snapshot_download(
    "pyannote/segmentation-3.0",
    local_dir=seg_dir,
    local_dir_use_symlinks=False,
    token=pyannote_token or None,
)
print("✓ Downloaded pyannote/segmentation-3.0")

# 2) embedding model
emb_dir = f"{pyannote_dir}/wespeaker-voxceleb-resnet34-LM"
snapshot_download(
    "pyannote/wespeaker-voxceleb-resnet34-LM",
    local_dir=emb_dir,
    local_dir_use_symlinks=False,
    token=pyannote_token or None,
)
print("✓ Downloaded pyannote/wespeaker-voxceleb-resnet34-LM")

# 3) config.yaml (상대경로 사용 → Docker/로컬 모두 호환)
#    pyannote Pipeline.from_pretrained(dir) 는 config.yaml 내 경로가
#    상대경로이면 config.yaml 기준으로 해석합니다.
config_yaml = """\
version: 3.1.0

pipeline:
  name: pyannote.audio.pipelines.SpeakerDiarization
  params:
    clustering: AgglomerativeClustering
    embedding: wespeaker-voxceleb-resnet34-LM/pytorch_model.bin
    embedding_batch_size: 32
    embedding_exclude_overlap: true
    segmentation: segmentation-3.0/pytorch_model.bin
    segmentation_batch_size: 32

params:
  clustering:
    method: centroid
    min_cluster_size: 12
    threshold: 0.7045654963945799
  segmentation:
    min_duration_off: 0.0
"""

config_path = f"{pyannote_dir}/config.yaml"
with open(config_path, "w") as f:
    f.write(config_yaml)

print(f"✓ Created pyannote config: {config_path}")
print(f"\nDone! Models saved to: {out_dir}")
print(f"  Qwen3-ASR:  {out_dir}/Qwen3-ASR-{{0.6B,1.7B}}")
print(f"  pyannote:   {pyannote_dir}")
PY
