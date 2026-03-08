#!/usr/bin/env bash
set -euo pipefail

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
# pyannote 파이프라인은 config.yaml + 하위 모델 2개가 필요
# snapshot_download로는 config.yaml만 받고 하위 모델은 HF repo ID라 오프라인 불가
# → 하위 모델도 직접 다운로드 + 로컬 경로용 config.yaml 생성

pyannote_dir = f"{out_dir}/pyannote-speaker-diarization"
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

# 3) config.yaml (로컬 경로 참조)
config_yaml = f"""\
version: 3.1.0

pipeline:
  name: pyannote.audio.pipelines.SpeakerDiarization
  params:
    clustering: AgglomerativeClustering
    embedding: {os.path.abspath(emb_dir)}/pytorch_model.bin
    embedding_batch_size: 32
    embedding_exclude_overlap: true
    segmentation: {os.path.abspath(seg_dir)}/pytorch_model.bin
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
PY
