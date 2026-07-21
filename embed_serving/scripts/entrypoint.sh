#!/bin/sh
# entrypoint.sh — download BGE-M3 to the model volume on first boot, then serve.
# The compose service mounts ./data/embed-models -> /app/models (persistent EBS).
# On first boot the volume is empty, so we snapshot_download BAAI/bge-m3 (public,
# no HF token needed). Subsequent boots skip the download (volume populated).
set -e

MODEL_DIR="${MODEL_PATH:-/app/models}"

if [ -z "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
  echo "[embed-serving] empty model volume -> downloading BAAI/bge-m3 from HuggingFace (~2.2GB)"
  python -c 'from huggingface_hub import snapshot_download; snapshot_download("BAAI/bge-m3", local_dir="'"$MODEL_DIR"'")'
  echo "[embed-serving] download complete."
else
  echo "[embed-serving] model volume populated -> skipping download."
fi

exec python serve_model.py