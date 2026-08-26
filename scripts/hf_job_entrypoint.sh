#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/pilot.yaml}"
: "${HF_TOKEN:?HF_TOKEN must be supplied as an encrypted Hugging Face Job secret}"
: "${HF_MODEL_REPO:?HF_MODEL_REPO must name the private destination model repository}"
: "${MANGA109_ROOT:=/data/manga109s}"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [[ ! -f data/prepared/manifests/train.jsonl ]]; then
  if [[ ! -d "${MANGA109_ROOT}/annotations" && ! -d "${MANGA109_ROOT}/Annotations" ]]; then
    mapfile -t manga_archives < <(find "${MANGA109_ROOT}" -maxdepth 2 -type f -name '*.zip' -print)
    if [[ "${#manga_archives[@]}" -ne 1 ]]; then
      echo "Expected exactly one official Manga109-s zip in ${MANGA109_ROOT}" >&2
      exit 2
    fi
    python scripts/extract_manga109s.py \
      --archive "${manga_archives[0]}" \
      --output /workspace/data/manga109s-extracted
    MANGA109_ROOT=/workspace/data/manga109s-extracted
  fi
  python scripts/prepare_dataset.py \
    --manga109-root "${MANGA109_ROOT}" \
    --output data/prepared \
    --seed 42
fi

python scripts/validate_environment.py --config "${CONFIG_PATH}" --load-processor
python scripts/train.py --config "${CONFIG_PATH}" --push-to-hub
python -m pip install -r requirements-baselines.txt

FINAL_MODEL="checkpoints/pilot-full/final"
for baseline in manga_ocr paddle_manga paddle_1_6; do
  python scripts/evaluate_baselines.py \
    --manifest data/prepared/manifests/test.jsonl \
    --model "${baseline}" \
    --output "outputs/predictions/${baseline}.jsonl"
done
python scripts/evaluate_baselines.py \
  --manifest data/prepared/manifests/test.jsonl \
  --model new_model \
  --model-id "${FINAL_MODEL}" \
  --output outputs/predictions/new_model.jsonl
python scripts/evaluate.py outputs/predictions/*.jsonl --output-dir outputs/evaluation
python scripts/sanitize_results_for_hub.py \
  --prediction-dir outputs/predictions \
  --evaluation-dir outputs/evaluation \
  --output outputs/hub-safe-benchmark

python - <<'PY'
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_folder(
    repo_id=os.environ["HF_MODEL_REPO"],
    repo_type="model",
    folder_path="outputs/hub-safe-benchmark",
    path_in_repo="benchmark/pilot-test",
    commit_message="Upload held-out pilot metrics and gold-free predictions",
)
PY

echo "Training, evaluation, and upload completed. The Job may now terminate."
