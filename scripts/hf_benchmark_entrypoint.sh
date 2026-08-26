#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN must be supplied as an encrypted Hugging Face Job secret}"
: "${HF_MODEL_REPO:?HF_MODEL_REPO must name the private trained model repository}"
: "${MANGA109_ROOT:=/data/manga109s}"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-baselines.txt
bash scripts/prepare_manga109s_for_job.sh materialize

mkdir -p outputs/predictions
for baseline in manga_ocr paddle_manga paddle_1_6; do
  python scripts/evaluate_baselines.py \
    --manifest data/prepared/manifests/test.jsonl \
    --model "${baseline}" \
    --output "outputs/predictions/${baseline}.jsonl"
done
python scripts/evaluate_baselines.py \
  --manifest data/prepared/manifests/test.jsonl \
  --model new_model \
  --model-id "${HF_MODEL_REPO}" \
  --output outputs/predictions/new_model.jsonl
python scripts/evaluate.py outputs/predictions/*.jsonl --output-dir outputs/evaluation
python scripts/sanitize_results_for_hub.py \
  --prediction-dir outputs/predictions \
  --evaluation-dir outputs/evaluation \
  --output outputs/hub-safe-benchmark

python - <<'PY'
import os
from huggingface_hub import HfApi

HfApi(token=os.environ["HF_TOKEN"]).upload_folder(
    repo_id=os.environ["HF_MODEL_REPO"],
    repo_type="model",
    folder_path="outputs/hub-safe-benchmark",
    path_in_repo="benchmark/pilot-test",
    commit_message="Upload held-out pilot metrics and gold-free predictions",
)
PY

echo "Benchmark and gold-free result upload completed. The Job may now terminate."
