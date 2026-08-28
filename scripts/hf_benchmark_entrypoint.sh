#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN must be supplied as an encrypted Hugging Face Job secret}"
: "${HF_MODEL_REPO:?HF_MODEL_REPO must name the private trained model repository}"
: "${HF_MODEL_REVISION:?HF_MODEL_REVISION must pin the promoted model commit}"
: "${HF_MODEL_SHA256:?HF_MODEL_SHA256 must identify the promoted root model weights}"
: "${MANGA109_ROOT:=/data/manga109s}"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-baselines.txt
bash scripts/prepare_manga109s_for_job.sh materialize

python - <<'PY'
import json
import os
from pathlib import Path
from huggingface_hub import HfApi

summary = json.loads(
    Path("data/prepared/manifests/split_summary.json").read_text(encoding="utf-8")
)
test_manifest = Path("data/prepared/manifests/test.jsonl")
test_samples = sum(1 for line in test_manifest.open(encoding="utf-8") if line.strip())
if summary.get("seed") != 42 or summary.get("test_used_for_training") is not False:
    raise RuntimeError("The prepared dataset is not the established held-out split")
if summary.get("sizes", {}).get("test") != 11063 or test_samples != 11063:
    raise RuntimeError(
        f"Expected 11063 held-out samples, got summary={summary.get('sizes', {}).get('test')} "
        f"manifest={test_samples}"
    )

info = HfApi(token=os.environ["HF_TOKEN"]).model_info(
    os.environ["HF_MODEL_REPO"],
    revision=os.environ["HF_MODEL_REVISION"],
    files_metadata=True,
)
if info.sha != os.environ["HF_MODEL_REVISION"]:
    raise RuntimeError(f"Resolved model revision {info.sha} does not match the requested commit")
root = next((item for item in info.siblings if item.rfilename == "model.safetensors"), None)
if root is None or root.lfs is None:
    raise RuntimeError("The pinned model revision has no root model.safetensors LFS object")
actual_sha256 = root.lfs.get("sha256") if isinstance(root.lfs, dict) else root.lfs.sha256
if actual_sha256 != os.environ["HF_MODEL_SHA256"]:
    raise RuntimeError(
        f"Pinned root model SHA-256 {actual_sha256} does not match the promoted weights"
    )
print(json.dumps({
    "benchmark_test_samples": test_samples,
    "benchmark_model_repo": os.environ["HF_MODEL_REPO"],
    "benchmark_model_revision": info.sha,
    "benchmark_root_path": root.rfilename,
    "benchmark_root_sha256": actual_sha256,
}, sort_keys=True))
PY

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
  --revision "${HF_MODEL_REVISION}" \
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
