#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN must be supplied as an encrypted Hugging Face Job secret}"
: "${HF_MODEL_REPO:?HF_MODEL_REPO must name the private V2 model repository}"
: "${HF_MODEL_REVISION:?HF_MODEL_REVISION must pin the V2 model commit}"
: "${HF_MODEL_SHA256:?HF_MODEL_SHA256 must identify the V2 root model weights}"
: "${MANGA109_ROOT:=/data/manga109s}"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
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

api = HfApi(token=os.environ["HF_TOKEN"])
info = api.model_info(
    os.environ["HF_MODEL_REPO"],
    revision=os.environ["HF_MODEL_REVISION"],
    files_metadata=True,
)
if info.sha != os.environ["HF_MODEL_REVISION"]:
    raise RuntimeError("The V2 model revision did not resolve to the requested immutable commit")
if not bool(getattr(info, "private", False)):
    raise RuntimeError("The V2 benchmark destination must remain private")
root = next((item for item in info.siblings if item.rfilename == "model.safetensors"), None)
if root is None or root.lfs is None:
    raise RuntimeError("The pinned V2 model revision has no root model.safetensors LFS object")
actual_sha256 = root.lfs.get("sha256") if isinstance(root.lfs, dict) else root.lfs.sha256
if actual_sha256 != os.environ["HF_MODEL_SHA256"]:
    raise RuntimeError("The pinned V2 root model SHA-256 does not match the expected value")
print(json.dumps({
    "benchmark_test_samples": test_samples,
    "benchmark_model_repo": os.environ["HF_MODEL_REPO"],
    "benchmark_model_revision": info.sha,
    "benchmark_root_path": root.rfilename,
    "benchmark_root_sha256": actual_sha256,
}, sort_keys=True))
PY

mkdir -p outputs/predictions
python scripts/evaluate_baselines.py \
  --manifest data/prepared/manifests/test.jsonl \
  --model new_model \
  --model-id "${HF_MODEL_REPO}" \
  --revision "${HF_MODEL_REVISION}" \
  --output outputs/predictions/new_model.jsonl
python scripts/evaluate.py \
  outputs/predictions/new_model.jsonl \
  --output-dir outputs/evaluation
python scripts/sanitize_results_for_hub.py \
  --prediction-dir outputs/predictions \
  --evaluation-dir outputs/evaluation \
  --output outputs/hub-safe-v2-benchmark

python - <<'PY'
import json
import os
from pathlib import Path

metrics = json.loads(Path("outputs/evaluation/metrics.json").read_text(encoding="utf-8"))
raw = metrics.get("new_model", {}).get("raw", {})
if raw.get("samples") != 11063:
    raise RuntimeError(f"V2 benchmark produced an unexpected sample count: {raw.get('samples')}")
metadata = {
    "model_id": os.environ["HF_MODEL_REPO"],
    "model_revision": os.environ["HF_MODEL_REVISION"],
    "root_model_sha256": os.environ["HF_MODEL_SHA256"],
    "samples": raw["samples"],
    "raw_metrics": raw,
    "method": "evaluate_baselines.py:new_model followed by evaluate.py",
    "test_manifest": "Manga109-s seed-42 book-grouped held-out test split",
}
Path("outputs/hub-safe-v2-benchmark/benchmark_metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print("V2_BENCHMARK_SUMMARY=" + json.dumps(metadata, ensure_ascii=False, sort_keys=True))
PY

python - <<'PY'
import os
from huggingface_hub import HfApi

commit = HfApi(token=os.environ["HF_TOKEN"]).upload_folder(
    repo_id=os.environ["HF_MODEL_REPO"],
    repo_type="model",
    folder_path="outputs/hub-safe-v2-benchmark",
    path_in_repo="benchmark/v2-only-test",
    commit_message="Upload V2-only held-out benchmark metrics and gold-free predictions",
)
print(f"V2_BENCHMARK_RESULTS_COMMIT={commit.oid}")
PY

echo "V2-only benchmark and gold-free result upload completed. The Job may now terminate."
