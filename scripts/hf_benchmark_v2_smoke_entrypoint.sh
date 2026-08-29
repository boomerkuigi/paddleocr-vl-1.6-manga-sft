#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN must be supplied as an encrypted Hugging Face Job secret}"
: "${HF_MODEL_REPO:?HF_MODEL_REPO must name the private V2 model repository}"
: "${HF_MODEL_REVISION:?HF_MODEL_REVISION must pin the V2 model commit}"
: "${HF_MODEL_SHA256:?HF_MODEL_SHA256 must identify the V2 root model weights}"
: "${HF_SMOKE_DATASET_REPO:?HF_SMOKE_DATASET_REPO must name the private smoke dataset}"
: "${HF_SMOKE_DATASET_REVISION:?HF_SMOKE_DATASET_REVISION must pin the smoke dataset}"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python - <<'PY'
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

api = HfApi(token=os.environ["HF_TOKEN"])
dataset_info = api.dataset_info(
    os.environ["HF_SMOKE_DATASET_REPO"],
    revision=os.environ["HF_SMOKE_DATASET_REVISION"],
)
if dataset_info.sha != os.environ["HF_SMOKE_DATASET_REVISION"] or not dataset_info.private:
    raise RuntimeError("Smoke dataset revision did not resolve exactly or is not private")
dataset_path = Path(snapshot_download(
    repo_id=os.environ["HF_SMOKE_DATASET_REPO"],
    repo_type="dataset",
    revision=os.environ["HF_SMOKE_DATASET_REVISION"],
    local_dir="/workspace/smoke-dataset",
    token=os.environ["HF_TOKEN"],
))
metadata = json.loads((dataset_path / "metadata.json").read_text(encoding="utf-8"))
manifest = dataset_path / "data" / "smoke.jsonl"
rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
if metadata.get("samples") != 100 or metadata.get("source_test_samples") != 11063:
    raise RuntimeError(f"Unexpected smoke dataset metadata: {metadata}")
if len(rows) != 100 or len({row["sample_id"] for row in rows}) != 100:
    raise RuntimeError("Smoke manifest must contain exactly 100 unique samples")
for row in rows:
    if row.get("original_split") != "test" or not (manifest.parent / row["image_path"]).is_file():
        raise RuntimeError(f"Invalid smoke row: {row.get('sample_id')}")

model_info = api.model_info(
    os.environ["HF_MODEL_REPO"],
    revision=os.environ["HF_MODEL_REVISION"],
    files_metadata=True,
)
root = next((item for item in model_info.siblings if item.rfilename == "model.safetensors"), None)
actual_sha = (
    root.lfs.get("sha256") if root is not None and isinstance(root.lfs, dict)
    else root.lfs.sha256 if root is not None and root.lfs is not None
    else None
)
if (
    model_info.sha != os.environ["HF_MODEL_REVISION"]
    or not bool(getattr(model_info, "private", False))
    or actual_sha != os.environ["HF_MODEL_SHA256"]
):
    raise RuntimeError("Promoted V2 model revision, privacy, or root weight hash mismatch")
print(json.dumps({
    "smoke_dataset_repo": os.environ["HF_SMOKE_DATASET_REPO"],
    "smoke_dataset_revision": dataset_info.sha,
    "smoke_samples": len(rows),
    "source_test_samples": metadata["source_test_samples"],
    "model_repo": os.environ["HF_MODEL_REPO"],
    "model_revision": model_info.sha,
    "model_root_sha256": actual_sha,
}, sort_keys=True))
PY

mkdir -p outputs/predictions
python scripts/evaluate_baselines.py \
  --manifest /workspace/smoke-dataset/data/smoke.jsonl \
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
  --output outputs/hub-safe-v2-smoke

python - <<'PY'
import json
import os
from pathlib import Path

metrics = json.loads(Path("outputs/evaluation/metrics.json").read_text(encoding="utf-8"))
raw = metrics.get("new_model", {}).get("raw", {})
if raw.get("samples") != 100:
    raise RuntimeError(f"V2 smoke produced an unexpected sample count: {raw.get('samples')}")
metadata = {
    "model_id": os.environ["HF_MODEL_REPO"],
    "model_revision": os.environ["HF_MODEL_REVISION"],
    "root_model_sha256": os.environ["HF_MODEL_SHA256"],
    "samples": raw["samples"],
    "raw_metrics": raw,
    "method": "evaluate_baselines.py:new_model followed by evaluate.py",
    "source_test_samples": 11063,
}
Path("outputs/hub-safe-v2-smoke/benchmark_metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print("V2_SMOKE_SUMMARY=" + json.dumps(metadata, ensure_ascii=False, sort_keys=True))
PY

python - <<'PY'
import os
from huggingface_hub import HfApi

commit = HfApi(token=os.environ["HF_TOKEN"]).upload_folder(
    repo_id=os.environ["HF_MODEL_REPO"],
    repo_type="model",
    folder_path="outputs/hub-safe-v2-smoke",
    path_in_repo="benchmark/v2-only-smoke-100",
    commit_message="Upload V2-only 100-sample benchmark smoke metrics",
)
print(f"V2_SMOKE_RESULTS_COMMIT={commit.oid}")
PY

echo "V2-only benchmark smoke and gold-free result upload completed. The Job may now terminate."
