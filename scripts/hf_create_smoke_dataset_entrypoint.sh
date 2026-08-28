#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN must be supplied as an encrypted Hugging Face Job secret}"
: "${HF_SMOKE_DATASET_REPO:?HF_SMOKE_DATASET_REPO must name the private destination dataset}"
: "${MANGA109_SOURCE_REVISION:?MANGA109_SOURCE_REVISION must pin the gated source dataset}"
: "${MANGA109_ROOT:=/data/manga109s}"

export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

python -m pip install --upgrade pip
python -m pip install Pillow==12.3.0 huggingface-hub==1.28.0
bash scripts/prepare_manga109s_for_job.sh materialize
python scripts/create_benchmark_smoke_dataset.py \
  --manifest data/prepared/manifests/test.jsonl \
  --output outputs/private-smoke-dataset \
  --count 100 \
  --selection-seed 42 \
  --source-dataset-repo hal-utokyo/Manga109-s \
  --source-dataset-revision "${MANGA109_SOURCE_REVISION}" \
  --source-split-seed 42

python - <<'PY'
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
repo_id = os.environ["HF_SMOKE_DATASET_REPO"]
api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
info = api.dataset_info(repo_id)
if not info.private:
    raise RuntimeError(f"Refusing to upload Manga109-s smoke data because {repo_id} is public")
commit = api.upload_folder(
    repo_id=repo_id,
    repo_type="dataset",
    folder_path="outputs/private-smoke-dataset",
    commit_message="Create deterministic Manga109-s held-out smoke subset",
)
print(f"SMOKE_DATASET_COMMIT={commit.oid}")
PY
