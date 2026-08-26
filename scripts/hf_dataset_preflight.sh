#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN must be supplied as an encrypted Hugging Face Job secret}"
: "${MANGA109_ROOT:=/data/manga109s}"

python -m pip install --upgrade pip
python -m pip install --no-deps -e .
python -m pip install -c constraints.txt Pillow huggingface-hub

python - <<'PY'
import os
from huggingface_hub import HfApi

HfApi(token=os.environ["HF_TOKEN"]).repo_info(
    "hal-utokyo/Manga109-s", repo_type="dataset"
)
print("Authenticated token can access the gated Manga109-s repository.")
PY

bash scripts/prepare_manga109s_for_job.sh preflight
echo "Dataset mount, archive extraction, source layout, split counts, and sample crop loading passed."
