#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/pilot.yaml}"
: "${HF_TOKEN:?HF_TOKEN must be supplied as an encrypted Hugging Face Job secret}"
: "${MANGA109_ROOT:=/data/manga109s}"
: "${PUSH_TO_HUB:=1}"

if [[ "${PUSH_TO_HUB}" == "1" ]]; then
  : "${HF_MODEL_REPO:?HF_MODEL_REPO must name the private destination model repository}"
elif [[ "${PUSH_TO_HUB}" != "0" ]]; then
  echo "PUSH_TO_HUB must be 0 or 1" >&2
  exit 2
fi

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
bash scripts/prepare_manga109s_for_job.sh materialize

python scripts/validate_environment.py --config "${CONFIG_PATH}" --load-processor
train_args=(--config "${CONFIG_PATH}")
if [[ "${PUSH_TO_HUB}" == "1" ]]; then
  train_args+=(--push-to-hub)
fi
python scripts/train.py "${train_args[@]}"
if [[ "${PUSH_TO_HUB}" == "1" ]]; then
  echo "Training and final model upload completed. The Job may now terminate."
else
  echo "Training smoke completed without a Hub upload. The Job may now terminate."
fi
