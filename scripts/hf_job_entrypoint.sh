#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/pilot.yaml}"
: "${HF_TOKEN:?HF_TOKEN must be supplied as an encrypted Hugging Face Job secret}"
: "${MANGA109_ROOT:=/data/manga109s}"
: "${PUSH_TO_HUB:=1}"

if [[ "${PUSH_TO_HUB}" != "0" && "${PUSH_TO_HUB}" != "1" ]]; then
  echo "PUSH_TO_HUB must be 0 or 1" >&2
  exit 2
fi

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

entrypoint_started_ns="$(date +%s%N)"
record_phase() {
  python - "$1" "$2" "$3" <<'PY'
import json
import sys

name, started, finished = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
print("JOB_PHASE_TIMING=" + json.dumps({"phase": name, "seconds": (finished - started) / 1e9}))
PY
}

phase_started_ns="$(date +%s%N)"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
phase_finished_ns="$(date +%s%N)"
record_phase dependency_install "${phase_started_ns}" "${phase_finished_ns}"

# The config, rather than a hard-coded legacy name, owns the Hub destination.
# Resolve only after requirements are installed because load_config requires
# PyYAML. hub_model_id_env validates the name for Bash indirect expansion.
HUB_MODEL_ID_ENV="$(PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}" python - "${CONFIG_PATH}" <<'PY'
import sys

from manga_sft.config import hub_model_id_env, load_config

print(hub_model_id_env(load_config(sys.argv[1]).get("hub", {})))
PY
)"
if [[ "${PUSH_TO_HUB}" == "1" && -z "${!HUB_MODEL_ID_ENV:-}" ]]; then
  echo "${HUB_MODEL_ID_ENV} must name the private destination model repository" >&2
  exit 2
fi

phase_started_ns="$(date +%s%N)"
bash scripts/prepare_manga109s_for_job.sh materialize
phase_finished_ns="$(date +%s%N)"
record_phase dataset_preparation "${phase_started_ns}" "${phase_finished_ns}"

phase_started_ns="$(date +%s%N)"
python scripts/validate_environment.py --config "${CONFIG_PATH}" --load-processor
phase_finished_ns="$(date +%s%N)"
record_phase environment_validation "${phase_started_ns}" "${phase_finished_ns}"

train_args=(--config "${CONFIG_PATH}")
if [[ "${PUSH_TO_HUB}" == "1" ]]; then
  train_args+=(--push-to-hub)
fi
phase_started_ns="$(date +%s%N)"
python scripts/train.py "${train_args[@]}"
phase_finished_ns="$(date +%s%N)"
record_phase training_command "${phase_started_ns}" "${phase_finished_ns}"
record_phase entrypoint_total "${entrypoint_started_ns}" "${phase_finished_ns}"
if [[ "${PUSH_TO_HUB}" == "1" ]]; then
  echo "Training and final model upload completed. The Job may now terminate."
else
  echo "Training run completed without a Hub upload. The Job may now terminate."
fi
