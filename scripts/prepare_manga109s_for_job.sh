#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-materialize}"
: "${MANGA109_ROOT:=/data/manga109s}"
: "${MANGA109_ARCHIVE_CACHE:=/workspace/data/manga109s-archive-cache}"

if [[ "${MODE}" != "materialize" && "${MODE}" != "preflight" ]]; then
  echo "Usage: $0 [materialize|preflight]" >&2
  exit 2
fi

SOURCE_ROOT="${MANGA109_ROOT}"
annotations_dir="$(find "${SOURCE_ROOT}" -maxdepth 4 -type d \
  \( -name annotations -o -name Annotations -o -name annotation \) -print -quit)"
images_dir="$(find "${SOURCE_ROOT}" -maxdepth 4 -type d \
  \( -name images -o -name Images -o -name image \) -print -quit)"
if [[ -z "${annotations_dir}" || -z "${images_dir}" ]]; then
  mapfile -t manga_archives < <(find "${SOURCE_ROOT}" -maxdepth 2 -type f -name '*.zip' -print)
  if [[ "${#manga_archives[@]}" -ne 1 ]]; then
    echo "Expected extracted Manga109-s directories or exactly one official zip in ${SOURCE_ROOT}" >&2
    exit 2
  fi
  python scripts/extract_manga109s.py \
    --archive "${manga_archives[0]}" \
    --stage-directory "${MANGA109_ARCHIVE_CACHE}" \
    --output /workspace/data/manga109s-extracted
  SOURCE_ROOT=/workspace/data/manga109s-extracted
fi

if [[ "${MODE}" == "preflight" ]]; then
  python scripts/prepare_dataset.py \
    --manga109-root "${SOURCE_ROOT}" \
    --output data/prepared \
    --seed 42 \
    --preflight-only
elif [[ ! -f data/prepared/manifests/train.jsonl ]]; then
  python scripts/prepare_dataset.py \
    --manga109-root "${SOURCE_ROOT}" \
    --output data/prepared \
    --seed 42
fi
