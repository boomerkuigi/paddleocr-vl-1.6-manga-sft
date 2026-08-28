#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from manga_sft.dataset import read_jsonl


SELECTION_METHOD = "lowest-sha256(seed|sample_id), version manga109s-test-smoke-v1"


def selection_digest(sample_id: str, seed: int) -> str:
    payload = f"manga109s-test-smoke-v1|{seed}|{sample_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def select_rows(rows: list[dict], count: int, seed: int) -> list[tuple[int, dict]]:
    if count <= 0 or count > len(rows):
        raise ValueError(f"count must be between 1 and {len(rows)}")
    indexed = list(enumerate(rows))
    ranked = sorted(
        indexed,
        key=lambda item: (selection_digest(str(item[1]["sample_id"]), seed), item[0]),
    )
    return ranked[:count]


def create_dataset(
    manifest: Path,
    output: Path,
    count: int,
    seed: int,
    source_dataset_repo: str,
    source_dataset_revision: str,
    source_split_seed: int,
) -> dict:
    rows = read_jsonl(manifest, verify_images=True)
    selected = select_rows(rows, count, seed)
    data_dir = output / "data"
    image_dir = data_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for rank, (manifest_index, row) in enumerate(selected, start=1):
        if row.get("split") != "test":
            raise ValueError(f"Smoke candidate {row['sample_id']} is not in the test split")
        source_image = Path(row["image_path"])
        destination = image_dir / f"{row['sample_id']}.png"
        shutil.copy2(source_image, destination)
        copied = dict(row)
        copied.update(
            {
                "image_path": f"images/{destination.name}",
                "original_split": "test",
                "original_test_manifest_index": manifest_index,
                "original_test_split_identity": f"seed-{source_split_seed}:test:{row['sample_id']}",
                "source_dataset_repo": source_dataset_repo,
                "source_dataset_revision": source_dataset_revision,
                "source_split_seed": source_split_seed,
                "smoke_selection_seed": seed,
                "smoke_selection_method": SELECTION_METHOD,
                "smoke_selection_rank": rank,
            }
        )
        output_rows.append(copied)

    manifest_path = data_dir / "smoke.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    metadata = {
        "dataset": "Manga109-s held-out OCR benchmark smoke subset",
        "samples": len(output_rows),
        "source_test_samples": len(rows),
        "source_dataset_repo": source_dataset_repo,
        "source_dataset_revision": source_dataset_revision,
        "source_split": "test",
        "source_split_seed": source_split_seed,
        "source_split_method": "book-grouped deterministic 80/10/10",
        "selection_seed": seed,
        "selection_method": SELECTION_METHOD,
        "contains_gold_transcriptions": True,
        "visibility_required": "private",
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# Private Manga109-s benchmark smoke subset\n\n"
        "Deterministic 100-sample subset of the repository's seed-42 held-out test split. "
        "Contains Manga109-s crops and gold transcriptions and must remain private. "
        f"Selection: `{SELECTION_METHOD}`.\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--source-dataset-repo", required=True)
    parser.add_argument("--source-dataset-revision", required=True)
    parser.add_argument("--source-split-seed", type=int, default=42)
    args = parser.parse_args()
    print(
        json.dumps(
            create_dataset(
                args.manifest,
                args.output,
                args.count,
                args.selection_seed,
                args.source_dataset_repo,
                args.source_dataset_revision,
                args.source_split_seed,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
