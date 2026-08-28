#!/usr/bin/env python3
"""Build the private smoke subset without materializing the full crop archive."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from manga_sft.dataset import (
    deterministic_group_split,
    find_image,
    image_sha256,
    iter_manga109_regions,
    stable_sample_id,
)
from scripts.create_benchmark_smoke_dataset import SELECTION_METHOD, selection_digest
from scripts.prepare_dataset import locate_roots


def collect_test_rows(root: Path, seed: int) -> list[dict]:
    annotations_root, images_root = locate_roots(root)
    xml_files = sorted(annotations_root.rglob("*.xml"))
    book_splits = deterministic_group_split((path.stem for path in xml_files), seed=seed)
    seen_regions: set[tuple[str, str, tuple[int, ...], str]] = set()
    dimensions: dict[Path, tuple[int, int]] = {}
    rows: list[dict] = []

    for xml_path in xml_files:
        book = xml_path.stem
        if book_splits[book] != "test":
            continue
        for page, bbox, gold in iter_manga109_regions(xml_path):
            source_image = find_image(images_root, book, page)
            page_dimensions = dimensions.get(source_image)
            if page_dimensions is None:
                with Image.open(source_image) as image:
                    page_dimensions = image.size
                dimensions[source_image] = page_dimensions
            width, height = page_dimensions
            clipped = [
                max(0, min(width, bbox[0])),
                max(0, min(height, bbox[1])),
                max(0, min(width, bbox[2])),
                max(0, min(height, bbox[3])),
            ]
            if clipped[2] - clipped[0] < 10 or clipped[3] - clipped[1] < 10:
                continue
            region_key = (book, str(page), tuple(clipped), gold)
            if region_key in seen_regions:
                continue
            seen_regions.add(region_key)
            rows.append(
                {
                    "sample_id": stable_sample_id(book, page, clipped, gold),
                    "gold": gold,
                    "book": book,
                    "page": str(page),
                    "bbox": clipped,
                    "split": "test",
                    "_source_image": source_image,
                }
            )
    return sorted(rows, key=lambda row: row["sample_id"])


def create_dataset(
    manga109_root: Path,
    output: Path,
    count: int,
    selection_seed: int,
    source_dataset_repo: str,
    source_dataset_revision: str,
    source_split_seed: int,
) -> dict:
    rows = collect_test_rows(manga109_root, source_split_seed)
    if count <= 0 or count > len(rows):
        raise ValueError(f"count must be between 1 and {len(rows)}")
    selected = sorted(
        enumerate(rows),
        key=lambda item: (selection_digest(item[1]["sample_id"], selection_seed), item[0]),
    )[:count]
    image_dir = output / "data" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict] = []
    for rank, (manifest_index, source_row) in enumerate(selected, start=1):
        destination = image_dir / f"{source_row['sample_id']}.png"
        with Image.open(source_row["_source_image"]) as image:
            image.convert("RGB").crop(tuple(source_row["bbox"])).save(
                destination, format="PNG", optimize=True
            )
        row = {key: value for key, value in source_row.items() if key != "_source_image"}
        row.update(
            {
                "image_path": f"images/{destination.name}",
                "image_sha256": image_sha256(destination),
                "original_split": "test",
                "original_test_manifest_index": manifest_index,
                "original_test_split_identity": (
                    f"seed-{source_split_seed}:test:{source_row['sample_id']}"
                ),
                "source_dataset_repo": source_dataset_repo,
                "source_dataset_revision": source_dataset_revision,
                "source_split_seed": source_split_seed,
                "smoke_selection_seed": selection_seed,
                "smoke_selection_method": SELECTION_METHOD,
                "smoke_selection_rank": rank,
            }
        )
        manifest_rows.append(row)

    data_dir = output / "data"
    (data_dir / "smoke.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    metadata = {
        "dataset": "Manga109-s held-out OCR benchmark smoke subset",
        "samples": len(manifest_rows),
        "source_test_samples": len(rows),
        "source_dataset_repo": source_dataset_repo,
        "source_dataset_revision": source_dataset_revision,
        "source_split": "test",
        "source_split_seed": source_split_seed,
        "source_split_method": "book-grouped deterministic 80/10/10",
        "selection_seed": selection_seed,
        "selection_method": SELECTION_METHOD,
        "contains_gold_transcriptions": True,
        "visibility_required": "private",
        "full_crop_archive_materialized": False,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# Private Manga109-s benchmark smoke subset\n\n"
        "Deterministic held-out subset. Contains Manga109-s crops and gold transcriptions "
        "and must remain private.\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manga109-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--source-dataset-repo", required=True)
    parser.add_argument("--source-dataset-revision", required=True)
    parser.add_argument("--source-split-seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(create_dataset(
        args.manga109_root,
        args.output,
        args.count,
        args.selection_seed,
        args.source_dataset_repo,
        args.source_dataset_revision,
        args.source_split_seed,
    ), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
