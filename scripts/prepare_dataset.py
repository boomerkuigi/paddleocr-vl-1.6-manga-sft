#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from manga_sft.dataset import (
    Sample,
    deterministic_group_split,
    find_image,
    image_sha256,
    iter_manga109_regions,
    stable_sample_id,
    validate_no_leakage,
    write_jsonl,
)


def locate_roots(root: Path) -> tuple[Path, Path]:
    annotation_candidates = [root / "annotations", root / "Annotations", root / "annotation"]
    image_candidates = [root / "images", root / "Images", root / "image"]
    annotations = next((path for path in annotation_candidates if path.is_dir()), None)
    images = next((path for path in image_candidates if path.is_dir()), None)
    if not annotations or not images:
        for candidate in sorted(path for path in root.rglob("*") if path.is_dir()):
            nested_annotations = next(
                (candidate / name for name in ("annotations", "Annotations", "annotation") if (candidate / name).is_dir()),
                None,
            )
            nested_images = next(
                (candidate / name for name in ("images", "Images", "image") if (candidate / name).is_dir()),
                None,
            )
            if nested_annotations and nested_images:
                annotations, images = nested_annotations, nested_images
                break
    if not annotations or not images:
        raise FileNotFoundError(
            "Expected a common parent containing Manga109-s annotations/{book}.xml "
            f"and images/{book}/{page}.jpg beneath {root}"
        )
    return annotations, images


def prepare(root: Path, output: Path, seed: int) -> dict:
    annotations_root, images_root = locate_roots(root)
    xml_files = sorted(annotations_root.rglob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No Manga109-s XML files below {annotations_root}")
    book_splits = deterministic_group_split((path.stem for path in xml_files), seed=seed)
    crop_root = output / "crops"
    manifest_root = output / "manifests"
    rows: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    filtered = Counter()

    for xml_path in xml_files:
        book = xml_path.stem
        split = book_splits[book]
        for page, bbox, gold in iter_manga109_regions(xml_path):
            page_path = find_image(images_root, book, page)
            with Image.open(page_path) as source:
                width, height = source.size
                clipped = [
                    max(0, min(width, bbox[0])),
                    max(0, min(height, bbox[1])),
                    max(0, min(width, bbox[2])),
                    max(0, min(height, bbox[3])),
                ]
                if clipped[2] - clipped[0] < 10 or clipped[3] - clipped[1] < 10:
                    filtered["invalid_after_clipping"] += 1
                    continue
                sample_id = stable_sample_id(book, page, clipped, gold)
                crop_path = crop_root / split / book / f"{sample_id}.png"
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                source.convert("RGB").crop(tuple(clipped)).save(crop_path, format="PNG", optimize=True)
            relative = Path("..") / "crops" / split / book / crop_path.name
            sample = Sample(
                sample_id=sample_id,
                image_path=str(relative),
                gold=gold,
                book=book,
                page=str(page),
                bbox=clipped,
                split=split,
                image_sha256=image_sha256(crop_path),
            )
            rows[split].append(asdict(sample))

    for split in rows:
        rows[split].sort(key=lambda item: item["sample_id"])
    validate_no_leakage(rows)
    sizes = {split: write_jsonl(manifest_root / f"{split}.jsonl", items) for split, items in rows.items()}
    summary = {
        "source": "Manga109-s",
        "split_method": "book-grouped deterministic 80/10/10",
        "seed": seed,
        "sizes": sizes,
        "books": {
            split: sorted(book for book, assigned in book_splits.items() if assigned == split)
            for split in rows
        },
        "filtering": dict(filtered),
        "normalization": "preserve",
        "test_used_for_training": False,
    }
    (manifest_root / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def inspect_source(root: Path, seed: int) -> dict:
    """Validate the real source layout and count usable regions without writing crops."""
    annotations_root, images_root = locate_roots(root)
    xml_files = sorted(annotations_root.rglob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No Manga109-s XML files below {annotations_root}")
    book_splits = deterministic_group_split((path.stem for path in xml_files), seed=seed)
    sizes: Counter[str] = Counter()
    filtered = Counter()
    page_paths: dict[tuple[str, str], Path] = {}
    image_sizes: dict[Path, tuple[int, int]] = {}
    crop_checks: set[str] = set()

    for xml_path in xml_files:
        book = xml_path.stem
        split = book_splits[book]
        for page, bbox, _gold in iter_manga109_regions(xml_path):
            key = (book, page)
            page_path = page_paths.get(key)
            if page_path is None:
                page_path = find_image(images_root, book, page)
                page_paths[key] = page_path
            dimensions = image_sizes.get(page_path)
            if dimensions is None:
                with Image.open(page_path) as source:
                    dimensions = source.size
                image_sizes[page_path] = dimensions
            width, height = dimensions
            clipped = [
                max(0, min(width, bbox[0])),
                max(0, min(height, bbox[1])),
                max(0, min(width, bbox[2])),
                max(0, min(height, bbox[3])),
            ]
            if clipped[2] - clipped[0] < 10 or clipped[3] - clipped[1] < 10:
                filtered["invalid_after_clipping"] += 1
                continue
            sizes[split] += 1
            if split not in crop_checks:
                with Image.open(page_path) as source:
                    crop = source.convert("RGB").crop(tuple(clipped))
                    crop.load()
                crop_checks.add(split)

    missing_splits = {"train", "validation", "test"} - crop_checks
    if missing_splits:
        raise ValueError(f"No usable crop could be loaded for splits: {sorted(missing_splits)}")
    return {
        "status": "source_preflight_ok",
        "source": "Manga109-s",
        "split_method": "book-grouped deterministic 80/10/10",
        "seed": seed,
        "xml_books": len(xml_files),
        "referenced_pages": len(page_paths),
        "sizes": {split: sizes[split] for split in ("train", "validation", "test")},
        "book_counts": {
            split: sum(1 for assigned in book_splits.values() if assigned == split)
            for split in ("train", "validation", "test")
        },
        "filtering": dict(filtered),
        "sample_crops_loaded": len(crop_checks),
        "materialized_crops": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create non-redistributable manga text crops")
    parser.add_argument("--manga109-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/prepared"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate the real archive layout and count usable regions without writing crops",
    )
    args = parser.parse_args()
    if args.preflight_only:
        summary = inspect_source(args.manga109_root.resolve(), args.seed)
    else:
        summary = prepare(args.manga109_root.resolve(), args.output.resolve(), args.seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
