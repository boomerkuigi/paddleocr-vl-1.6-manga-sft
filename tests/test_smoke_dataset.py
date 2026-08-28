import json
from pathlib import Path

from PIL import Image

from scripts.create_benchmark_smoke_dataset import (
    SELECTION_METHOD,
    create_dataset,
    select_rows,
)
from scripts.create_benchmark_smoke_dataset_from_manga109 import create_dataset as create_from_source
from manga_sft.dataset import deterministic_group_split


def test_smoke_selection_is_deterministic_and_order_independent():
    rows = [{"sample_id": f"sample-{index}"} for index in range(150)]
    first = [row["sample_id"] for _, row in select_rows(rows, 100, 42)]
    second = [row["sample_id"] for _, row in select_rows(list(reversed(rows)), 100, 42)]
    assert first == second
    assert len(first) == len(set(first)) == 100


def test_private_smoke_dataset_preserves_identity_and_provenance(tmp_path):
    manifest_dir = tmp_path / "prepared" / "manifests"
    crop_dir = tmp_path / "prepared" / "crops" / "test" / "book"
    manifest_dir.mkdir(parents=True)
    crop_dir.mkdir(parents=True)
    manifest = manifest_dir / "test.jsonl"
    rows = []
    for index in range(3):
        sample_id = f"sample-{index}"
        image = crop_dir / f"{sample_id}.png"
        Image.new("RGB", (12, 12), color=(index, index, index)).save(image)
        rows.append(
            {
                "sample_id": sample_id,
                "image_path": f"../crops/test/book/{image.name}",
                "gold": f"gold-{index}",
                "book": "book",
                "page": str(index),
                "bbox": [0, 0, 12, 12],
                "split": "test",
            }
        )
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    output = tmp_path / "smoke"
    metadata = create_dataset(
        manifest,
        output,
        count=2,
        seed=42,
        source_dataset_repo="source/repo",
        source_dataset_revision="source-sha",
        source_split_seed=42,
    )

    smoke_rows = [
        json.loads(line)
        for line in (output / "data" / "smoke.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert metadata["samples"] == 2
    assert metadata["source_test_samples"] == 3
    assert len(smoke_rows) == 2
    for row in smoke_rows:
        assert row["gold"].startswith("gold-")
        assert row["book"] == "book"
        assert row["page"] in {"0", "1", "2"}
        assert row["original_split"] == "test"
        assert row["original_test_split_identity"].endswith(row["sample_id"])
        assert row["source_dataset_revision"] == "source-sha"
        assert row["smoke_selection_method"] == SELECTION_METHOD
        assert (output / "data" / row["image_path"]).is_file()


def test_smoke_dataset_job_entrypoint_can_import_the_local_package():
    entrypoint = Path("scripts/hf_create_smoke_dataset_entrypoint.sh").read_text(
        encoding="utf-8"
    )
    assert 'export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"' in entrypoint


def test_source_smoke_builder_crops_only_selected_held_out_samples(tmp_path):
    root = tmp_path / "Manga109-s"
    annotations = root / "annotations"
    images = root / "images"
    books = ["book-a", "book-b", "book-c"]
    for index, book in enumerate(books):
        (images / book).mkdir(parents=True)
        annotations.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 24), color=(index, 0, 0)).save(images / book / "001.jpg")
        (annotations / f"{book}.xml").write_text(
            '<root><page index="1"><text xmin="1" ymin="2" xmax="22" ymax="20" '
            f'text="gold-{book}" /></page></root>',
            encoding="utf-8",
        )
    test_book = next(
        book for book, split in deterministic_group_split(books, seed=42).items() if split == "test"
    )
    output = tmp_path / "smoke"
    metadata = create_from_source(
        root, output, 1, 42, "source/repo", "source-sha", 42
    )
    row = json.loads((output / "data" / "smoke.jsonl").read_text(encoding="utf-8"))
    assert metadata["source_test_samples"] == 1
    assert metadata["full_crop_archive_materialized"] is False
    assert row["book"] == test_book
    assert row["gold"] == f"gold-{test_book}"
    assert row["original_test_manifest_index"] == 0
    assert (output / "data" / row["image_path"]).is_file()
