import json

from PIL import Image

from scripts.create_benchmark_smoke_dataset import (
    SELECTION_METHOD,
    create_dataset,
    select_rows,
)


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
