import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from manga_sft.dataset import (
    deterministic_group_split,
    iter_manga109_regions,
    materialize_archive,
    read_jsonl,
    safe_extract_zip,
    stable_sample_id,
    validate_no_leakage,
)
from scripts.prepare_dataset import inspect_source, prepare


def test_group_split_is_deterministic_and_disjoint():
    books = [f"book-{index}" for index in range(20)]
    first = deterministic_group_split(books, seed=42)
    second = deterministic_group_split(reversed(books), seed=42)
    assert first == second
    assert set(first) == set(books)
    assert set(first.values()) == {"train", "validation", "test"}


def test_stable_id_preserves_japanese():
    assert stable_sample_id("本", "1", [1, 2, 30, 40], "えっ…♡") == stable_sample_id(
        "本", "1", [1, 2, 30, 40], "えっ…♡"
    )


def test_xml_parsing_and_image_manifest(tmp_path: Path):
    xml = tmp_path / "book.xml"
    xml.write_text(
        '<book><page index="0"><text xmin="1" ymin="2" xmax="31" ymax="42" '
        'text="アナタ専用ウシ乳マヤでちゅよ〜"/></page></book>',
        encoding="utf-8",
    )
    assert list(iter_manga109_regions(xml)) == [
        ("0", [1, 2, 31, 42], "アナタ専用ウシ乳マヤでちゅよ〜")
    ]
    image = tmp_path / "crop.png"
    Image.new("RGB", (32, 16), "white").save(image)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"sample_id": "one", "image_path": "crop.png", "gold": "〜"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    rows = read_jsonl(manifest, verify_images=True)
    assert rows[0]["gold"] == "〜"
    assert Path(rows[0]["image_path"]) == image


def test_no_split_overlap():
    with pytest.raises(ValueError, match="sample_id leakage"):
        validate_no_leakage(
            {
                "train": [{"sample_id": "same", "book": "a"}],
                "validation": [{"sample_id": "same", "book": "b"}],
            }
        )


def test_no_book_overlap():
    with pytest.raises(ValueError, match="book leakage"):
        validate_no_leakage(
            {
                "train": [{"sample_id": "one", "book": "same-book"}],
                "test": [{"sample_id": "two", "book": "same-book"}],
            }
        )


def test_safe_zip_extraction_rejects_traversal(tmp_path: Path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "no")
    with pytest.raises(ValueError, match="Unsafe path"):
        safe_extract_zip(archive, tmp_path / "out")


def test_materialize_archive_creates_valid_regular_copy(tmp_path: Path):
    source = tmp_path / "mounted" / "source.zip"
    source.parent.mkdir()
    with zipfile.ZipFile(source, "w") as handle:
        handle.writestr("Manga109-s/annotations/book.xml", "<book />")

    result = materialize_archive(
        source, tmp_path / "cache" / "source.zip", chunk_size=7
    )

    assert result.path.is_file()
    assert result.path.read_bytes() == source.read_bytes()
    assert result.size_bytes == source.stat().st_size
    assert len(result.sha256) == 64
    with zipfile.ZipFile(result.path) as handle:
        assert handle.read("Manga109-s/annotations/book.xml") == b"<book />"


def test_materialize_archive_rejects_invalid_zip(tmp_path: Path):
    source = tmp_path / "not-a-zip.zip"
    source.write_bytes(b"not a zip archive")

    with pytest.raises(zipfile.BadZipFile, match="not a readable ZIP"):
        materialize_archive(source, tmp_path / "cache" / "not-a-zip.zip")


def test_end_to_end_crop_preparation(tmp_path: Path):
    source = tmp_path / "Manga109-s"
    for index in range(3):
        book = f"book-{index}"
        (source / "annotations").mkdir(parents=True, exist_ok=True)
        page_dir = source / "images" / book
        page_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (80, 60), (index * 70, 20, 30)).save(page_dir / "000.jpg")
        text = (
            '<text xmin="5" ymin="5" xmax="60" ymax="45" '
            f'text="台詞{index}"/>'
        )
        (source / "annotations" / f"{book}.xml").write_text(
            f'<book><page index="0">{text}{text}</page></book>',
            encoding="utf-8",
        )
    preflight = inspect_source(source, seed=42)
    assert preflight["status"] == "source_preflight_ok"
    assert preflight["sizes"] == {"train": 1, "validation": 1, "test": 1}
    assert preflight["filtering"] == {"duplicate_annotation": 3}
    assert preflight["sample_crops_loaded"] == 3
    assert preflight["materialized_crops"] is False

    output = tmp_path / "prepared"
    summary = prepare(source, output, seed=42)
    assert summary["sizes"] == {"train": 1, "validation": 1, "test": 1}
    assert summary["filtering"] == {"duplicate_annotation": 3}
    manifests = {
        split: read_jsonl(output / "manifests" / f"{split}.jsonl", verify_images=True)
        for split in ("train", "validation", "test")
    }
    validate_no_leakage(manifests)
