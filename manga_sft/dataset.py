from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Iterator
import xml.etree.ElementTree as ET
import zipfile

from PIL import Image

from .normalization import training_target


@dataclass(frozen=True)
class Sample:
    sample_id: str
    image_path: str
    gold: str
    book: str
    page: str
    bbox: list[int]
    split: str
    source: str = "Manga109-s"
    image_sha256: str | None = None


def safe_extract_zip(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"Unsafe path in archive: {member.filename}")
        handle.extractall(destination)


def stable_sample_id(book: str, page: str, bbox: Iterable[int], gold: str) -> str:
    payload = json.dumps(
        [book, str(page), list(bbox), gold], ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def image_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_group_split(
    groups: Iterable[str], seed: int = 42, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> dict[str, str]:
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-9 or any(r < 0 for r in ratios):
        raise ValueError("ratios must contain three non-negative values summing to one")
    unique = sorted(set(groups))
    random.Random(seed).shuffle(unique)
    n_groups = len(unique)
    n_train = round(n_groups * ratios[0])
    n_validation = round(n_groups * ratios[1])
    if n_groups >= 3:
        n_train = min(max(1, n_train), n_groups - 2)
        n_validation = min(max(1, n_validation), n_groups - n_train - 1)
    mapping: dict[str, str] = {}
    for index, group in enumerate(unique):
        if index < n_train:
            split = "train"
        elif index < n_train + n_validation:
            split = "validation"
        else:
            split = "test"
        mapping[group] = split
    return mapping


def find_image(images_root: Path, book: str, page_index: str) -> Path:
    candidates = []
    numeric = int(page_index)
    for extension in ("jpg", "jpeg", "png", "webp"):
        candidates.extend(
            [
                images_root / book / f"{numeric:03d}.{extension}",
                images_root / book / f"{numeric}.{extension}",
                images_root / f"{book}_{numeric:03d}.{extension}",
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No page image for book={book!r}, page={page_index!r}")


def iter_manga109_regions(xml_path: Path) -> Iterator[tuple[str, list[int], str]]:
    root = ET.parse(xml_path).getroot()
    for page in root.iter("page"):
        page_index = page.get("index") or page.get("id") or "0"
        for text in page.iter("text"):
            raw = text.get("text")
            if raw is None:
                raw = text.text or ""
            gold = training_target(raw)
            if not gold:
                continue
            try:
                bbox = [int(float(text.attrib[name])) for name in ("xmin", "ymin", "xmax", "ymax")]
            except KeyError:
                continue
            if bbox[2] - bbox[0] < 10 or bbox[3] - bbox[1] < 10:
                continue
            yield str(page_index), bbox, gold


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path, verify_images: bool = False) -> list[dict]:
    manifest_path = Path(path)
    rows: list[dict] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            for required in ("sample_id", "image_path", "gold"):
                if required not in row:
                    raise ValueError(f"{manifest_path}:{line_number} missing {required}")
            image_path = Path(row["image_path"])
            if not image_path.is_absolute():
                image_path = (manifest_path.parent / image_path).resolve()
            row["image_path"] = str(image_path)
            if verify_images:
                with Image.open(image_path) as image:
                    image.verify()
            rows.append(row)
    return rows


def validate_no_leakage(manifests: dict[str, list[dict]]) -> None:
    seen_ids: dict[str, str] = {}
    seen_hashes: dict[str, str] = {}
    seen_books: dict[str, str] = {}
    for split, rows in manifests.items():
        for row in rows:
            sample_id = str(row["sample_id"])
            if sample_id in seen_ids:
                raise ValueError(f"sample_id leakage: {sample_id} in {seen_ids[sample_id]} and {split}")
            seen_ids[sample_id] = split
            digest = row.get("image_sha256")
            if digest and digest in seen_hashes and seen_hashes[digest] != split:
                raise ValueError(f"image hash leakage: {digest} crosses splits")
            if digest:
                seen_hashes[digest] = split
            book = row.get("book")
            if book and book in seen_books and seen_books[book] != split:
                raise ValueError(f"book leakage: {book} crosses splits")
            if book:
                seen_books[book] = split


class ManifestDataset:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]
