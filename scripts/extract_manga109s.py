#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from manga_sft.dataset import materialize_archive, safe_extract_zip


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safely extract the official gated Manga109-s archive"
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage-directory",
        type=Path,
        help=(
            "Copy a lazily mounted archive into this local directory before extraction. "
            "Recommended for Hugging Face repository volumes."
        ),
    )
    args = parser.parse_args()
    archive = args.archive.resolve()
    if args.stage_directory is not None:
        staged = materialize_archive(
            archive,
            args.stage_directory.resolve() / archive.name,
        )
        archive = staged.path
        print(
            "Materialized archive to regular local storage: "
            f"{archive} ({staged.size_bytes} bytes, sha256={staged.sha256})"
        )
    safe_extract_zip(archive, args.output.resolve())
    print(f"Extracted official archive to {args.output.resolve()}")


if __name__ == "__main__":
    main()
