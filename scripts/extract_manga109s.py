#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from manga_sft.dataset import safe_extract_zip


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely extract the official gated Manga109-s archive")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    safe_extract_zip(args.archive.resolve(), args.output.resolve())
    print(f"Extracted official archive to {args.output.resolve()}")


if __name__ == "__main__":
    main()
