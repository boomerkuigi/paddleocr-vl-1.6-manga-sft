#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from manga_sft.config import load_config, resolve_project_path
from manga_sft.dataset import read_jsonl, validate_no_leakage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))
    parser.add_argument("--load-processor", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest_dir = resolve_project_path(args.config, config["data"]["manifest_dir"])
    manifests = {
        split: read_jsonl(manifest_dir / config["data"][name], verify_images=True)
        for split, name in (("train", "train"), ("validation", "validation"), ("test", "test"))
    }
    validate_no_leakage(manifests)
    result = {
        "config": "valid",
        "sizes": {key: len(value) for key, value in manifests.items()},
        "split_leakage": "none",
        "images": "valid",
        "processor": "not requested",
    }
    if args.load_processor:
        from transformers import AutoConfig, AutoProcessor

        model = config["model"]
        common = {
            "revision": model.get("revision"),
            "trust_remote_code": model.get("trust_remote_code", True),
        }
        processor = AutoProcessor.from_pretrained(model["id"], **common)
        model_config = AutoConfig.from_pretrained(model["id"], **common)
        result["processor"] = type(processor).__name__
        result["model_type"] = getattr(model_config, "model_type", None)
        result["architectures"] = getattr(model_config, "architectures", None)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

