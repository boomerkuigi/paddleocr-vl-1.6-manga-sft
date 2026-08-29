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
    split_names = (("train", "train"), ("validation", "validation"))
    if not bool(config["data"].get("forbid_test_access", False)):
        split_names += (("test", "test"),)
    manifests = {
        split: read_jsonl(manifest_dir / config["data"][name], verify_images=True)
        for split, name in split_names
    }
    validate_no_leakage(manifests)
    result = {
        "config": "valid",
        "sizes": {key: len(value) for key, value in manifests.items()},
        "split_leakage": "none",
        "test_manifest_accessed": "test" in manifests,
        "images": "valid",
        "processor": "not requested",
    }
    if args.load_processor:
        from transformers import AutoConfig
        from manga_sft.inference import load_paddleocr_vl_processor, normalize_paddleocr_vl_config

        model = config["model"]
        common = {
            "revision": model.get("revision"),
            "trust_remote_code": model.get("trust_remote_code", True),
        }
        processor = load_paddleocr_vl_processor(model["id"], revision=model.get("revision"))
        model_config = normalize_paddleocr_vl_config(AutoConfig.from_pretrained(model["id"], **common))
        result["processor"] = type(processor).__name__
        result["model_type"] = getattr(model_config, "model_type", None)
        result["architectures"] = getattr(model_config, "architectures", None)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
