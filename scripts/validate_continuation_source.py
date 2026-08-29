#!/usr/bin/env python3
"""CPU-only strict loading check for a continuation source model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from manga_sft.config import load_config
from manga_sft.inference import load_paddleocr_vl_processor, normalize_paddleocr_vl_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--meta-only",
        action="store_true",
        help="Validate conversion/key mapping without materializing all tensors in CPU RAM",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    model_cfg = config["model"]

    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText

    common = {"revision": model_cfg.get("revision"), "trust_remote_code": False}
    processor = load_paddleocr_vl_processor(model_cfg["id"], revision=model_cfg.get("revision"))
    architecture = normalize_paddleocr_vl_config(
        AutoConfig.from_pretrained(model_cfg["id"], **common)
    )
    model, loading_info = AutoModelForImageTextToText.from_pretrained(
        model_cfg["id"],
        config=architecture,
        dtype=torch.bfloat16,
        device_map={"": "meta"} if args.meta_only else "cpu",
        low_cpu_mem_usage=True,
        attn_implementation=model_cfg.get("attention_implementation", "sdpa"),
        output_loading_info=True,
        **common,
    )
    del model
    problems = {
        key: sorted(loading_info.get(key, []))
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    }
    if any(problems.values()):
        raise RuntimeError(f"Continuation source checkpoint did not load exactly: {problems}")
    print(
        json.dumps(
            {
                "status": "strict_meta_mapping_passed" if args.meta_only else "strict_cpu_load_passed",
                "model_id": model_cfg["id"],
                "revision": model_cfg.get("revision"),
                "processor": type(processor).__name__,
                "loading_problems": problems,
                "meta_only": args.meta_only,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
