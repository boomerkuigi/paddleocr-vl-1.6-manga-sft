#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from manga_sft.collator import PaddleOCRVLCollator
from manga_sft.config import load_config, resolve_project_path
from manga_sft.dataset import ManifestDataset, read_jsonl, validate_no_leakage


def auto_checkpoint(output_dir: Path) -> str | None:
    checkpoints = []
    if output_dir.is_dir():
        for path in output_dir.glob("checkpoint-*"):
            try:
                checkpoints.append((int(path.name.rsplit("-", 1)[1]), path))
            except ValueError:
                continue
    return str(max(checkpoints)[1]) if checkpoints else None


def hub_checkpoint(model_id: str | None) -> str | None:
    """Resolve the Hub's rolling recovery checkpoint only when explicitly requested."""
    enabled = os.environ.get("HF_RESUME_FROM_HUB", "").strip().lower()
    if enabled not in {"1", "true", "yes"} or not model_id:
        return None
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=model_id,
            repo_type="model",
            allow_patterns=["last-checkpoint/**"],
            token=os.environ.get("HF_TOKEN"),
        )
    )
    checkpoint = snapshot / "last-checkpoint"
    if not (checkpoint / "trainer_state.json").is_file():
        raise FileNotFoundError(
            f"HF_RESUME_FROM_HUB was requested but {model_id} has no usable last-checkpoint"
        )
    return str(checkpoint)


def select_best_checkpoint(trainer, training_cfg: dict) -> str | None:
    """Load the best evaluated save while allowing more frequent recovery saves."""
    if not training_cfg.get("select_best_checkpoint_at_end", False):
        return None
    metric_name = str(training_cfg.get("metric_for_best_model", "eval_loss"))
    if not metric_name.startswith("eval_"):
        metric_name = f"eval_{metric_name}"
    final_metrics = next(
        (
            entry
            for entry in reversed(trainer.state.log_history)
            if entry.get("step") == trainer.state.global_step and metric_name in entry
        ),
        None,
    )
    if final_metrics is None:
        final_metrics = trainer.evaluate()
    final_metric = float(final_metrics[metric_name])
    previous_best = trainer.state.best_metric
    greater_is_better = bool(training_cfg.get("greater_is_better", False))
    final_is_better = previous_best is None or (
        final_metric > previous_best if greater_is_better else final_metric < previous_best
    )
    if final_is_better:
        trainer.state.best_metric = final_metric
        return f"final-step-{trainer.state.global_step}"
    checkpoint = trainer.state.best_model_checkpoint
    if not checkpoint:
        raise RuntimeError(
            "Best-checkpoint selection was requested but no prior evaluated checkpoint was saved. "
            "Ensure eval_steps coincides with a save step and occurs before training ends."
        )
    # Transformers 5.16 requires save_steps to be a multiple of eval_steps when
    # load_best_model_at_end is enabled, which prevents saves every 500 with
    # evaluations every 2500. The pinned Trainer's loader correctly handles both
    # full and PEFT checkpoints, so invoke it explicitly after training.
    trainer._load_best_model()
    return str(checkpoint)


def write_release_files(
    output_dir: Path,
    project_root: Path,
    config: dict,
    train_count: int,
    validation_count: int,
    test_count: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_config.resolved.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    template = (project_root / "MODEL_CARD_TEMPLATE.md").read_text(encoding="utf-8")
    version = str(config["data"].get("dataset_version", "version recorded by dataset mount"))
    card = template.replace("[VERSION]", version).replace(
        "Add the required Manga109 citations and the exact train/validation/test counts from `split_summary.json` before release.",
        f"Prepared split sizes: train={train_count}, validation={validation_count}, test={test_count}. "
        "Add the required Manga109 citations before any public release.",
    )
    (output_dir / "README.md").write_text(card, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune PaddleOCR-VL 1.6 on manga crops")
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load config/manifests/processor/config but not model weights or training",
    )
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    project_root = args.config.resolve().parent.parent
    model_cfg = config["model"]
    training_cfg = config["training"]
    manifest_dir = resolve_project_path(args.config, config["data"]["manifest_dir"])
    output_dir = resolve_project_path(args.config, training_cfg["output_dir"])
    train_rows = read_jsonl(
        manifest_dir / config["data"]["train"], config["data"].get("verify_images", True)
    )
    validation_rows = read_jsonl(
        manifest_dir / config["data"]["validation"], config["data"].get("verify_images", True)
    )
    test_rows = read_jsonl(
        manifest_dir / config["data"]["test"], config["data"].get("verify_images", True)
    )
    validate_no_leakage({"train": train_rows, "validation": validation_rows, "test": test_rows})
    if not train_rows or not validation_rows:
        raise ValueError("Train and validation manifests must both be non-empty")

    import torch
    from transformers import (
        AutoConfig,
        AutoModelForImageTextToText,
        AutoProcessor,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(int(config["project"]["seed"]))
    common = {
        "revision": model_cfg.get("revision"),
        "trust_remote_code": model_cfg.get("trust_remote_code", False),
    }
    processor = AutoProcessor.from_pretrained(
        model_cfg["id"],
        min_pixels=model_cfg.get("min_pixels"),
        max_pixels=model_cfg.get("max_pixels"),
        **common,
    )
    architecture = AutoConfig.from_pretrained(model_cfg["id"], **common)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "validated",
                    "model_id": model_cfg["id"],
                    "revision": model_cfg.get("revision"),
                    "architectures": getattr(architecture, "architectures", None),
                    "train_samples": len(train_rows),
                    "validation_samples": len(validation_rows),
                    "test_samples_excluded_from_trainer": len(test_rows),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if not torch.cuda.is_available():
        raise RuntimeError("Actual PaddleOCR-VL 1.6 training requires a CUDA GPU; use --validate-only here")
    dtype = getattr(torch, model_cfg.get("dtype", "bfloat16"))
    model = AutoModelForImageTextToText.from_pretrained(
        model_cfg["id"],
        torch_dtype=dtype,
        attn_implementation=model_cfg.get("attention_implementation", "sdpa"),
        **common,
    )
    model.config.use_cache = False
    if model_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if config["method"]["type"] == "lora":
        from peft import LoraConfig, get_peft_model

        method = config["method"]
        model = get_peft_model(
            model,
            LoraConfig(
                r=int(method["r"]),
                lora_alpha=int(method["alpha"]),
                lora_dropout=float(method["dropout"]),
                target_modules=method.get("target_modules", "all-linear"),
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        model.print_trainable_parameters()

    hub_cfg = config.get("hub", {})
    hub_model_id = os.environ.get(hub_cfg.get("model_id_env", "HF_MODEL_REPO"))
    push_to_hub = bool(args.push_to_hub or hub_cfg.get("push_to_hub", False))
    if push_to_hub and not hub_model_id:
        raise ValueError("HF_MODEL_REPO must be set when pushing to the Hub")

    arguments = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(training_cfg["epochs"]),
        per_device_train_batch_size=int(training_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
        learning_rate=float(training_cfg["learning_rate"]),
        optim=training_cfg.get("optim", "adamw_torch"),
        adam_beta1=float(training_cfg.get("adam_beta1", 0.9)),
        adam_beta2=float(training_cfg.get("adam_beta2", 0.999)),
        adam_epsilon=float(training_cfg.get("adam_epsilon", 1e-8)),
        weight_decay=float(training_cfg["weight_decay"]),
        warmup_steps=float(training_cfg["warmup_steps"]),
        lr_scheduler_type=training_cfg["lr_scheduler_type"],
        max_grad_norm=float(training_cfg["max_grad_norm"]),
        bf16=bool(training_cfg["bf16"]),
        gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        tf32=bool(training_cfg.get("tf32", True)),
        dataloader_num_workers=int(training_cfg.get("dataloader_num_workers", 4)),
        eval_strategy=training_cfg["eval_strategy"],
        eval_steps=int(training_cfg["eval_steps"]),
        save_strategy=training_cfg["save_strategy"],
        save_steps=int(training_cfg["save_steps"]),
        save_total_limit=int(training_cfg["save_total_limit"]),
        logging_steps=int(training_cfg["logging_steps"]),
        report_to=training_cfg.get("report_to", "none"),
        load_best_model_at_end=bool(training_cfg.get("load_best_model_at_end", True)),
        metric_for_best_model=training_cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=bool(training_cfg.get("greater_is_better", False)),
        max_steps=int(training_cfg.get("max_steps", -1)),
        remove_unused_columns=False,
        seed=int(config["project"]["seed"]),
        data_seed=int(config["project"]["seed"]),
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        hub_private_repo=bool(hub_cfg.get("private", True)),
        hub_strategy="checkpoint",
    )
    collator = PaddleOCRVLCollator(
        processor=processor,
        prompt=model_cfg.get("prompt", "OCR:"),
        max_length=int(model_cfg.get("max_length", 2048)),
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=ManifestDataset(train_rows),
        eval_dataset=ManifestDataset(validation_rows),
        data_collator=collator,
        processing_class=processor,
    )
    resume_setting = training_cfg.get("resume_from_checkpoint", "auto")
    if resume_setting == "auto":
        resume = auto_checkpoint(output_dir) or hub_checkpoint(hub_model_id)
    else:
        resume = resume_setting
    result = trainer.train(resume_from_checkpoint=resume or None)
    selected_checkpoint = select_best_checkpoint(trainer, training_cfg)
    if selected_checkpoint:
        result.metrics["selected_best_checkpoint"] = selected_checkpoint
    trainer.save_model(str(output_dir / "final"))
    processor.save_pretrained(str(output_dir / "final"))
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()
    write_release_files(
        output_dir,
        project_root,
        config,
        len(train_rows),
        len(validation_rows),
        len(test_rows),
    )
    if push_to_hub:
        trainer.push_to_hub(
            commit_message="Complete PaddleOCR-VL 1.6 manga pilot",
            dataset="Manga109-s (local/gated; images not redistributed)",
        )


if __name__ == "__main__":
    main()
