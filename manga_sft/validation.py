from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from .metrics import aggregate_metrics, edit_distance
from .mixture import feature_groups


VALIDATION_GROUPS = (
    "all_validation",
    "repeated_or_long_mark",
    "likely_sfx",
    "punctuation_form",
    "visual_or_unusual_unicode",
)


def validation_group_indices(rows: list[dict]) -> dict[str, list[int]]:
    """Use only the canonical validation manifest; no reweighting or test rows."""
    groups = {group: [] for group in VALIDATION_GROUPS}
    groups["all_validation"] = list(range(len(rows)))
    for index, row in enumerate(rows):
        for group in feature_groups(row):
            groups[group].append(index)
    return groups


def diagnostic_metrics(rows: list[dict], predictions: list[str]) -> dict:
    if len(rows) != len(predictions):
        raise ValueError("validation rows and predictions must have the same length")
    groups = validation_group_indices(rows)
    metrics: dict[str, dict] = {}
    for group, indices in groups.items():
        pairs = [(str(rows[index]["gold"]), predictions[index]) for index in indices]
        summary = aggregate_metrics(pairs)
        distances = [edit_distance(gold, prediction) for gold, prediction in pairs]
        summary["edit_distance_distribution"] = {
            "0": sum(distance == 0 for distance in distances),
            "1": sum(distance == 1 for distance in distances),
            "2": sum(distance == 2 for distance in distances),
            "3_plus": sum(distance >= 3 for distance in distances),
        }
        metrics[group] = summary
    return metrics


def run_validation_diagnostics(
    model,
    processor,
    rows: list[dict],
    *,
    prompt: str,
    output_path: Path,
    step: int,
) -> dict:
    """Generate one canonical validation prediction per row and write metrics.

    This intentionally evaluates the unweighted validation split.  Targeted
    groups are overlapping views of the same predictions, so they add no extra
    generation calls and cannot influence best-checkpoint selection.
    """
    import torch

    was_training = model.training
    model.eval()
    predictions: list[str] = []
    target_device = next(model.parameters()).device
    for index, row in enumerate(rows, start=1):
        with Image.open(row["image_path"]) as source:
            image = source.convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        inputs = {key: value.to(target_device) for key, value in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        prompt_tokens = inputs["input_ids"].shape[1]
        predictions.append(
            processor.batch_decode(generated[:, prompt_tokens:], skip_special_tokens=True)[0]
        )
        if index % 100 == 0 or index == len(rows):
            print(
                "VALIDATION_DIAGNOSTIC_PROGRESS="
                + json.dumps({"step": step, "completed": index, "total": len(rows)})
            )
    if was_training:
        model.train()
    report = {"step": step, "groups": diagnostic_metrics(rows, predictions)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("VALIDATION_DIAGNOSTICS_JSON=" + json.dumps(report, ensure_ascii=False, sort_keys=True))
    return report
