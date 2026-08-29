from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


REQUIRED_PATHS = (
    "project.seed",
    "model.id",
    "model.prompt",
    "method.type",
    "data.manifest_dir",
    "training.output_dir",
)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _get_path(config: dict[str, Any], dotted: str) -> Any:
    value: Any = config
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"Missing required configuration key: {dotted}")
        value = value[part]
    return value


def validate_config(config: dict[str, Any]) -> None:
    for path in REQUIRED_PATHS:
        _get_path(config, path)
    if config["method"]["type"] not in {"full", "lora"}:
        raise ValueError("method.type must be 'full' or 'lora'")
    if config["training"].get("max_steps", -1) == 0:
        raise ValueError("training.max_steps may not be zero")
    if int(config["project"]["seed"]) < 0:
        raise ValueError("project.seed must be non-negative")
    training = config.get("training", {})
    timing = config.get("timing", {})
    if timing.get("enabled", False):
        max_steps = int(training.get("max_steps", -1))
        excluded = int(timing.get("exclude_first_optimizer_steps", 1))
        if max_steps <= 1:
            raise ValueError("timing runs require training.max_steps greater than one")
        if excluded < 0 or excluded >= max_steps:
            raise ValueError(
                "timing.exclude_first_optimizer_steps must be non-negative and below max_steps"
            )
    if training.get("select_best_checkpoint_at_end", False):
        if training.get("load_best_model_at_end", False):
            raise ValueError(
                "Use either load_best_model_at_end or select_best_checkpoint_at_end, not both"
            )
        if training.get("eval_strategy") != "steps" or training.get("save_strategy") != "steps":
            raise ValueError(
                "select_best_checkpoint_at_end currently requires step-based evaluation and saving"
            )
        eval_steps = int(training.get("eval_steps", 0))
        save_steps = int(training.get("save_steps", 0))
        additional_save_steps = {int(value) for value in training.get("additional_save_steps", [])}
        if (
            eval_steps <= 0
            or save_steps <= 0
            or (eval_steps % save_steps and eval_steps not in additional_save_steps)
        ):
            raise ValueError(
                "eval_steps must be a positive multiple of save_steps or listed in additional_save_steps"
            )
    mixture = config.get("data", {}).get("targeted_mixture")
    if mixture is not None:
        if not bool(config["data"].get("forbid_test_access", False)):
            raise ValueError("targeted pilot mixtures must set data.forbid_test_access: true")
        if int(mixture.get("extra_draws", 0)) <= 0:
            raise ValueError("data.targeted_mixture.extra_draws must be positive")
        weights = mixture.get("target_weights", {})
        expected_groups = {
            "repeated_or_long_mark",
            "likely_sfx",
            "punctuation_form",
            "visual_or_unusual_unicode",
        }
        if set(weights) != expected_groups or abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-9:
            raise ValueError("targeted_mixture.target_weights must name the four V2 groups and sum to 1")
    diagnostics = config.get("validation_diagnostics", {})
    if diagnostics.get("enabled", False):
        if training.get("eval_strategy") != "steps":
            raise ValueError("validation diagnostics require step-based evaluation")
        steps = [int(value) for value in diagnostics.get("steps", [])]
        if not steps or any(step <= 0 for step in steps):
            raise ValueError("validation_diagnostics.steps must contain positive optimizer steps")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    inherited = loaded.pop("inherits", None)
    if inherited:
        parent = load_config(config_path.parent / str(inherited))
        loaded = _deep_merge(parent, loaded)
    validate_config(loaded)
    return loaded


def resolve_project_path(config_file: str | Path, value: str | Path) -> Path:
    value_path = Path(value)
    if value_path.is_absolute():
        return value_path
    project_root = Path(config_file).resolve().parent.parent
    return (project_root / value_path).resolve()
