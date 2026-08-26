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

