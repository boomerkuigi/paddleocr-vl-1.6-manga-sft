from pathlib import Path

import pytest

from manga_sft.config import load_config, validate_config


ROOT = Path(__file__).resolve().parents[1]


def test_pilot_config_loads():
    config = load_config(ROOT / "configs/pilot.yaml")
    assert config["model"]["id"] == "PaddlePaddle/PaddleOCR-VL-1.6"
    assert config["model"]["trust_remote_code"] is False
    assert config["training"]["epochs"] == 3
    assert config["training"]["warmup_steps"] == 0.03
    assert "warmup_ratio" not in config["training"]
    assert config["training"]["eval_steps"] == 2500
    assert config["training"]["save_steps"] == 500
    assert config["training"]["select_best_checkpoint_at_end"] is True


def test_lora_inheritance():
    config = load_config(ROOT / "configs/pilot_lora.yaml")
    assert config["method"]["type"] == "lora"
    assert config["model"]["id"] == "PaddlePaddle/PaddleOCR-VL-1.6"
    assert config["training"]["output_dir"].endswith("pilot-lora")


def test_gpu_smoke_uses_real_manifests_for_one_step():
    config = load_config(ROOT / "configs/gpu_smoke.yaml")
    assert config["data"]["manifest_dir"] == "data/prepared/manifests"
    assert config["training"]["max_steps"] == 1
    assert config["training"]["eval_strategy"] == "no"
    assert config["training"]["select_best_checkpoint_at_end"] is False


def test_best_selection_requires_evaluation_to_coincide_with_a_save():
    config = load_config(ROOT / "configs/pilot.yaml")
    config["training"]["eval_steps"] = 2400
    with pytest.raises(ValueError, match="multiple of save_steps"):
        validate_config(config)
