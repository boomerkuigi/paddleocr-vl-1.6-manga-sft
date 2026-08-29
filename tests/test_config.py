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


def test_l4_timing_inherits_real_pilot_training_settings():
    config = load_config(ROOT / "configs/l4_timing.yaml")
    assert config["method"]["type"] == "full"
    assert config["model"]["dtype"] == "bfloat16"
    assert config["model"]["gradient_checkpointing"] is True
    assert config["training"]["gradient_accumulation_steps"] == 16
    assert config["training"]["optim"] == "adamw_torch"
    assert config["training"]["warmup_steps"] == 0.03
    assert config["training"]["eval_steps"] == 2500
    assert config["training"]["save_steps"] == 500
    assert config["training"]["max_steps"] == 20
    assert config["training"]["resume_from_checkpoint"] is False
    assert config["training"]["select_best_checkpoint_at_end"] is False
    assert config["timing"]["exclude_first_optimizer_steps"] == 1


def test_best_selection_requires_evaluation_to_coincide_with_a_save():
    config = load_config(ROOT / "configs/pilot.yaml")
    config["training"]["eval_steps"] = 2400
    with pytest.raises(ValueError, match="multiple of save_steps"):
        validate_config(config)


def test_v2_continuation_pilot_is_read_only_and_test_excluding():
    config = load_config(ROOT / "configs/v2_continuation_pilot.yaml")
    assert config["model"]["id"] == "AlphaBeta07/PaddleOCR-VL-1.6-For-Manga"
    assert config["model"]["revision"] == "103c97c277d688b31b8adb1bb2228380b77a640b"
    assert config["data"]["forbid_test_access"] is True
    assert config["data"]["immutable_test_samples"] == 11063
    assert config["data"]["targeted_mixture"]["extra_draws"] == 25000
    assert config["training"]["max_steps"] == 2500
    assert config["training"]["learning_rate"] == 2.5e-6
    assert config["training"]["eval_steps"] == 1250
    assert config["training"]["save_steps"] == 500
    assert config["training"]["additional_save_steps"] == [1250]
    assert config["hub"]["push_to_hub"] is False
