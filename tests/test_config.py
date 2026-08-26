from pathlib import Path

from manga_sft.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_pilot_config_loads():
    config = load_config(ROOT / "configs/pilot.yaml")
    assert config["model"]["id"] == "PaddlePaddle/PaddleOCR-VL-1.6"
    assert config["training"]["epochs"] == 3


def test_lora_inheritance():
    config = load_config(ROOT / "configs/pilot_lora.yaml")
    assert config["method"]["type"] == "lora"
    assert config["model"]["id"] == "PaddlePaddle/PaddleOCR-VL-1.6"
    assert config["training"]["output_dir"].endswith("pilot-lora")

