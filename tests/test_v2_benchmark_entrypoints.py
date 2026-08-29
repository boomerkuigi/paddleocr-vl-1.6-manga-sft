from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v2_full_benchmark_is_single_model_and_uses_the_fixed_test_methodology():
    entrypoint = (ROOT / "scripts/hf_benchmark_v2_entrypoint.sh").read_text(encoding="utf-8")

    assert "requirements-baselines.txt" not in entrypoint
    assert "run_benchmark_smoke.py" not in entrypoint
    assert "--model new_model" in entrypoint
    assert entrypoint.count("python scripts/evaluate_baselines.py") == 1
    assert entrypoint.count("python scripts/evaluate.py") == 1
    assert "--manifest data/prepared/manifests/test.jsonl" in entrypoint
    assert "test_samples != 11063" in entrypoint
    assert '--model-id "${HF_MODEL_REPO}"' in entrypoint
    assert '--revision "${HF_MODEL_REVISION}"' in entrypoint
    assert "path_in_repo=\"benchmark/v2-only-test\"" in entrypoint
    for old_alias in ("manga_ocr", "paddle_manga", "paddle_1_6"):
        assert f'--model "{old_alias}"' not in entrypoint


def test_v2_smoke_is_single_model_and_reuses_the_private_fixed_subset():
    entrypoint = (ROOT / "scripts/hf_benchmark_v2_smoke_entrypoint.sh").read_text(
        encoding="utf-8"
    )

    assert "requirements-baselines.txt" not in entrypoint
    assert "run_benchmark_smoke.py" not in entrypoint
    assert "--model new_model" in entrypoint
    assert entrypoint.count("python scripts/evaluate_baselines.py") == 1
    assert entrypoint.count("python scripts/evaluate.py") == 1
    assert "source_test_samples\") != 11063" in entrypoint
    assert "len(rows) != 100" in entrypoint
    assert "original_split" in entrypoint
    assert "path_in_repo=\"benchmark/v2-only-smoke-100\"" in entrypoint
    for old_alias in ("manga_ocr", "paddle_manga", "paddle_1_6"):
        assert f'--model "{old_alias}"' not in entrypoint


def test_legacy_all_baseline_entrypoints_remain_present_and_unchanged_in_scope():
    full = (ROOT / "scripts/hf_benchmark_entrypoint.sh").read_text(encoding="utf-8")
    smoke = (ROOT / "scripts/hf_benchmark_smoke_entrypoint.sh").read_text(encoding="utf-8")

    assert "for baseline in manga_ocr paddle_manga paddle_1_6" in full
    assert "run_benchmark_smoke.py" in smoke
