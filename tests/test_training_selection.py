import json
import sys
from types import SimpleNamespace

import pytest

from scripts.train import (
    auto_checkpoint,
    ensure_private_hub_repo,
    hub_checkpoint,
    select_best_checkpoint,
)


class DummyTrainer:
    def __init__(self, checkpoint, best_metric=0.2, final_metric=0.4, final_logged=True):
        log_history = [{"step": 10, "eval_loss": final_metric}] if final_logged else []
        self.state = SimpleNamespace(
            best_model_checkpoint=checkpoint,
            best_metric=best_metric,
            global_step=10,
            log_history=log_history,
        )
        self.final_metric = final_metric
        self.loaded = False

    def _load_best_model(self):
        self.loaded = True

    def evaluate(self):
        return {"eval_loss": self.final_metric}


def test_explicit_best_checkpoint_selection_loads_evaluated_save():
    trainer = DummyTrainer("checkpoint-2500")
    selected = select_best_checkpoint(
        trainer, {"select_best_checkpoint_at_end": True}
    )
    assert selected == "checkpoint-2500"
    assert trainer.loaded is True


def test_unscheduled_final_evaluation_can_select_current_weights():
    trainer = DummyTrainer(
        "checkpoint-2500", best_metric=0.2, final_metric=0.1, final_logged=False
    )
    selected = select_best_checkpoint(
        trainer, {"select_best_checkpoint_at_end": True, "greater_is_better": False}
    )
    assert selected == "final-step-10"
    assert trainer.loaded is False
    assert trainer.state.best_metric == 0.1


def test_explicit_best_checkpoint_selection_fails_without_evaluation():
    trainer = DummyTrainer(None)
    with pytest.raises(RuntimeError, match="no prior evaluated checkpoint"):
        select_best_checkpoint(trainer, {"select_best_checkpoint_at_end": True})


def test_auto_checkpoint_ignores_incomplete_save(tmp_path):
    incomplete = tmp_path / "checkpoint-20"
    incomplete.mkdir()
    complete = tmp_path / "checkpoint-10"
    complete.mkdir()
    (complete / "trainer_state.json").write_text("{}", encoding="utf-8")
    assert auto_checkpoint(tmp_path) == str(complete)


class DummyHubApi:
    def __init__(self, private):
        self.private = private
        self.created = None

    def create_repo(self, **kwargs):
        self.created = kwargs

    def model_info(self, **kwargs):
        return SimpleNamespace(private=self.private)


def test_private_hub_destination_is_verified():
    api = DummyHubApi(private=True)
    assert ensure_private_hub_repo("owner/model", "secret", api=api) is api
    assert api.created["private"] is True


def test_public_hub_destination_is_rejected():
    with pytest.raises(RuntimeError, match="is not private"):
        ensure_private_hub_repo("owner/model", "secret", api=DummyHubApi(private=False))


def test_hub_resume_repoints_older_best_checkpoint(tmp_path, monkeypatch):
    last = tmp_path / "last-checkpoint"
    best = tmp_path / "best-checkpoint"
    last.mkdir()
    best.mkdir()
    (best / "model.safetensors").write_bytes(b"weights")
    (last / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": 5000,
                "best_global_step": 2500,
                "best_model_checkpoint": "checkpoints/pilot-full/checkpoint-2500",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HF_RESUME_FROM_HUB", "1")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=lambda **kwargs: str(tmp_path)),
    )

    assert hub_checkpoint("owner/private-model") == str(last)
    state = json.loads((last / "trainer_state.json").read_text(encoding="utf-8"))
    assert state["best_model_checkpoint"] == str(best)
