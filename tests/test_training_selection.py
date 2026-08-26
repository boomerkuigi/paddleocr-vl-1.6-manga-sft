from types import SimpleNamespace

import pytest

from scripts.train import select_best_checkpoint


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
