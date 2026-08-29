from collections import Counter

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from torch import nn
from transformers import Trainer, TrainingArguments

from manga_sft.dataset import ManifestDataset, validate_no_leakage
from manga_sft.mixture import DEFAULT_TARGET_WEIGHTS, DeterministicMixtureSampler, build_mixture_plan
from scripts.train import make_mixture_trainer_class


def _row(index: int, text: str, bbox: list[int]) -> dict:
    return {
        "sample_id": f"train-{index:05d}",
        "image_path": f"/unused/{index}.png",
        "gold": text,
        "bbox": bbox,
        "book": "train-book",
        "split": "train",
    }


def _train_rows() -> list[dict]:
    rows = []
    for index in range(500):
        rows.append(_row(index, "ああ", [0, 0, 100, 100]))
    for index in range(500, 1000):
        rows.append(_row(index, "カタカナ", [0, 0, 100, 100]))
    for index in range(1000, 1500):
        rows.append(_row(index, "...", [0, 0, 100, 100]))
    for index in range(1500, 2000):
        rows.append(_row(index, "漢字", [0, 0, 20, 80]))
    return rows


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, input_ids=None, labels=None):
        del input_ids, labels
        return {"loss": self.weight.sum() * 0}


def _collate(examples: list[dict]) -> dict:
    return {
        "sample_ids": [example["sample_id"] for example in examples],
        "indices": torch.tensor(
            [int(example["sample_id"].rsplit("-", 1)[1]) for example in examples]
        ),
    }


def test_v2_mixture_trainer_loads_first_batch_with_transformers_5_sampler_contract(tmp_path):
    assert transformers.__version__ == "5.16.0"
    train_rows = _train_rows()
    validation_rows = [
        {
            **row,
            "sample_id": row["sample_id"].replace("train-", "validation-"),
            "book": "validation-book",
            "split": "validation",
        }
        for row in train_rows[:7]
    ]
    test_rows = [
        {
            **row,
            "sample_id": row["sample_id"].replace("train-", "test-"),
            "book": "test-book",
            "split": "test",
        }
        for row in train_rows[7:10]
    ]
    validate_no_leakage(
        {"train": train_rows, "validation": validation_rows, "test": test_rows}
    )

    plan = build_mixture_plan(
        train_rows,
        seed=42,
        extra_draws=100,
        target_weights=DEFAULT_TARGET_WEIGHTS,
    )
    mixture_sampler = DeterministicMixtureSampler(plan)
    mixture_trainer_class = make_mixture_trainer_class(Trainer)
    trainer = mixture_trainer_class(
        model=_TinyModel(),
        args=TrainingArguments(
            output_dir=str(tmp_path / "trainer"),
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to="none",
            use_cpu=True,
        ),
        train_dataset=ManifestDataset(train_rows),
        eval_dataset=ManifestDataset(validation_rows),
        data_collator=_collate,
        mixture_sampler=mixture_sampler,
    )

    assert trainer._get_train_sampler(trainer.train_dataset) is mixture_sampler
    train_dataloader = trainer.get_train_dataloader()
    first_batch = next(iter(train_dataloader))
    assert first_batch["sample_ids"] == [
        train_rows[index]["sample_id"] for index, _ in plan.sampler_entries[:2]
    ]

    observed_train_ids = [
        sample_id
        for batch in train_dataloader
        for sample_id in batch["sample_ids"]
    ]
    expected_train_ids = [
        train_rows[index]["sample_id"] for index, _ in plan.sampler_entries
    ]
    assert observed_train_ids == expected_train_ids

    sampler_entries = list(mixture_sampler)
    assert sampler_entries == list(DeterministicMixtureSampler(plan))
    counts = Counter(index for index, _ in sampler_entries)
    assert len(counts) == len(train_rows)
    assert set(counts.values()) <= {1, 2}
    assert sum(count == 2 for count in counts.values()) == plan.extra_draws
    assert Counter(group for _, group in sampler_entries)["ordinary"] == len(train_rows)
    assert sum(group != "ordinary" for _, group in sampler_entries) == plan.extra_draws

    replayed = build_mixture_plan(
        train_rows,
        seed=42,
        extra_draws=100,
        target_weights=DEFAULT_TARGET_WEIGHTS,
    )
    mixture_sampler.set_epoch(99)
    assert replayed == plan
    assert list(mixture_sampler) == list(DeterministicMixtureSampler(replayed))

    eval_dataloader = trainer.get_eval_dataloader()
    assert eval_dataloader.sampler is not mixture_sampler
    assert list(eval_dataloader.sampler) == list(range(len(validation_rows)))
    observed_validation_ids = [
        sample_id
        for batch in eval_dataloader
        for sample_id in batch["sample_ids"]
    ]
    assert observed_validation_ids == [row["sample_id"] for row in validation_rows]
