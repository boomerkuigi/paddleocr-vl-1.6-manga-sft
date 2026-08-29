# V2 continuation-pilot data contract

`configs/v2_continuation_pilot.yaml` starts from the immutable promoted V1
revision `103c97c277d688b31b8adb1bb2228380b77a640b`.  It is a full-BF16,
2,500-optimizer-step pilot, not a benchmark or a model release.

Each mixture epoch contains the 99,128 de-duplicated Manga109-s train rows
once plus 25,000 additional rows.  The added slots are exactly 10,000
repeated-character/long-mark, 6,250 likely-SFX, 5,000 punctuation/form, and
3,750 vertical/compact/unusual-Unicode exposures.

Rows may have several feature tags.  The sampler first creates a seeded,
shuffled sequence of those exact quota labels, ranks each label's eligible rows
by a salted SHA-256 of seed, label, and `sample_id`, and skips a row already
chosen for another added slot.  A selected row consequently appears exactly
twice in an epoch (once ordinary, once targeted), never once per tag.  The
final index order is also seeded and fixed, which makes a checkpoint restart
replayable.

The pilot opens only `train.jsonl` and `validation.jsonl`.  The held-out
11,063-sample manifest is deliberately not read by its runtime.  CPU preflight
tests may separately verify all three split manifests for leakage; they do not
make pilot selections from the test split.

`scripts/validate_continuation_source.py` uses the same native Transformers 5
processor repair, config normalization, untied-embedding contract, and strict
missing/unexpected/mismatched-key checks as training.  Its `--meta-only` mode
can check a large private source's conversion/key mapping without requiring the
full model to fit in local CPU RAM.

At steps 1,250 and 2,500, Trainer computes ordinary unweighted validation loss
over the canonical validation split.  A diagnostic pass then generates one
prediction per validation crop and reports unweighted CER/exact plus the same
metrics and edit-distance bins for overlapping repeated/long-mark, likely-SFX,
punctuation/form, and visual/unusual-Unicode views.  Checkpoint selection stays
on raw `eval_loss`; diagnostics only report what changed.
