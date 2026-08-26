# Paid-run preflight review

Reviewed 2026-08-26 before any Hugging Face Job was launched.

## 1. Validation and checkpoint cadence

The original `eval_steps: 500` matched `save_steps: 500` because Transformers'
built-in `load_best_model_at_end` requires step-based saving to be a multiple of
the evaluation interval. That was simple and maximized recovery granularity, but
it coupled a cheap recovery save to an expensive full validation pass.

Manga109-v2026 contains 159,598 text annotations across all 109 books. The
commercially usable Manga109-s subset has 87 books. The older manga model reports
roughly 0.1M Manga109-s crops; proportional scaling of the 2026 total gives an
upper planning estimate near 127k. Actual book-grouped counts will be emitted by
the CPU preflight.

| Planning total | Train / validation | Optimizer steps per epoch | 500-step evaluations in 3 epochs | Validation sample forwards |
|---:|---:|---:|---:|---:|
| 100,000 | 80,000 / 10,000 | 5,000 | 30 | 300,000 |
| 127,000 | 101,600 / 12,700 | 6,350 | 38 | 482,600 |

These are loss-only teacher-forced validation forwards, not autoregressive
generation, but they are still substantial. A validation forward commonly costs
roughly 25–50% of a checkpointed forward+backward training sample. On that basis,
the old cadence could add roughly 30–80% over training-only time, depending on
the real split and image/token lengths.

The selected 2,500-step cadence gives approximately 6–7 scheduled validation
passes. If the last optimizer step is not already an evaluation boundary,
`train.py` performs one final pass so the final weights can also win selection.
That is 6–8 passes and about 60k–102k validation forwards. Once-per-epoch
validation would be cheapest at three passes, but gives only three selection
points. Every 2,000 steps gives about 7–9 scheduled passes; every 3,000 gives
about 5–6. A 2,500-step interval is the middle ground: approximately twice per
epoch at the lower estimate, includes the expected 5,000/15,000-step
epoch/final boundaries, and remains useful if the 2026 subset is larger.

Recovery saves remain every 500 steps. `load_best_model_at_end` is disabled only
to satisfy the Transformers interval constraint; `train.py` evaluates the final
weights when necessary and reloads the prior best checkpoint when it is better.
The relevant intervals are validated at config-load time. The rolling Hub
`last-checkpoint` is resumable in a fresh Job only when
`HF_RESUME_FROM_HUB=1` is explicitly set.

Verdict: **valid concern; cadence changed without weakening recovery or model
selection**.

## 2. Training and benchmarking lifecycle

The original all-in-one entrypoint minimized setup duplication: the gated data
was extracted/cropped once, the just-trained local model needed no redownload,
and all artifacts were produced by one command. It already pushed rolling
checkpoints and the final model before baseline installation/evaluation, so a
late baseline failure would not erase successful weights.

However, the all-in-one Job still had two material drawbacks:

1. the L4 remained billable while four sequential models performed held-out
   inference; and
2. an optional baseline dependency/download failure made the expensive Job end
   in a failed state even after the final model was safely uploaded.

Training and benchmark entrypoints are now separate. Both remount the gated
dataset and create crops ephemerally. This duplicates preparation, but preserves
the redistribution restriction and is preferable to persisting derived crops in
a public or shared dataset. The 1B BF16 Paddle baselines and the new model benefit
from an L4; Manga-OCR alone could run cheaper, but splitting the benchmark into
additional Jobs would repeatedly prepare the gated data and complicate the first
experiment.

Verdict: **valid concern; use a training Job followed by a separate L4 benchmark
Job**.

## 3. Gated dataset access

The local smoke fixture proved XML parsing, crop creation, Japanese labels,
deterministic splitting, leakage checks, and processor/config loading. It did
not prove the supplied identity can mount the gated repository, the secret token
can query it, the current repository contains the expected ZIP/layout, or the
real book/sample counts.

Hugging Face's current Jobs CLI documents `cpu-basic` as a supported flavor and
the volume syntax
`hf://datasets/hal-utokyo/Manga109-s:/data/manga109s:ro`. Dataset/model volumes
are read-only and authorized when the Job is created. Current Jobs pricing lists
CPU Basic at $0.01/hour with 50 GB ephemeral storage; the gated dataset page
lists a total repository size of 3.3 GB.

The new CPU preflight performs both independent authorization checks:

- the mount must succeed and expose the archive/directories; and
- `HfApi` must access the gated repository using the forwarded encrypted
  `HF_TOKEN`.

It then safely extracts the archive, scans every XML and referenced page, counts
the deterministic splits, validates bounding boxes, and loads one real crop from
each split without writing the full derived crop set. Its two-hour ceiling is
about $0.02. No preflight Job was launched during repository review.

Verdict: **valid concern; run the negligible-cost CPU preflight before any L4**.
