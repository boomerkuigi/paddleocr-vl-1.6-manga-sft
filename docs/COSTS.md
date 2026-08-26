# Cost controls and estimates

No command in the setup or offline validation sections requests paid compute.
Only the prominently warned `hf jobs run` command does so.

Safeguards:

- explicit `max_steps` for smoke and three epochs for pilot;
- a CPU-only gated-dataset preflight before any GPU allocation;
- explicit 40-hour Job timeout (about a $32 L4 hard ceiling at the listed rate);
- shell fail-fast behavior (`set -euo pipefail`);
- no retry loop;
- recovery checkpoints every 500 optimizer steps, last three retained locally;
- rolling `last-checkpoint` upload to the private model repository;
- local automatic resume, plus explicit fresh-Job Hub resume with
  `HF_RESUME_FROM_HUB=1` to avoid accidentally reusing an old experiment;
- validation every 2,500 steps, a final evaluation when needed, and explicit
  selection of the best evaluated weights before final export;
- test excluded from training and model selection;
- immutable base-model revision and deterministic seeds;
- push checkpoints/final outputs to a private destination;
- the training process exits immediately after final-model upload;
- held-out baseline evaluation runs as a separate, bounded Job, so its failure
  cannot change the completed training Job's status.

Current planning rates are L4 24 GB $0.80/h, A10G-small 24 GB $1.00/h,
A10G-large 24 GB $1.50/h, and A100 80 GB $2.50/h. The broad runtime ranges in
the README reflect the older reported RTX 3060 12 GB / ~27-hour three-epoch run
and the 1.6 KIE A100 report, neither of which is a controlled benchmark of this
code.

A 20–50 step GPU timing sample after the mandatory one-step smoke should replace
these estimates before committing to three epochs. Include startup, processor,
evaluation, and upload overhead when projecting cost.

Current staged ceilings are:

- dataset preflight: `cpu-basic`, 2 hours, about $0.02 maximum;
- one-step L4 smoke: 3 hours, $2.40 maximum (expected to finish much sooner);
- three-epoch L4 training: 40 hours, $32 maximum;
- separate L4 benchmark: 12 hours, $9.60 maximum.

See `docs/PREFLIGHT_REVIEW.md` for the validation-pass arithmetic and staging
decision.
