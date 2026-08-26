# Cost controls and estimates

No command in the setup or offline validation sections requests paid compute.
Only the prominently warned `hf jobs run` command does so.

Safeguards:

- explicit `max_steps` for smoke and three epochs for pilot;
- explicit 40-hour Job timeout (about a $32 L4 hard ceiling at the listed rate);
- shell fail-fast behavior (`set -euo pipefail`);
- no retry loop;
- checkpoints every 500 optimizer steps, last three retained;
- automatic resume from the newest valid checkpoint;
- test excluded from training and model selection;
- immutable base-model revision and deterministic seeds;
- push checkpoints/final outputs to a private destination;
- the process exits immediately after evaluation/upload, allowing the Job to
  release its GPU.

Current planning rates are L4 24 GB $0.80/h, A10G-small 24 GB $1.00/h,
A10G-large 24 GB $1.50/h, and A100 80 GB $2.50/h. The broad runtime ranges in
the README reflect the older reported RTX 3060 12 GB / ~27-hour three-epoch run
and the 1.6 KIE A100 report, neither of which is a controlled benchmark of this
code.

A 20–50 step GPU timing sample after the mandatory one-step smoke should replace
these estimates before committing to three epochs. Include startup, processor,
evaluation, and upload overhead when projecting cost.
