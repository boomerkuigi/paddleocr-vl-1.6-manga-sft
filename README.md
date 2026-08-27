# PaddleOCR-VL 1.6 manga SFT

Reproducible preparation, fine-tuning, and comparative evaluation for a future
`PaddleOCR-VL-1.6-For-Manga`. This is a text-region OCR project, not a full-page
detector or a Linguistik integration.

## Why

`kha-white/manga-ocr-base` and `jzhang533/PaddleOCR-VL-For-Manga` have
complementary errors. The goal is to port an already successful manga-specific
PaddleOCR-VL recipe to the newer PaddleOCR-VL 1.6, then measure whether the new
model improves the existing Paddle manga model while retaining useful
complementarity with Manga-OCR.

The main practical precedents are:

- [`openvino-book/PaddleOCR-VL-SFT-for-Japanese-Manga-on-RTX-3060`](https://github.com/openvino-book/PaddleOCR-VL-SFT-for-Japanese-Manga-on-RTX-3060):
  an older-generation full fine-tune on Manga109-s text crops. It reports about
  **64.4% exact sentence accuracy**, **10.88% CER**, **~27 hours**, and an
  **RTX 3060 12 GB**.
- [`megemini/PaddleOCR-VL-KIE`](https://github.com/megemini/PaddleOCR-VL-KIE):
  practical PaddleOCR-VL 1.6 full fine-tuning using Paddle/ERNIEKit. It is a KIE
  project, not a manga recipe, but confirms current 1.6 data/training behavior.

The pilot combines the manga task design from the first reference with the
current model/processor behavior learned from the second and from the official
[`PaddlePaddle/PaddleOCR-VL-1.6`](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6)
repository. See [`docs/PORTING_NOTES.md`](docs/PORTING_NOTES.md).

## Important data restriction

Manga109-s forbids redistribution to third parties. Do **not** commit or upload
its pages, annotations, derived crops, or a derivative dataset repository. The
official terms expressly allow publishing trained models when use of Manga109-s
is disclosed. Obtain and accept access from the official gated
[`hal-utokyo/Manga109-s`](https://huggingface.co/datasets/hal-utokyo/Manga109-s)
repository, then prepare crops locally. See [`docs/LICENSES.md`](docs/LICENSES.md).

## Setup

Python 3.11 or 3.12 is recommended.

```bash
git clone https://github.com/boomerkuigi/paddleocr-vl-1.6-manga-sft.git
cd paddleocr-vl-1.6-manga-sft
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pytest
```

Never put a Hugging Face token in a command committed to Git, a config, or an
`.env` file. Use `hf auth login` locally and `--secrets HF_TOKEN` for Jobs.

## Dataset preparation

The expected local shape is documented in [`data/README.md`](data/README.md).

```bash
python scripts/prepare_dataset.py \
  --manga109-root /absolute/path/to/Manga109-s \
  --output data/prepared \
  --seed 42

python scripts/inspect_samples.py data/prepared/manifests/train.jsonl --limit 10
python scripts/validate_environment.py --config configs/pilot.yaml --load-processor
```

The script creates manga text-region crops and a deterministic, book-grouped
80/10/10 split. Whole books—not random crops—are assigned to train, validation,
or held-out test. It records sizes, book assignments, seed, filtering, and image
hashes in `data/prepared/manifests/split_summary.json`. Duplicate sample IDs,
image hashes, and books crossing splits cause a hard failure.

Training targets preserve annotation text. Headline exact accuracy and CER use
raw strings. A line-ending-only result and a clearly labeled NFKC/whitespace
diagnostic are also emitted; neither replaces raw metrics. Raw predictions are
always retained.

## Format-only smoke test

This checks configuration, JSONL, Japanese UTF-8, images, splitting, and
serialization without claiming OCR quality:

```bash
python scripts/create_smoke_fixture.py
python scripts/validate_environment.py --config configs/smoke.yaml
python scripts/train.py --config configs/smoke.yaml --validate-only
```

`configs/smoke.yaml` is format-only and uses generated fixtures. The first real
GPU action uses `configs/gpu_smoke.yaml` with the prepared Manga109-s manifests.
That one-step run is the only conclusive check for model forward/backward tensor
shapes, 24 GB peak VRAM, optimizer allocation, and checkpoint writing.

## Pilot training

Recommended first method: **full BF16 fine-tuning**. Both successful references
used full fine-tuning, the official 1.6 Transformers model accepts labels, and
the 1B model plus crop inputs, batch size 1, gradient accumulation, and gradient
checkpointing is a reasonable 24 GB target. LoRA is retained as a contingency
in `configs/pilot_lora.yaml`; QLoRA is deliberately not the first experiment.

`configs/pilot.yaml` uses:

| Setting | Value |
|---|---:|
| Base revision | `c5630abae1d940eafe0697512a0325494b02ab42` |
| Epochs | 3 |
| Micro / effective batch | 1 / 16 |
| Learning rate | `1e-5`, cosine, 3% warmup |
| Optimizer | AdamW (PyTorch), betas 0.9/0.999, epsilon `1e-8`, weight decay 0.01 |
| Precision | BF16, TF32 enabled |
| Gradient checkpointing | enabled |
| Attention | PyTorch SDPA |
| Maximum sequence length | 2048; oversize samples fail rather than truncate |
| Validation interval | 2,500 optimizer steps (about twice per epoch at the expected size) |
| Recovery checkpoint interval | 500 optimizer steps |
| Retained checkpoints | 3 |
| Seed | 42 |

Run locally only after selecting a GPU:

```bash
python scripts/train.py --config configs/pilot.yaml
```

Training receives only train and validation datasets. The test manifest is read
solely for leakage checks and is never passed to `Trainer`. Existing
`checkpoint-*` directories resume automatically. Each 500-step save is also
pushed as the private Hub repo's rolling `last-checkpoint`. A fresh Job resumes
that Hub checkpoint only when `HF_RESUME_FROM_HUB=1` is explicitly set, avoiding
accidental reuse by a later experiment. Because Transformers requires built-in
best-model loading to couple save/eval intervals, `train.py` evaluates the final
weights when needed and reloads the prior best checkpoint if it is better before
saving the selected model at `checkpoints/pilot-full/`. For fresh-Job resume,
the rolling private `last-checkpoint` includes optimizer state and a separate
private `best-checkpoint` retains the model-only weights needed for correct
best-model selection after an interruption. The destination is verified private
before Trainer is allowed to upload anything.
The quantitative rationale is in [`docs/PREFLIGHT_REVIEW.md`](docs/PREFLIGHT_REVIEW.md).

## Evaluation and disagreement analysis

Install the Manga-OCR baseline adapter only on evaluation machines:

```bash
python -m pip install -r requirements-baselines.txt
```

Evaluate one model at a time so weights do not coexist in VRAM:

```bash
MANIFEST=data/prepared/manifests/test.jsonl
mkdir -p outputs/predictions

python scripts/evaluate_baselines.py --manifest "$MANIFEST" --model manga_ocr \
  --output outputs/predictions/manga_ocr.jsonl
python scripts/evaluate_baselines.py --manifest "$MANIFEST" --model paddle_manga \
  --output outputs/predictions/paddle_manga.jsonl
python scripts/evaluate_baselines.py --manifest "$MANIFEST" --model paddle_1_6 \
  --output outputs/predictions/paddle_1_6.jsonl
python scripts/evaluate_baselines.py --manifest "$MANIFEST" --model new_model \
  --model-id checkpoints/pilot-full/final \
  --output outputs/predictions/new_model.jsonl

python scripts/evaluate.py outputs/predictions/*.jsonl \
  --output-dir outputs/evaluation
```

Supported evaluator aliases:

| Alias | Default model |
|---|---|
| `manga_ocr` | `kha-white/manga-ocr-base` |
| `paddle_manga` | `jzhang533/PaddleOCR-VL-For-Manga` |
| `paddle_1_6` | `PaddlePaddle/PaddleOCR-VL-1.6` |
| `new_model` | explicit local path or `HF_MODEL_REPO` |

Outputs include raw per-model JSONL, `predictions.csv`, `metrics.json`,
`disagreements.csv`, and a local image-centric `report.html`. Reports distinguish
both correct/both wrong, either older model uniquely correct, new-model wins over
both, and pairwise regressions between the old/new Paddle manga models.

## Linguistik hard cases

The schema is ready under [`data/linguistik_hard_cases/`](data/linguistik_hard_cases/README.md).
Add only real, authorized problem crops. The known gold reading
`アナタ専用ウシ乳マヤでちゅよ〜` is included without a fabricated image. These
samples remain evaluation-only.

## Future Hugging Face Jobs

Before launch:

1. Install/update the CLI: `python -m pip install -U huggingface_hub`.
2. Run `hf auth login` locally; do not put the token in shell history.
3. Accept the gated Manga109-s terms.
4. Create a private `<HF_USER>/PaddleOCR-VL-1.6-For-Manga` model repository.
5. Replace `<HF_USER>` below. The GitHub repository must be public or readable
   without placing a GitHub credential in the command.

### 1. Gated-dataset preflight

The local fixture tests do not prove the Job identity/token can mount the gated
repository or that its real archive layout is recognized. Run this first. It
uses the verified `cpu-basic` Jobs flavor, validates token access and the mount,
streams the lazily mounted archive into regular ephemeral storage, validates its
byte count, ZIP structure, and SHA-256, then extracts it. It counts every usable
region with the deterministic split and loads a real crop from each split
without materializing the crop dataset. Staging avoids random-access ZIP reads
directly against the lazy repository mount; no Manga109-s data is uploaded.

> **WARNING: THIS STARTS BILLABLE CPU COMPUTE.** At the current $0.01/hour rate,
> the two-hour timeout is a hard compute ceiling of about $0.02.

```bash
hf jobs run \
  --name paddleocr-vl-1-6-manga-data-preflight \
  --flavor cpu-basic \
  --timeout 2h \
  --secrets HF_TOKEN \
  --env MANGA109_ROOT=/data/manga109s \
  --volume hf://datasets/hal-utokyo/Manga109-s:/data/manga109s:ro \
  python:3.11-slim \
  bash -lc 'set -euo pipefail; apt-get update >/dev/null; apt-get install -y --no-install-recommends ca-certificates git >/dev/null; git clone https://github.com/boomerkuigi/paddleocr-vl-1.6-manga-sft.git /workspace/project; cd /workspace/project; bash scripts/hf_dataset_preflight.sh'
```

### 2. One-step GPU smoke

Only after the CPU preflight succeeds, run the real-manifest one-step smoke.

> **WARNING: THIS STARTS BILLABLE GPU COMPUTE.**

```bash
hf jobs run \
  --name paddleocr-vl-1-6-manga-gpu-smoke \
  --flavor l4x1 \
  --timeout 3h \
  --secrets HF_TOKEN \
  --env PUSH_TO_HUB=0 \
  --env MANGA109_ROOT=/data/manga109s \
  --volume hf://datasets/hal-utokyo/Manga109-s:/data/manga109s:ro \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  python -c 'import io,os,subprocess,urllib.request,zipfile; z=zipfile.ZipFile(io.BytesIO(urllib.request.urlopen("https://codeload.github.com/boomerkuigi/paddleocr-vl-1.6-manga-sft/zip/refs/heads/main").read())); z.extractall("/workspace"); os.rename("/workspace/paddleocr-vl-1.6-manga-sft-main","/workspace/project"); os.chdir("/workspace/project"); subprocess.run(["bash","scripts/hf_job_entrypoint.sh","configs/gpu_smoke.yaml"],check=True)'
```

### 3. Twenty-step L4 timing run

This uses the full pilot's BF16 full-fine-tuning settings, including 16-way
gradient accumulation, but stops after 20 optimizer steps. It reports phase
timings, per-optimizer-step CUDA timings (excluding the first startup step from
the steady-state average), and peak allocated/reserved VRAM. It performs no
evaluation, recovery save, resume, or Hub upload because the real 2,500/500
intervals are not reached.

> **WARNING: THIS STARTS BILLABLE GPU COMPUTE.**

```bash
hf jobs run \
  --name paddleocr-vl-1-6-manga-l4-timing-20 \
  --flavor l4x1 \
  --timeout 3h \
  --secrets HF_TOKEN \
  --env PUSH_TO_HUB=0 \
  --env MANGA109_ROOT=/data/manga109s \
  --volume hf://datasets/hal-utokyo/Manga109-s:/data/manga109s:ro \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  python -c 'import io,os,subprocess,urllib.request,zipfile; z=zipfile.ZipFile(io.BytesIO(urllib.request.urlopen("https://codeload.github.com/boomerkuigi/paddleocr-vl-1.6-manga-sft/zip/refs/heads/main").read())); z.extractall("/workspace"); os.rename("/workspace/paddleocr-vl-1.6-manga-sft-main","/workspace/project"); os.chdir("/workspace/project"); subprocess.run(["bash","scripts/hf_job_entrypoint.sh","configs/l4_timing.yaml"],check=True)'
```

### 4. Three-epoch training

The training Job prepares the private data, trains, uploads rolling recovery
checkpoints, selects the best evaluated weights, uploads the final model, and
then terminates. It deliberately does not install or run baseline evaluators.

> **WARNING: THIS STARTS BILLABLE GPU COMPUTE.**

```bash
hf jobs run \
  --name paddleocr-vl-1-6-manga-pilot \
  --flavor l4x1 \
  --timeout 40h \
  --secrets HF_TOKEN \
  --env HF_MODEL_REPO=<HF_USER>/PaddleOCR-VL-1.6-For-Manga \
  --env MANGA109_ROOT=/data/manga109s \
  --volume hf://datasets/hal-utokyo/Manga109-s:/data/manga109s:ro \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  python -c 'import io,os,subprocess,urllib.request,zipfile; z=zipfile.ZipFile(io.BytesIO(urllib.request.urlopen("https://codeload.github.com/boomerkuigi/paddleocr-vl-1.6-manga-sft/zip/refs/heads/main").read())); z.extractall("/workspace"); os.rename("/workspace/paddleocr-vl-1.6-manga-sft-main","/workspace/project"); os.chdir("/workspace/project"); subprocess.run(["bash","scripts/hf_job_entrypoint.sh","configs/pilot.yaml"],check=True)'
```

The explicit 40-hour ceiling limits the worst case, and Hugging Face releases
compute when the process reaches a terminal state.

### 5. Held-out benchmark

Run this only after verifying the final model exists in the private destination.
It remounts Manga109-s and rebuilds crops ephemerally because the test data may
not be redistributed or persisted in a public repository. Each model is loaded
sequentially. Only gold-free predictions and aggregate metrics are uploaded.

> **WARNING: THIS STARTS BILLABLE GPU COMPUTE.**

```bash
hf jobs run \
  --name paddleocr-vl-1-6-manga-benchmark \
  --flavor l4x1 \
  --timeout 12h \
  --secrets HF_TOKEN \
  --env HF_MODEL_REPO=<HF_USER>/PaddleOCR-VL-1.6-For-Manga \
  --env MANGA109_ROOT=/data/manga109s \
  --volume hf://datasets/hal-utokyo/Manga109-s:/data/manga109s:ro \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  python -c 'import io,os,subprocess,urllib.request,zipfile; z=zipfile.ZipFile(io.BytesIO(urllib.request.urlopen("https://codeload.github.com/boomerkuigi/paddleocr-vl-1.6-manga-sft/zip/refs/heads/main").read())); z.extractall("/workspace"); os.rename("/workspace/paddleocr-vl-1.6-manga-sft-main","/workspace/project"); os.chdir("/workspace/project"); subprocess.run(["bash","scripts/hf_benchmark_entrypoint.sh"],check=True)'
```

## Runtime and cost estimates

These are planning estimates, not quotes or benchmarks of this code. They use
the older RTX 3060/~27-hour result as a practical anchor and current published
Hugging Face hourly rates. Dataset version, crop count, processor cost, and
kernel behavior can change them materially.

| GPU | Estimated 3-epoch runtime | Current rate | Estimated compute |
|---|---:|---:|---:|
| L4 24 GB | 18–30 h | $0.80/h | $14.40–$24.00 |
| A10G 24 GB (small) | 14–24 h | $1.00/h | $14.00–$24.00 |
| A10G 24 GB (large) | 13–22 h | $1.50/h | $19.50–$33.00 |
| A100 80 GB | 5–9 h | $2.50/h | $12.50–$22.50 |

The table is training-only. The less frequent validation cadence removes roughly
24–30 redundant full validation passes compared with the original 500-step
proposal; the broad training estimate is retained until the GPU timing smoke.
The separate L4 benchmark is provisionally estimated at 3–10 hours ($2.40–$8),
with a 12-hour/$9.60 hard ceiling. The training Job's 40-hour ceiling remains a
$32 maximum exposure, not an expected cost.

Start with the CPU dataset preflight, then the L4 24 GB one-step smoke. L4 has
BF16 support, more host RAM and disk than A10G-small, and the lowest listed 24 GB
rate. If full fine-tuning OOMs, record peak memory, retry with
`pilot_lora.yaml`, or move to A100 only after the failure is understood. See
[`docs/COSTS.md`](docs/COSTS.md).

## Output map

| Artifact | Location |
|---|---|
| Prepared local crops/manifests | `data/prepared/` |
| Split record | `data/prepared/manifests/split_summary.json` |
| Checkpoints and selected final model | `checkpoints/pilot-full/` |
| Raw baseline predictions | `outputs/predictions/*.jsonl` |
| Aggregate metrics | `outputs/evaluation/metrics.json` |
| Flat comparison | `outputs/evaluation/predictions.csv` |
| Disagreements | `outputs/evaluation/disagreements.csv` |
| Visual report | `outputs/evaluation/report.html` |

All data, caches, checkpoints, model binaries, outputs, credentials, and token-
like filenames are ignored by Git. No paid GPU action is performed by repository
setup or tests.
