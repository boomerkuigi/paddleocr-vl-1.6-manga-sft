# Porting the proven manga recipe to PaddleOCR-VL 1.6

## What remains valid

The RTX 3060 reproduction established a useful task recipe: Manga109-s
text-region crops; an `OCR:` prompt; full BF16 fine-tuning; batch size 1 with
gradient accumulation; gradient checkpointing; three epochs; cosine scheduling;
and evaluation with sentence exact match and character error rate. Its reported
12 GB fit is meaningful evidence that a 24 GB attempt is reasonable.

## What changes for 1.6

| Area | Older manga reproduction | This 1.6 port |
|---|---|---|
| Base model | `PaddlePaddle/PaddleOCR-VL` | Pinned `PaddlePaddle/PaddleOCR-VL-1.6` revision |
| Auto class | causal-LM-era custom loading | `AutoModelForImageTextToText` |
| Assistant boundary | older `Assistant: ` assumption | exact 1.6 template marker `Assistant:\n` |
| Training API | custom loss/backward Trainer | standard Transformers label loss and Trainer |
| Split | crop-level random 90/10 | book-grouped deterministic 80/10/10 |
| Validation/test | one held-out set | validation for model selection; untouched test for final comparison |
| Metric text | whitespace-stripped headline | raw headline; normalized values secondary and labeled |
| Processor limits | older defaults | official 1.6 min/max pixel defaults recorded in config |
| Sequence handling | fixed max length | hard failure instead of silent target truncation |
| Attention | no FlashAttention on RTX 3060 | portable SDPA first; no FlashAttention dependency |
| Base revision | floating | immutable commit SHA |

The official 1.6 model is about 1B parameters, BF16, and uses an 18-layer text
component plus a 27-layer vision component. It is architecture-compatible with
1.5 according to the model card. The processor uses a 14-pixel patch size and a
2x spatial merge; its current default pixel bounds are captured in the config.

`megemini/PaddleOCR-VL-KIE` instead uses Paddle/ERNIEKit, a 16K context, packing,
sparse FlashAttention, BF16 O2, recomputation, and sharded unified checkpoints.
Those choices demonstrate full 1.6 fine-tuning on A100, but are not automatically
appropriate for short manga crops or a low-cost single 24 GB GPU. The pilot uses
the official Transformers implementation because it matches the established
manga workflow, supports the desired four-model evaluator, and keeps deployment
simple.

## Method decision

Full fine-tuning is the first experiment. It has the strongest direct evidence
for this task and for 1.6. LoRA is a prepared fallback if the one-step GPU run
demonstrates that optimizer/activation memory exceeds 24 GB. QLoRA adds custom
vision-language quantization and export risk without evidence that it improves
this OCR task, so it is not part of the pilot.

The chosen `1e-5` learning rate is deliberately between the older manga recipe's
`2e-5` and the 1.6 KIE example's `5e-6`. It is a hypothesis to validate, not a
claim of optimality.

