# Math Benchmark Fine-Tuning Pipeline 

Fine-tuning Qwen3-8B on math problems where the training data is generated
and graded entirely by symbolic computation — no LLM writes a single problem,
answer, or reasoning step. If sympy can't solve it and re-verify it
independently, it doesn't make it into the dataset.

## TL;DR Results

| | Baseline (Qwen3-8B) | Fine-tuned (LoRA) |
|---|---|---|
| **Overall** | 10.8% | **69.7%** |

Per-domain breakdown (fine-tuned):

| Domain | Score |
|---|---|
| Arithmetic | 100% |
| Linear equations | 99.3% |
| Integrals | 93.3% |
| Derivatives | 81.7% |
| Quadratic equations | 32.0% |
| Systems of equations | 12.0% |

A ~6.5x improvement, measured on an eval set with **zero overlap** with
training data (verified by direct string comparison, not assumed).

## Why zero LLM involvement matters

Every problem is built from a randomized sympy template. The ground-truth
answer *and* the step-by-step reasoning trace both come directly out of
sympy's symbolic engine — `diff()`, `integrate()`, `solve()` — not from a
language model imagining what the steps should look like. Every single
example is then **independently re-verified**: the verifier re-parses the
problem from scratch and re-solves it using a separate code path than the
generator, so a bug in one can't silently poison the other.

This means the training data can't hallucinate. It can still have *bugs*
(see below) — but it can't lie about what 2+2 is.

## Pipeline
generate_data.py → sympy-templated problems, symbolically solved & verified
verify.py → independent from-scratch re-derivation (the actual gate)
train_tinker.py → LoRA SFT on Tinker (real forward_backward/optim_step)
eval_benchmark.py → exact-match + symbolic-equivalence grading
## Setup

```bash
pip install sympy tinker torch
export TINKER_API_KEY="your-key-here"
```

## Run order

```bash
# generate train + eval sets — eval is HARD-DEDUPED against train
python generate_data.py --domain all --n 5000 --out data/train_v2.jsonl --seed 0
python generate_data.py --domain all --n 300 --out data/eval_v2.jsonl --seed 999 \
    --exclude-file data/train_v2.jsonl

# baseline
python eval_benchmark.py --model Qwen/Qwen3-8B --data data/eval_v2.jsonl --max_tokens 1500

# fine-tune
python train_tinker.py --data data/train_v2.jsonl --model Qwen/Qwen3-8B \
    --epochs 1 --rank 32 --lr 2e-5 --batch_size 128 --out checkpoints/math_lora_v2

# eval the checkpoint
python eval_benchmark.py --model Qwen/Qwen3-8B \
    --checkpoint "tinker://<your-checkpoint-path>" \
    --data data/eval_v2.jsonl --max_tokens 1500
```

## The plot twist: how I almost fooled myself

An earlier version of this pipeline hit **85.1%**. Great number. Wrong
number. A self-audit found that the small parameter ranges in the data
generators (e.g. quadratic coefficients only spanning `-8` to `8`) meant the
"held-out" eval set — generated with a different random seed — wasn't
actually held out. Up to **99.3%** of the eval problems for some domains
were byte-identical to problems already seen during training.

The tell was in the data itself: the one domain with genuinely zero overlap
(`linear_system`) also happened to be the *worst*-scoring domain both times —
a strong signal that the other domains' high scores were partly memorization,
not generalization.

Fixed by widening the parameter spaces and adding a hard `--exclude-file`
dedup check that verifies zero overlap before training even starts — not
just hoping the ranges are wide enough. Post-fix score dropped to a much
less impressive, much more *honest* 69.7%. That's the number in the table
above.

## Known limitations

- `linear_system` and `quadratic_equation` remain the weakest domains —
  likely genuinely harder for the model to generalize on with the current
  data volume, not a leakage artifact this time (verified: 0/1800 overlap).
- ~1.6% of `linear_system` training examples include unsolvable systems
  (parallel lines) that get labeled with an empty solution set — a known
  minor data quality issue, not yet patched in the generator.

## Stack

Tinker (LoRA SFT), sympy (ground truth + verification), Qwen3-8B (base model).## Setup

```bash
pip install sympy tinker torch
export TINKER_API_KEY="your-key-here"
```

## Run order

```bash
# generate train + eval sets — eval is HARD-DEDUPED against train
python generate_data.py --domain all --n 5000 --out data/train_v2.jsonl --seed 0
python generate_data.py --domain all --n 300 --out data/eval_v2.jsonl --seed 999 \
    --exclude-file data/train_v2.jsonl

# baseline
python eval_benchmark.py --model Qwen/Qwen3-8B --data data/eval_v2.jsonl --max_tokens 1500

# fine-tune
python train_tinker.py --data data/train_v2.jsonl --model Qwen/Qwen3-8B \
    --epochs 1 --rank 32 --lr 2e-5 --batch_size 128 --out checkpoints/math_lora_v2

# eval the checkpoint
python eval_benchmark.py --model Qwen/Qwen3-8B \
    --checkpoint "tinker://<your-checkpoint-path>" \
    --data data/eval_v2.jsonl --max_tokens 1500
```

## The plot twist: how I almost fooled myself

An earlier version of this pipeline hit **85.1%**. Great number. Wrong
number. A self-audit found that the small parameter ranges in the data
generators (e.g. quadratic coefficients only spanning `-8` to `8`) meant the
"held-out" eval set — generated with a different random seed — wasn't
actually held out. Up to **99.3%** of the eval problems for some domains
were byte-identical to problems already seen during training.

The tell was in the data itself: the one domain with genuinely zero overlap
(`linear_system`) also happened to be the *worst*-scoring domain both times —
a strong signal that the other domains' high scores were partly memorization,
not generalization.

Fixed by widening the parameter spaces and adding a hard `--exclude-file`
dedup check that verifies zero overlap before training even starts — not
just hoping the ranges are wide enough. Post-fix score dropped to a much
less impressive, much more *honest* 69.7%. That's the number in the table
above.

## Known limitations

- `linear_system` and `quadratic_equation` remain the weakest domains —
  likely genuinely harder for the model to generalize on with the current
  data volume, not a leakage artifact this time (verified: 0/1800 overlap).
- ~1.6% of `linear_system` training examples include unsolvable systems
  (parallel lines) that get labeled with an empty solution set — a known
  minor data quality issue, not yet patched in the generator.

## Stack

Tinker (LoRA SFT), sympy (ground truth + verification), Qwen3-8B (base model).
