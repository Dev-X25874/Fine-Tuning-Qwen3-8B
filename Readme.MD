# Math Benchmark Fine-Tuning Pipeline

Fine-tuning Qwen3-8B on math problems where the training data is generated and graded entirely by symbolic computation. No LLM writes a single problem, answer, or reasoning step. If sympy can't solve it and independently re-verify it, it doesn't make it into the dataset.

## Results

**Overall:** baseline 10.8% → fine-tuned 69.7% (roughly 6.5x improvement)

| Domain | Baseline | Fine-tuned |
|---|---|---|
| Arithmetic | 55.3% | 100% |
| Linear equations | 9.0% | 99.3% |
| Integrals | 0% | 93.3% |
| Derivatives | 0.7% | 81.7% |
| Quadratic equations | 0% | 32.0% |
| Systems of equations | 0% | 12.0% |

Measured on an eval set with zero overlap with training data, verified directly, not assumed. The base model already had real arithmetic ability but was essentially at zero for anything requiring multi-step symbolic manipulation. Fine-tuning brought every domain up, including ones the base model couldn't touch at all.

## Why zero LLM involvement matters

Every problem is built from a randomized sympy template. The ground-truth answer and the step-by-step reasoning trace both come directly out of sympy's symbolic engine (`diff`, `integrate`, `solve`), not from a language model guessing what the steps should look like. Every example is then independently re-verified: the verifier re-parses the problem from scratch and re-solves it using a separate code path than the generator, so a bug in one can't silently poison the other.

## Pipeline files

| File | Role |
|---|---|
| `generate_data.py` | sympy-templated problems, symbolically solved and verified |
| `verify.py` | independent from-scratch re-derivation, the actual gate |
| `train_tinker.py` | LoRA SFT on Tinker, real `forward_backward` and `optim_step` |
| `eval_benchmark.py` | exact-match plus symbolic-equivalence grading |

## Setup

```bash
pip install sympy tinker torch
export TINKER_API_KEY=your-key-here
```

## Run order

```bash
# 1. Generate train + eval sets — eval is hard-deduped against train
python generate_data.py --domain all --n 5000 --out data/train_v2.jsonl --seed 0
python generate_data.py --domain all --n 300 --out data/eval_v2.jsonl --seed 999 --exclude-file data/train_v2.jsonl

# 2. Baseline eval
python eval_benchmark.py --model Qwen/Qwen3-8B --data data/eval_v2.jsonl --max_tokens 1500

# 3. Fine-tune
python train_tinker.py --data data/train_v2.jsonl --model Qwen/Qwen3-8B --epochs 1 --rank 32 --lr 2e-5 --batch_size 128 --out checkpoints/math_lora_v2

# 4. Eval the fine-tuned checkpoint
python eval_benchmark.py --model Qwen/Qwen3-8B --checkpoint tinker://your-checkpoint-path --data data/eval_v2.jsonl --max_tokens 1500
```

## The plot twist: how I almost fooled myself

An earlier version of this pipeline hit 85.1%. Great number. Wrong number. A self-audit found that the small parameter ranges in the data generators — for example, quadratic coefficients only spanning -8 to 8 — meant the "held-out" eval set, generated with a different random seed, was not actually held out. Up to **99.3%** of the eval problems for some domains were byte-identical to problems already seen during training.

The tell was in the data itself: the one domain with genuinely zero overlap, `linear_system`, also happened to be the worst-scoring domain both times. That's a strong signal the other domains' high scores were partly memorization, not generalization.

Fixed by widening the parameter spaces and adding a hard `--exclude-file` dedup check that verifies zero overlap before training even starts, instead of just hoping the ranges are wide enough. Post-fix score dropped to a less impressive but more honest 69.7%.

## Known limitations

- Systems of equations and quadratic equations remain the weakest domains — likely genuinely harder to generalize on with the current data volume, not a leakage artifact this time (verified: 0/1800 overlap).
- Roughly 1.6% of systems-of-equations training examples include unsolvable systems (parallel lines) that get labeled with an empty solution set. Known minor data quality issue, not yet patched in the generator.

## Stack

Tinker for LoRA supervised fine-tuning, sympy for ground truth and verification, Qwen3-8B as the base model.
