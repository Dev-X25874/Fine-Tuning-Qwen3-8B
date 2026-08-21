# Fine-Tuning Qwen3-8B on GSM8K

LoRA fine-tuning of Qwen3-8B on GSM8K's official training split, evaluated against GSM8K's official test split.

## Results

| Model | GSM8K Accuracy (n=500) |
|---|---|
| Qwen3-8B (base) | 79.4% |
| Qwen3-8B (fine-tuned) | **83.8%** |

+4.4 points over the untouched base model. The evaluation set is GSM8K's public test split — an external, widely used benchmark not designed or tunable by this project. Training used only GSM8K's official training split (7,473 problems), which was verified to have zero string-level overlap with the 1,319-problem test split before training began (see `prepare_gsm8k_train.py`).

This result was reached after six training runs; two earlier runs regressed the base model's score. See [Benchmark Believers Welcome. Benchmark Skeptics, You Were Right.](https://sayak.bearblog.dev/benchmark-believers-welcome-benchmark-skeptics-you-were-right/) for the full account of what failed and why.

## Rationale

A self-constructed benchmark is a weak signal — the author controls its difficulty and can shape it, deliberately or not, toward a favorable result. GSM8K is external and standard, so an improvement here is harder to attribute to benchmark design and more likely to reflect a genuine capability change.

## Training Data

Training examples are GSM8K's own official, human-written problems and solutions. No LLM was used to generate, paraphrase, or rewrite any training example. The only transformation applied is reformatting the answer marker (`#### N`) to a consistent `Final answer: N` format used by the evaluation scripts; the underlying question, reasoning, and answer content is unmodified.

## Pipeline

| File | Role |
|---|---|
| `prepare_gsm8k_train.py` | Downloads GSM8K's official train split; verifies zero overlap with the test split before writing training data |
| `train_tinker.py` | LoRA fine-tuning on the Tinker API |
| `eval_gsm8k.py` | Evaluates a base or fine-tuned checkpoint against the official GSM8K test split |
| `generate_data.py` / `verify.py` | Synthetic symbolic math data generator and independent sympy-based verifier (used in earlier iterations; not part of the GSM8K result above) |
| `generate_word_problems.py` | Synthetic word-problem data generator with independent arithmetic verification (used in earlier iterations; not part of the GSM8K result above) |
| `eval_benchmark.py` | Evaluates against the synthetic benchmark, not GSM8K |

## Setup

```bash
pip install sympy tinker torch
export TINKER_API_KEY=your-key-here
```

## Reproduction

```bash
# 1. Prepare official GSM8K training data (verifies zero overlap with test split)
python prepare_gsm8k_train.py --out data/gsm8k_train.jsonl

# 2. Fine-tune
python train_tinker.py \
  --data data/gsm8k_train.jsonl \
  --model Qwen/Qwen3-8B \
  --epochs 3 \
  --rank 32 \
  --lr 2e-5 \
  --batch_size 64 \
  --out checkpoints/math_lora_v6

# 3. Evaluate against the official GSM8K test split
python eval_gsm8k.py \
  --model Qwen/Qwen3-8B \
  --checkpoint tinker://your-checkpoint-path \
  --n 500
```

## Known Failure Modes (Development History)

The first fine-tuning run reduced GSM8K accuracy from 83.0% to 76.0%. Training on a narrow, terse synthetic dataset shifted the model's general reasoning style, not just its math-specific knowledge. Two subsequent attempts to fix this also failed; one introduced a silent mismatch between the training-time and evaluation-time formatting, which masked the actual cause of the regression until it was identified and corrected.

Full write-up: [Benchmark Believers Welcome. Benchmark Skeptics, You Were Right.](https://sayak.bearblog.dev/benchmark-believers-welcome-benchmark-skeptics-you-were-right/)

## Post-Result Maintenance

After the result above was produced, the following defects were identified via static analysis (`py_compile`, `pyflakes`) and fixed:

- `eval_gsm8k.py`, `eval_benchmark.py`: `ZeroDivisionError` when zero examples score successfully
- `eval_benchmark.py`: `IndexError` in a debug print when the evaluation file has fewer than two rows
- `generate_word_problems.py`: unused variable (`remaining`) in `gen_money_shopping`
- `verify.py`: unused import (`re`)
- `generate_data.py`: stray `f`-string prefix with no interpolated values

None of these fixes touch the score-computation path in `eval_gsm8k.py`, and none of the fixed conditions occurred during the run that produced the results reported above. However, the result has **not been re-run against the patched code**. The numbers in this README should be attributed to the pre-patch version of the pipeline until `eval_gsm8k.py` is re-run against the saved checkpoint and the score is confirmed unchanged.

## Stack

Tinker (LoRA SFT) · GSM8K official train/test splits · Qwen3-8B (base model)
