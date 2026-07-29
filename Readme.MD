# Fine-Tuning Qwen3-8B on Real Math (GSM8K, Not My Own Benchmark)

I built my own math benchmark first. It was fun. It was also mine — a benchmark you build yourself is always going to be easier to beat than a real one, because you control the difficulty. So instead of shipping that as the headline result, I went and tested against something I don't control: GSM8K, a widely-used, saturated, globally-recognized math benchmark. If a model does well there, that actually means something.

## The result

| | GSM8K (n=500) |
|---|---|
| Base Qwen3-8B | 79.4% |
| Fine-tuned | **83.8%** |

A real 4.4 point improvement over the untouched base model, on a benchmark I didn't design and can't quietly make easier. Trained purely on GSM8K's own official training split — 7,473 problems, completely separate from the 1,319 test problems the score above is measured on, verified with zero overlap by direct comparison, not assumed.

This wasn't the first attempt. It took six training runs to get here, including two that made the model *worse* than the untouched base model before I found what was actually wrong. Full story of what broke and why: [Benchmark Believers Welcome. Benchmark Skeptics, You Were Right.](https://sayak.bearblog.dev/benchmark-believers-welcome-benchmark-skeptics-you-were-right/)

## Why GSM8K over a self-made benchmark

Anyone can build a benchmark their own model does well on. That doesn't prove much. GSM8K is external, standard, and used everywhere — if a fine-tune moves the needle here, it's a real signal, not a number I engineered to look good.

## How the training data works

No LLM writes any part of this. The training data is GSM8K's own official, human-written training problems and solutions — not synthetic, not LLM-generated, not paraphrased. I only reformatted the answer marker to match a consistent output format, nothing about the actual math or reasoning content was touched.

## Pipeline

| File | Role |
|---|---|
| `prepare_gsm8k_train.py` | fetches GSM8K's official train split, verifies zero overlap with the test split |
| `train_tinker.py` | LoRA fine-tuning on Tinker |
| `eval_gsm8k.py` | evaluates against real GSM8K test problems |

## Setup

```bash
pip install sympy tinker torch
export TINKER_API_KEY=your-key-here
```

## Run order

```bash
# 1. Prepare official GSM8K training data (verified zero overlap with test set)
python prepare_gsm8k_train.py --out data/gsm8k_train.jsonl

# 2. Fine-tune
python train_tinker.py --data data/gsm8k_train.jsonl --model Qwen/Qwen3-8B --epochs 3 --rank 32 --lr 2e-5 --batch_size 64 --out checkpoints/math_lora_v6

# 3. Evaluate against real GSM8K
python eval_gsm8k.py --model Qwen/Qwen3-8B --checkpoint tinker://your-checkpoint-path --n 500
```

## What actually went wrong along the way

The first fine-tune made the model *worse* at GSM8K, not better — 83.0% down to 76.0%. Training on a narrow, terse dataset had quietly retrained the model's whole reasoning style, not just its math knowledge. Two more attempts at fixing it also failed, including one where I accidentally introduced a silent mismatch between how the model was trained versus how it was evaluated. Fixing that mismatch is what finally turned a regression into a real win.

Full breakdown of every failed attempt: [Benchmark Believers Welcome. Benchmark Skeptics, You Were Right.](https://sayak.bearblog.dev/benchmark-believers-welcome-benchmark-skeptics-you-were-right/)

## Stack

Tinker (LoRA SFT), GSM8K official train/test splits, Qwen3-8B (base model).