"""
eval_gsm8k.py

Evaluates a Tinker model (base or fine-tuned checkpoint) against GSM8K --
a standard, external, widely-used grade-school math word problem benchmark.
Unlike our synthetic sympy-based benchmark, this tests generalization to a
completely different problem style (natural language word problems) that
the model was NOT specifically trained on.

Data source: OpenAI's official GSM8K test set (1319 problems), fetched from
https://github.com/openai/grade-school-math

Usage:
  python eval_gsm8k.py --model Qwen/Qwen3-8B --n 200
  python eval_gsm8k.py --model Qwen/Qwen3-8B --checkpoint "tinker://..." --n 200
"""

import argparse
import json
import re
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import tinker
from tinker import types

GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"


def load_gsm8k(path="gsm8k_test.jsonl", n=None, seed=0):
    if not Path(path).exists():
        import urllib.request
        print(f"Downloading GSM8K test set to {path}...")
        urllib.request.urlretrieve(GSM8K_URL, path)

    examples = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            gold = ex["answer"].split("####")[-1].strip().replace(",", "")
            examples.append({"question": ex["question"], "gold": gold})

    if n is not None and n < len(examples):
        random.seed(seed)
        examples = random.sample(examples, n)
    return examples


def extract_number(text: str) -> str:
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        nums = re.findall(r"-?\d[\d,]*\.?\d*", boxed[-1])
        if nums:
            return nums[-1].replace(",", "")
    for line in text.splitlines()[::-1]:
        if "answer" in line.lower():
            nums = re.findall(r"-?\d[\d,]*\.?\d*", line)
            if nums:
                return nums[-1].replace(",", "")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "") if nums else ""


def numbers_match(predicted: str, gold: str) -> bool:
    try:
        return abs(float(predicted) - float(gold)) < 1e-4
    except (ValueError, TypeError):
        return predicted.strip() == gold.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max_tokens", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    examples = load_gsm8k(n=args.n, seed=args.seed)
    print(f"Evaluating on {len(examples)} GSM8K problems")

    service_client = tinker.ServiceClient()
    if args.checkpoint:
        sampling_client = service_client.create_sampling_client(
            base_model=args.model, model_path=args.checkpoint
        )
    else:
        sampling_client = service_client.create_sampling_client(base_model=args.model)

    tokenizer = sampling_client.get_tokenizer()

    def run_one(ex):
        prompt = f"{ex['question']}\nSolve step by step, then end with: Final answer: <number>"
        prompt_ids = tokenizer.encode(prompt + "\n")
        sample = sampling_client.sample(
            prompt=types.ModelInput.from_ints(tokens=prompt_ids),
            sampling_params=types.SamplingParams(max_tokens=args.max_tokens, temperature=0.0),
            num_samples=1,
        ).result()
        output_text = tokenizer.decode(sample.sequences[0].tokens)
        output_text = output_text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
        predicted = extract_number(output_text)
        correct = numbers_match(predicted, ex["gold"])
        return correct

    correct_count = 0
    total = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(run_one, ex) for ex in examples]
        for i, fut in enumerate(as_completed(futures)):
            try:
                correct = fut.result()
                correct_count += int(correct)
                total += 1
            except Exception:
                errors += 1
            if i % 20 == 0:
                print(f"  {i}/{len(examples)} done...")

    print(f"\nGSM8K results for {args.model}" + (f" + {args.checkpoint}" if args.checkpoint else " (base)"))
    print(f"  {correct_count}/{total}  ({100*correct_count/total:.1f}%)")
    if errors:
        print(f"  ({errors} examples errored and were skipped)")


if __name__ == "__main__":
    main()