"""
eval_benchmark.py

Evaluates a (base or fine-tuned) model against a held-out, symbolically-verified
math set. Grading matches against the machine-computed final_answer,
not vibes -- so your ~1% deltas are real.

Usage:
  # generate a held-out eval set (use a different --seed than training data!)
  python generate_data.py --domain all --n 100 --out data/eval.jsonl --seed 999

  # eval the base model
  python eval_benchmark.py --model Qwen/Qwen3-8B --data data/eval.jsonl

  # eval your fine-tuned checkpoint
  python eval_benchmark.py --model Qwen/Qwen3-8B --checkpoint checkpoints/math_lora --data data/eval.jsonl
"""

import argparse
import json
import re

import tinker
from tinker import types


def extract_final(text: str) -> str:
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        return boxed[-1].strip()
    for line in text.splitlines()[::-1]:
        if "final answer" in line.lower():
            return line.split(":", 1)[-1].strip()
    return text.strip().splitlines()[-1] if text.strip() else ""


def answers_match(predicted: str, expected: str) -> bool:
    """Loose comparison: strips f'(x)=/x=/F(x)=...+C style prefixes and
    normalizes ^ to ** before comparing exactly, then falls back to
    sympy symbolic equality."""
    def normalize(s: str) -> str:
        s = s.strip().strip("$").strip()
        s = re.sub(r"^f'\(x\)\s*=\s*", "", s)
        s = re.sub(r"^F\(x\)\s*=\s*", "", s)
        s = re.sub(r"^x\s*=\s*", "", s)
        s = re.sub(r"\s*\+\s*C\s*$", "", s)
        s = s.replace("^", "**")
        return s.strip()

    p, e = normalize(predicted), normalize(expected)
    if p == e:
        return True
    try:
        import sympy as sp
        x, y = sp.symbols("x y")
        return sp.simplify(sp.sympify(p) - sp.sympify(e)) == 0
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", default=None, help="path saved by train_tinker.py")
    ap.add_argument("--data", required=True)
    ap.add_argument("--max_tokens", type=int, default=200)
    args = ap.parse_args()

    with open(args.data) as f:
        examples = [json.loads(line) for line in f]

    service_client = tinker.ServiceClient()
    if args.checkpoint:
        sampling_client = service_client.create_sampling_client(
            base_model=args.model, model_path=args.checkpoint
        )
    else:
        sampling_client = service_client.create_sampling_client(base_model=args.model)

    tokenizer = sampling_client.get_tokenizer() if hasattr(sampling_client, "get_tokenizer") else None

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def run_one(ex):
        prompt_ids = tokenizer.encode(ex["prompt"] + "\n") if tokenizer else None
        sample = sampling_client.sample(
            prompt=types.ModelInput.from_ints(tokens=prompt_ids),
            sampling_params=types.SamplingParams(max_tokens=args.max_tokens, temperature=0.0),
            num_samples=1,
        ).result()
        output_text = tokenizer.decode(sample.sequences[0].tokens) if tokenizer else str(sample)
        output_text = output_text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
        predicted = extract_final(output_text)
        correct = answers_match(predicted, ex["final_answer"])
        if ex is examples[0] or ex is examples[1]:
            print(f"\nDBG | OUTPUT: {output_text[:200]!r} | PREDICTED: {predicted!r} | EXPECTED: {ex['final_answer']!r}\n")
        return ex["domain"], correct

    results_by_domain = {}
    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = [executor.submit(run_one, ex) for ex in examples]
        for i, fut in enumerate(as_completed(futures)):
            d, correct = fut.result()
            results_by_domain.setdefault(d, [0, 0])
            results_by_domain[d][0] += int(correct)
            results_by_domain[d][1] += 1
            if i % 20 == 0:
                print(f"  {i}/{len(examples)} done...")

    print(f"\nResults for {args.model}" + (f" + {args.checkpoint}" if args.checkpoint else " (base)"))
    total_correct, total_n = 0, 0
    for d, (c, n) in sorted(results_by_domain.items()):
        print(f"  {d:20s} {c}/{n}  ({100*c/n:.1f}%)")
        total_correct += c
        total_n += n
    print(f"  {'TOTAL':20s} {total_correct}/{total_n}  ({100*total_correct/total_n:.1f}%)")


if __name__ == "__main__":
    main()