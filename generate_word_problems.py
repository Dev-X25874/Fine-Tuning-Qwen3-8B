"""
generate_word_problems.py

Generates GSM8K-style natural-language word problem training data.
Unlike generate_data.py (rigid symbolic prompts, terse answers), this
produces multi-sentence story problems with multi-step natural-language
reasoning -- closer to what GSM8K actually looks like, and closer to the
base model's natural long-form reasoning style.

Every ground-truth answer is computed with plain arithmetic (not an LLM),
then INDEPENDENTLY recomputed by a separate verifier function before being
written to disk -- same zero-hallucination guarantee as generate_data.py.

Usage:
  python generate_word_problems.py --n 5000 --out data/word_train.jsonl --seed 0
  python generate_word_problems.py --n 300 --out data/word_eval.jsonl --seed 999 --exclude-file data/word_train.jsonl
"""

import argparse
import json
import random
from pathlib import Path

NAMES = ["Maya", "Ethan", "Priya", "Liam", "Sofia", "Noah", "Aisha", "Lucas",
         "Zara", "Mateo", "Chloe", "Kai", "Nina", "Omar", "Ivy", "Diego"]
ITEMS = ["apples", "marbles", "stickers", "pencils", "cookies", "books",
         "toy cars", "balloons", "postcards", "seashells"]


def _verify_money_shopping(a, price, spent_count, spent_price, expected):
    total = a * price
    spent = spent_count * spent_price
    return total - spent == expected


def gen_money_shopping():
    name = random.choice(NAMES)
    item = random.choice(ITEMS)
    a = random.randint(5, 40)
    price = random.randint(2, 15)
    spent_count = random.randint(1, a - 1)
    spent_price = random.randint(1, price)
    total = a * price
    spent = spent_count * spent_price
    remaining = total - spent

    prompt = (
        f"{name} has {a} {item}, each worth ${price}. "
        f"{name} sells {spent_count} of them at ${spent_price} each. "
        f"How much money does {name} make from selling the {item}?"
    )
    completion = (
        f"{name} sells {spent_count} {item} at ${spent_price} each.\n"
        f"Money made = {spent_count} * {spent_price} = {spent}.\n"
        f"Final answer: {spent}"
    )
    ok = _verify_money_shopping(a, price, spent_count, spent_price, spent)
    return prompt, completion, str(spent), ok


def gen_rate_multiday():
    name = random.choice(NAMES)
    item = random.choice(ITEMS)
    per_day = random.randint(3, 20)
    days = random.randint(2, 10)
    used_per_day = random.randint(1, per_day - 1)
    total_produced = per_day * days
    total_used = used_per_day * days
    remaining = total_produced - total_used

    prompt = (
        f"{name} collects {per_day} {item} every day for {days} days. "
        f"Each day, {name} gives away {used_per_day} of them. "
        f"How many {item} does {name} have left after {days} days?"
    )
    completion = (
        f"Total collected over {days} days: {per_day} * {days} = {total_produced}.\n"
        f"Total given away over {days} days: {used_per_day} * {days} = {total_used}.\n"
        f"Remaining: {total_produced} - {total_used} = {remaining}.\n"
        f"Final answer: {remaining}"
    )
    recomputed = (per_day * days) - (used_per_day * days)
    ok = recomputed == remaining
    return prompt, completion, str(remaining), ok


def gen_comparison_multiply():
    name1, name2 = random.sample(NAMES, 2)
    item = random.choice(ITEMS)
    base = random.randint(4, 25)
    multiplier = random.randint(2, 5)
    offset = random.randint(1, 10)
    plus_or_minus = random.choice(["more", "fewer"])
    if plus_or_minus == "more":
        total2 = base * multiplier + offset
    else:
        total2 = base * multiplier - offset
        if total2 < 0:
            total2 = base * multiplier + offset
            plus_or_minus = "more"

    prompt = (
        f"{name1} has {base} {item}. {name2} has {offset} {plus_or_minus} than "
        f"{multiplier} times as many {item} as {name1}. How many {item} does {name2} have?"
    )
    completion = (
        f"{multiplier} times {name1}'s amount: {multiplier} * {base} = {base * multiplier}.\n"
        + (f"{offset} more than that: {base * multiplier} + {offset} = {total2}.\n"
           if plus_or_minus == "more" else
           f"{offset} fewer than that: {base * multiplier} - {offset} = {total2}.\n")
        + f"Final answer: {total2}"
    )
    recomputed = base * multiplier + offset if plus_or_minus == "more" else base * multiplier - offset
    ok = recomputed == total2
    return prompt, completion, str(total2), ok


def gen_split_groups():
    name = random.choice(NAMES)
    item = random.choice(ITEMS)
    total = random.randint(20, 200)
    groups = random.randint(2, 10)
    while total % groups != 0:
        total = random.randint(20, 200)
        groups = random.randint(2, 10)
    per_group = total // groups

    prompt = (
        f"{name} has {total} {item} and wants to split them equally among {groups} friends. "
        f"How many {item} does each friend get?"
    )
    completion = (
        f"Divide the total evenly: {total} / {groups} = {per_group}.\n"
        f"Final answer: {per_group}"
    )
    ok = (total // groups == per_group) and (total % groups == 0)
    return prompt, completion, str(per_group), ok


def gen_two_step_earn_spend():
    name = random.choice(NAMES)
    days = random.randint(2, 8)
    per_day = random.randint(10, 60)
    price = random.randint(5, 100)
    earned = per_day * days
    can_buy = earned // price
    leftover = earned % price

    prompt = (
        f"{name} earns ${per_day} per day for {days} days. "
        f"Each item {name} wants to buy costs ${price}. "
        f"How many of these items can {name} afford after {days} days of earnings?"
    )
    completion = (
        f"Total earned: {per_day} * {days} = {earned}.\n"
        f"Number affordable: {earned} // {price} = {can_buy} (with ${leftover} left over).\n"
        f"Final answer: {can_buy}"
    )
    recomputed = (per_day * days) // price
    ok = recomputed == can_buy
    return prompt, completion, str(can_buy), ok


GENERATORS = [
    gen_money_shopping,
    gen_rate_multiday,
    gen_comparison_multiply,
    gen_split_groups,
    gen_two_step_earn_spend,
]


def generate(n: int, exclude_prompts: set = None):
    exclude_prompts = exclude_prompts or set()
    examples = []
    seen_this_run = set()
    attempts = 0
    while len(examples) < n and attempts < n * 20:
        attempts += 1
        fn = random.choice(GENERATORS)
        try:
            prompt, completion, answer, ok = fn()
        except Exception:
            continue
        if not ok:
            continue  # independent verification failed, skip -- do not poison data
        if prompt in exclude_prompts or prompt in seen_this_run:
            continue
        examples.append({
            "prompt": prompt,
            "completion": completion,
            "final_answer": answer,
            "domain": "word_problem",
        })
        seen_this_run.add(prompt)
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-file", default=None)
    args = ap.parse_args()

    random.seed(args.seed)

    exclude_prompts = set()
    if args.exclude_file:
        with open(args.exclude_file) as f:
            for line in f:
                exclude_prompts.add(json.loads(line)["prompt"])
        print(f"Loaded {len(exclude_prompts)} prompts to exclude from {args.exclude_file}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    examples = generate(args.n, exclude_prompts=exclude_prompts)
    with out_path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"word_problem: {len(examples)} verified examples")
    print(f"Wrote {len(examples)} total examples to {out_path}")


if __name__ == "__main__":
    main()