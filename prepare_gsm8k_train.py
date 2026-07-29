"""
prepare_gsm8k_train.py

Downloads GSM8K's OFFICIAL training split (7473 problems, completely
separate from the 1319-problem test split we evaluate on -- verified zero
overlap by exact question-string comparison) and converts it into our
training schema.

This is NOT training on the test set. GSM8K's train/test split is the
standard, intended way to fine-tune on this benchmark's distribution --
train on train.jsonl, evaluate on test.jsonl, exactly like any other ML
benchmark.

The original human-written step-by-step reasoning is preserved (not
regenerated or shortened) -- only the inline <<calc=result>> annotations
are stripped and the "#### N" answer marker is reformatted to match our
"Final answer: N" convention so eval_benchmark.py's grading logic works
on it too.

Usage:
  python prepare_gsm8k_train.py --out data/gsm8k_train.jsonl
"""

import argparse
import json
import re
import urllib.request
from pathlib import Path

TRAIN_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
TEST_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw-cache", default="gsm8k_train_official.jsonl")
    ap.add_argument("--test-cache", default="gsm8k_test.jsonl",
                     help="used only to double-check zero overlap, not for training")
    args = ap.parse_args()

    if not Path(args.raw_cache).exists():
        print(f"Downloading official GSM8K train split to {args.raw_cache}...")
        urllib.request.urlretrieve(TRAIN_URL, args.raw_cache)

    if not Path(args.test_cache).exists():
        print(f"Downloading official GSM8K test split to {args.test_cache} (for overlap check only)...")
        urllib.request.urlretrieve(TEST_URL, args.test_cache)

    train_questions = set()
    out_examples = []
    with open(args.raw_cache) as f:
        for line in f:
            ex = json.loads(line)
            question = ex["question"]
            answer_text = ex["answer"]
            reasoning = re.sub(r"<<[^>]*>>", "", answer_text)
            gold = reasoning.split("####")[-1].strip().replace(",", "")
            reasoning_only = reasoning.split("####")[0].strip()
            prompt = f"{question}\nSolve step by step, then end with: Final answer: <number>"
            completion = reasoning_only + f"\nFinal answer: {gold}"
            out_examples.append({
                "prompt": prompt,
                "completion": completion,
                "final_answer": gold,
                "domain": "gsm8k_word_problem",
            })
            train_questions.add(question)

    test_questions = set()
    with open(args.test_cache) as f:
        for line in f:
            test_questions.add(json.loads(line)["question"])

    overlap = train_questions & test_questions
    print(f"Overlap check: {len(overlap)} questions shared between train and test (should be 0)")
    if overlap:
        raise RuntimeError("Train/test overlap detected -- aborting, do not use this data.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for ex in out_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(out_examples)} verified-non-overlapping GSM8K train examples to {out_path}")


if __name__ == "__main__":
    main()