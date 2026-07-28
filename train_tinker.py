"""
train_tinker.py

Supervised fine-tuning on Tinker using the verified math dataset produced by
generate_data.py. Uses the real Tinker SDK primitives: create_lora_training_client,
forward_backward, optim_step.

Requires: pip install tinker torch
Requires: TINKER_API_KEY set in your environment (Tinker reads it automatically).

Usage:
  python train_tinker.py --data data/all.jsonl --model Qwen/Qwen3-8B --epochs 1 --out checkpoints/math_lora
"""

import argparse
import json
import random
from pathlib import Path

import torch
import tinker
from tinker import types


def load_examples(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def build_datum(tokenizer, prompt: str, completion: str):
    """
    Turns one (prompt, completion) pair into a Tinker training Datum.
    Follows Tinker's documented convention exactly: model_input = ids[:-1],
    target_tokens = ids[1:] (same length, shifted by one), both wrapped as
    TensorData.from_torch(...). Loss is masked to 0 on prompt tokens so the
    model isn't penalized for tokens it didn't generate.
    """
    prompt_ids = tokenizer.encode(prompt + "\n")
    completion_ids = tokenizer.encode(completion)
    eos = tokenizer.eos_token_id
    if eos is not None:
        completion_ids = completion_ids + [eos]

    ids = prompt_ids + completion_ids
    if len(ids) < 2:
        return None  # degenerate example, skip

    model_input_ids = ids[:-1]
    target_ids = ids[1:]
    # weight 0 on prompt tokens (masked), weight 1 on completion tokens.
    # weights align with target_ids (i.e. position i predicts ids[i+1]),
    # so the mask boundary is at len(prompt_ids) - 1.
    boundary = max(len(prompt_ids) - 1, 0)
    weights = [0.0] * boundary + [1.0] * (len(target_ids) - boundary)

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=model_input_ids),
        loss_fn_inputs={
            "weights": types.TensorData.from_torch(torch.tensor(weights)),
            "target_tokens": types.TensorData.from_torch(torch.tensor(target_ids)),
        },
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="JSONL from generate_data.py")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--rank", type=int, default=32, help="LoRA rank")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="checkpoints/math_lora")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)

    examples = load_examples(args.data)
    print(f"Loaded {len(examples)} examples from {args.data}")

    service_client = tinker.ServiceClient()  # reads TINKER_API_KEY from env
    training_client = service_client.create_lora_training_client(
        base_model=args.model, rank=args.rank
    )
    tokenizer = training_client.get_tokenizer()

    print("Building training data (masked loss on completion tokens only)...")
    data = [build_datum(tokenizer, ex["prompt"], ex["completion"]) for ex in examples]
    data = [d for d in data if d is not None]

    # Diagnostic: check first few datums for anything that could cause nan loss
    print("\n--- DIAGNOSTIC: checking first 5 datums ---")
    for idx, d in enumerate(data[:5]):
        w = d.loss_fn_inputs["weights"]
        t = d.loss_fn_inputs["target_tokens"]
        w_list = list(w.data) if hasattr(w, "data") else list(w)
        t_list = list(t.data) if hasattr(t, "data") else list(t)
        m_ids = d.model_input.to_ints() if hasattr(d.model_input, "to_ints") else None
        print(f"datum {idx}: model_input_len={len(m_ids) if m_ids else 'NA'} "
              f"target_len={len(t_list)} weight_len={len(w_list)} "
              f"weight_sum={sum(w_list)} target_min={min(t_list)} target_max={max(t_list)}")
    print("--- END DIAGNOSTIC ---\n")

    step = 0
    for epoch in range(args.epochs):
        random.shuffle(data)
        for i in range(0, len(data), args.batch_size):
            batch = data[i : i + args.batch_size]

            fwd_bwd_future = training_client.forward_backward(
                batch, loss_fn="cross_entropy"
            )
            optim_future = training_client.optim_step(
                types.AdamParams(learning_rate=args.lr)
            )

            fwd_bwd_result = fwd_bwd_future.result()
            optim_future.result()

            step += 1
            if step % 10 == 0:
                loss_sum = fwd_bwd_result.metrics.get("loss:sum", None)
                if loss_sum is not None:
                    total_weight = sum(
                        sum(list(d.loss_fn_inputs["weights"].data)) for d in batch
                    )
                    loss = loss_sum / total_weight if total_weight > 0 else float("nan")
                else:
                    loss = float("nan")
                    print(f"  (available metric keys: {list(fwd_bwd_result.metrics.keys())})")
                print(f"epoch {epoch} step {step} loss {loss:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_result = training_client.save_weights_for_sampler(name=out_path.name).result()
    print(f"Saved checkpoint: {save_result}")


if __name__ == "__main__":
    main()