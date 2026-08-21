"""
generate_data.py — v2, widened ranges + hard dedup against train set
"""

import argparse
import json
import random
from pathlib import Path

import sympy as sp
from sympy import symbols, sympify, diff, integrate, solve, simplify, expand, Eq

x, y = symbols("x y")


def gen_arithmetic():
    ops = ["+", "-", "*"]
    a, b, c = [random.randint(-20, 20) for _ in range(3)]
    op1, op2 = random.choice(ops), random.choice(ops)
    expr_str = f"({a} {op1} {b}) {op2} {c}"
    expr = sympify(expr_str)
    result = sp.nsimplify(expr)
    steps = [
        f"Evaluate the parenthesized term first: {a} {op1} {b} = {sp.sympify(f'{a}{op1}{b}')}",
        f"Substitute back: ({sp.sympify(f'{a}{op1}{b}')}) {op2} {c} = {result}",
    ]
    prompt = f"Compute the value of {expr_str}."
    completion = "\n".join(steps) + f"\nFinal answer: {result}"
    return prompt, completion, str(result)


def gen_linear_equation():
    a = random.choice([i for i in range(-30, 31) if i != 0])
    b = random.randint(-50, 50)
    c = random.randint(-50, 50)
    eq = Eq(a * x + b, c)
    sol = solve(eq, x)[0]
    steps = [
        f"Start with the equation: {a}*x + {b} = {c}",
        f"Subtract {b} from both sides: {a}*x = {c - b}",
        f"Divide both sides by {a}: x = {sp.nsimplify(sp.Rational(c - b, a))}",
    ]
    prompt = f"Solve for x: {a}*x + {b} = {c}"
    completion = "\n".join(steps) + f"\nFinal answer: x = {sol}"
    return prompt, completion, f"x = {sol}"


def gen_quadratic_equation():
    a = random.choice([1, 1, 2, -1, 3, -2])
    b = random.randint(-30, 30)
    c = random.randint(-30, 30)
    expr = a * x**2 + b * x + c
    roots = solve(Eq(expr, 0), x)
    disc = b**2 - 4 * a * c
    steps = [
        f"Equation: {a}*x^2 + ({b})*x + ({c}) = 0",
        f"Discriminant: b^2 - 4ac = {b}^2 - 4*{a}*{c} = {disc}",
        "Roots via quadratic formula: x = (-b +/- sqrt(disc)) / (2a)",
    ]
    prompt = f"Solve for x: {a}*x^2 + ({b})*x + ({c}) = 0"
    completion = "\n".join(steps) + f"\nFinal answer: x = {roots}"
    return prompt, completion, f"x = {roots}"


def gen_derivative():
    terms = []
    for _ in range(random.randint(2, 4)):
        coeff = random.randint(-25, 25) or 1
        power = random.randint(0, 5)
        terms.append(coeff * x**power)
    expr = expand(sum(terms))
    deriv = simplify(diff(expr, x))
    steps = [
        f"f(x) = {expr}",
        "Differentiate term by term using the power rule d/dx[x^n] = n*x^(n-1):",
        f"f'(x) = {deriv}",
    ]
    prompt = f"Find the derivative of f(x) = {expr}."
    completion = "\n".join(steps) + f"\nFinal answer: f'(x) = {deriv}"
    return prompt, completion, f"f'(x) = {deriv}"


def gen_integral():
    terms = []
    for _ in range(random.randint(2, 4)):
        coeff = random.randint(-25, 25) or 1
        power = random.randint(0, 4)
        terms.append(coeff * x**power)
    expr = expand(sum(terms))
    antideriv = integrate(expr, x)
    check = simplify(diff(antideriv, x) - expr)
    assert check == 0, "integration self-check failed"
    steps = [
        f"f(x) = {expr}",
        "Integrate term by term using the power rule Int[x^n dx] = x^(n+1)/(n+1):",
        f"F(x) = {antideriv} + C",
        f"Verify by differentiating F(x): d/dx[{antideriv}] = {expr} (matches f(x))",
    ]
    prompt = f"Find the indefinite integral of f(x) = {expr}."
    completion = "\n".join(steps) + f"\nFinal answer: F(x) = {antideriv} + C"
    return prompt, completion, f"F(x) = {antideriv} + C"


def gen_linear_system():
    a1, b1, c1 = [random.randint(-9, 9) or 1 for _ in range(3)]
    a2, b2, c2 = [random.randint(-9, 9) or 1 for _ in range(3)]
    sol = solve([Eq(a1 * x + b1 * y, c1), Eq(a2 * x + b2 * y, c2)], [x, y])
    steps = [
        f"Equation 1: {a1}*x + {b1}*y = {c1}",
        f"Equation 2: {a2}*x + {b2}*y = {c2}",
        "Solve the system using elimination/substitution.",
        f"Result: {sol}",
    ]
    prompt = f"Solve the system:\n{a1}*x + {b1}*y = {c1}\n{a2}*x + {b2}*y = {c2}"
    completion = "\n".join(steps) + f"\nFinal answer: {sol}"
    return prompt, completion, str(sol)


GENERATORS = {
    "arithmetic": gen_arithmetic,
    "linear_equation": gen_linear_equation,
    "quadratic_equation": gen_quadratic_equation,
    "derivative": gen_derivative,
    "integral": gen_integral,
    "linear_system": gen_linear_system,
}


def generate(domain: str, n: int, exclude_prompts: set = None):
    from verify import verify_example
    exclude_prompts = exclude_prompts or set()
    fn = GENERATORS[domain]
    examples = []
    seen_this_run = set()
    attempts = 0
    while len(examples) < n and attempts < n * 20:
        attempts += 1
        try:
            prompt, completion, answer = fn()
        except Exception:
            continue
        if prompt in exclude_prompts or prompt in seen_this_run:
            continue
        ex = {"prompt": prompt, "completion": completion, "final_answer": answer, "domain": domain}
        if verify_example(ex):
            examples.append(ex)
            seen_this_run.add(prompt)
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=list(GENERATORS) + ["all"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-file", default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    domains = list(GENERATORS) if args.domain == "all" else [args.domain]

    exclude_prompts = set()
    if args.exclude_file:
        with open(args.exclude_file) as f:
            for line in f:
                exclude_prompts.add(json.loads(line)["prompt"])
        print(f"Loaded {len(exclude_prompts)} prompts to exclude from {args.exclude_file}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with out_path.open("w") as f:
        for d in domains:
            examples = generate(d, args.n, exclude_prompts=exclude_prompts)
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
            total += len(examples)
            print(f"{d}: {len(examples)} verified examples")

    print(f"Wrote {total} total examples to {out_path}")


if __name__ == "__main__":
    main()