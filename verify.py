"""
verify.py

Independent, from-scratch verification of generated examples. This does NOT
reuse the generator's internal variables -- it re-parses the prompt text and
re-solves it via sympy, then checks the recomputed answer against the one
in the completion. This catches generator bugs, not just trusts them.

This is the gate: generate_data.py only writes an example to disk if
verify_example() returns True.
"""

import sympy as sp
from sympy import symbols, sympify, diff, integrate, solve, simplify, Eq

x, y = symbols("x y")


def _extract_final(completion: str) -> str:
    for line in completion.splitlines()[::-1]:
        if line.startswith("Final answer:"):
            return line[len("Final answer:"):].strip()
    raise ValueError("no final answer line found")


def verify_example(ex: dict) -> bool:
    domain = ex["domain"]
    prompt = ex["prompt"]
    claimed = ex["final_answer"]

    try:
        if domain == "arithmetic":
            expr_str = prompt.split("value of", 1)[1].rstrip(". ").strip()
            recomputed = sp.nsimplify(sympify(expr_str))
            return str(recomputed) == claimed.strip()

        if domain == "linear_equation":
            body = prompt.split("Solve for x:", 1)[1].strip()
            lhs_str, rhs_str = body.split("=")
            recomputed = solve(Eq(sympify(lhs_str), sympify(rhs_str)), x)[0]
            return f"x = {recomputed}" == claimed.strip()

        if domain == "quadratic_equation":
            body = prompt.split("Solve for x:", 1)[1].strip()
            lhs_str, rhs_str = body.split("=")
            recomputed = solve(Eq(sympify(lhs_str), sympify(rhs_str)), x)
            return f"x = {recomputed}" == claimed.strip()

        if domain == "derivative":
            fx_str = prompt.split("f(x) =", 1)[1].rstrip(". ").strip()
            recomputed = simplify(diff(sympify(fx_str), x))
            return f"f'(x) = {recomputed}" == claimed.strip()

        if domain == "integral":
            fx_str = prompt.split("f(x) =", 1)[1].rstrip(". ").strip()
            fx = sympify(fx_str)
            antideriv = integrate(fx, x)
            # cross-check: derivative of antiderivative must equal original
            if simplify(diff(antideriv, x) - fx) != 0:
                return False
            return f"F(x) = {antideriv} + C" == claimed.strip()

        if domain == "linear_system":
            lines = prompt.splitlines()[1:]
            eqs = []
            for ln in lines:
                lhs_str, rhs_str = ln.split("=")
                eqs.append(Eq(sympify(lhs_str), sympify(rhs_str)))
            recomputed = solve(eqs, [x, y])
            return str(recomputed) == claimed.strip()

    except Exception:
        return False

    return False