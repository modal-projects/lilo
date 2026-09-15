"""Held-out sampling-budget metrics (TailRL paper, Appendix G).

Estimate performance from every size-k subset of an n-rollout pool, without
enumerating subsets or extrapolating past the number of collected rollouts.
"""

from __future__ import annotations

import math
import statistics


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased Pass@k estimate from c successes among n independent samples."""
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError("Require 0 <= c <= n and 1 <= k <= n")
    if n - c < k:
        return 1.0
    return 1.0 - math.prod(1.0 - c / (n - i) for i in range(k))


def best_of_k(rewards: list[float], k: int) -> float:
    """Unbiased expected maximum reward of k samples; supports signed rewards."""
    n = len(rewards)
    if not 1 <= k <= n:
        raise ValueError("Require 1 <= k <= number of rewards")
    if not all(math.isfinite(r) for r in rewards):
        raise ValueError("Evaluation rewards must be finite")
    if k == 1:
        return statistics.fmean(rewards)
    # Descending rank i has weight C(n-i-1, k-1) / C(n, k).
    # Recurrence avoids enormous binomial coefficients at large eval budgets.
    weight = k / n
    terms = []
    for rank, reward in enumerate(sorted(rewards, reverse=True)):
        terms.append(weight * reward)
        remaining = n - rank - 1
        if remaining < k:
            break
        weight *= (remaining - k + 1) / remaining
    return math.fsum(terms)


def sampling_metrics(records) -> dict:
    """Average each metric over problems, using powers of two and the full pool."""
    if not records or any(not record["rows"] for record in records):
        raise ValueError("Evaluation requires nonempty rollout groups")
    sizes = {len(record["rows"]) for record in records}
    if len(sizes) != 1:
        raise ValueError("Evaluation groups must have the same sample count")
    n = sizes.pop()
    budgets = sorted({1 << i for i in range(n.bit_length())} | {n})
    rewards = [[r["reward"] for r in record["rows"]] for record in records]
    successes = [sum(bool(r["passed"]) for r in g["rows"]) for g in records]
    return {
        "eval_problems": len(records),
        "eval_samples": n,
        "pass_at_k": {
            str(k): statistics.fmean(pass_at_k(n, c, k) for c in successes)
            for k in budgets
        },
        "best_of_k": {
            str(k): statistics.fmean(best_of_k(group, k) for group in rewards)
            for k in budgets
        },
    }
