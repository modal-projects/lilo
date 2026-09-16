from __future__ import annotations

import math
import re


def extract_code(text: str, *, require_thinking_end: bool = False) -> str:
    # An unfinished thinking section is not a submitted solution.
    if require_thinking_end and "</think>" not in text:
        return ""
    # Qwen may emit reasoning even when disabled. Never measure reasoning as code.
    text = text.rsplit("</think>", 1)[-1].strip()
    blocks = re.findall(r"```(?:python3?|py)?\s*\n(.*?)```", text, re.DOTALL)
    return (blocks[-1] if blocks else text).strip()


def score(
    passed: bool,
    code: str,
    scale: float = 256,
    bonus: float = 0.1,
    *,
    output_tokens: int = 0,
    token_penalty: float = 0.0,
    token_scale: int = 16384,
) -> float:
    if scale <= 0 or token_scale <= 0 or output_tokens < 0:
        raise ValueError("Invalid reward scale or token count")
    penalty = token_penalty * min(output_tokens / token_scale, 1.0)
    if not passed or not code.strip():
        return -penalty
    return 1.0 + bonus * math.exp(-len(code.encode("utf-8")) / scale) - penalty


def row_score(row, config):
    """Score every sampled output token, including prose outside extracted code."""
    return score(
        row["passed"],
        row["code"],
        scale=config.get("reward_scale", 256),
        bonus=config["reward_bonus"],
        output_tokens=len(row["tokens"]),
        token_penalty=config.get("output_token_penalty", 0.0),
        token_scale=config.get("output_token_scale", 16384),
    )


def tailrl_advantages(rewards: list[float]) -> list[float]:
    """Mean-centered TailRL, with the released code-optimization N scaling.

    For ascending rewards, w_i = sum_{j<=i} (r_j-r_{j-1})/(N-j+1),
    and A_i = N * (w_i - mean(w)). See arXiv:2609.02987v2, Sec. 4
    and Appendix D, and Zanette-Labs/TailRL code_opt/advantages.py at
    commit 5682c6ac03387355e017ce966693266bb148fa10.

    The N factor matches a loss averaged over sequences. There is no std
    normalization, clipping, or rank transform. Centering cancels the first
    reward gap, so start at the minimum for numerical stability and to handle
    signed codegolf rewards directly. Equal rewards receive equal advantages.
    """
    if not all(math.isfinite(r) for r in rewards):
        raise ValueError("TailRL rewards must be finite")
    n = len(rewards)
    if n <= 1:
        return [0.0] * n
    order = sorted(range(n), key=rewards.__getitem__)
    previous = rewards[order[0]]
    cumulative = 0.0
    weights = [0.0] * n
    for rank, index in enumerate(order):
        reward = rewards[index]
        cumulative += (reward - previous) / (n - rank)
        weights[index] = n * cumulative
        previous = reward
    mean = math.fsum(weights) / n
    return [weight - mean for weight in weights]


def advantages(
    rewards: list[float], std_floor: float = 0.5, *, estimator: str = "grpo"
) -> list[float]:
    if estimator == "tailrl":
        return tailrl_advantages(rewards)
    if estimator != "grpo":
        raise ValueError(f"Unknown advantage estimator: {estimator}")
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
    return [(r - mean) / max(std, std_floor) for r in rewards]


def datum(prompt, tokens, logprobs, advantage, sequence_weight=1.0):
    from tinker import types

    if not prompt or not tokens or len(tokens) != len(logprobs):
        raise ValueError("Missing prompt/completion or misaligned sampling logprobs")
    prefix = len(prompt) - 1
    return types.Datum(
        model_input=types.ModelInput.from_ints(prompt + tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": prompt[1:] + tokens,
            "logprobs": [0.0] * prefix + logprobs,
            "advantages": [0.0] * prefix + [advantage] * len(tokens),
            "weights": [0.0] * prefix + [sequence_weight] * len(tokens),
        },
    )
