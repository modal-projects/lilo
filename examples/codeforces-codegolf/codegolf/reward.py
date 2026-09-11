from __future__ import annotations

import math
import re


def extract_code(text: str) -> str:
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


def advantages(rewards: list[float], std_floor: float = 0.5) -> list[float]:
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
