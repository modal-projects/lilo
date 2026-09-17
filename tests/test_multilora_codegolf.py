"""The LoRA runner must preserve codegolf's signed, weighted PPO objective."""

import ast
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples/codeforces-codegolf"))
from codegolf.reward import datum


def translation():
    # Load the pure helper without constructing any Modal deployments/images.
    tree = ast.parse((ROOT / "scripts/multilora_codegolf.py").read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "weighted_ppo_datum"
    )
    scope = {"math": math, "datum": datum}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), "<translation>", "exec"),
        scope,
    )
    return scope["weighted_ppo_datum"]


@pytest.mark.parametrize("advantage", [-2.0, -0.1, 0.0, 0.1, 2.0])
@pytest.mark.parametrize("weight", [0.0, 0.03, 1.0, 7.0])
def test_matches_fft_weighted_ppo_loss_and_gradient(advantage, weight):
    fn = translation()
    prompt, tokens = [1, 2, 3], [4, 5, 6, 7, 8]
    behavior = np.array([-2.0, -3.0, -1.5, -2.5, -4.0])
    out = fn(prompt, tokens, behavior.tolist(), advantage, weight)
    assert out.model_input.to_ints() == prompt + tokens[:-1]
    assert out.loss_fn_inputs["target_tokens"].data == prompt[1:] + tokens
    assert "weights" not in out.loss_fn_inputs
    a = np.array(out.loss_fn_inputs["advantages"].data)
    assert np.array_equal(a[:2], [0, 0])
    a = a[2:]
    # Exercise both clipping boundaries, interior ratios, and both advantage signs.
    current = behavior + np.log([0.5, 0.9, 1.0, 1.1, 1.5])

    def reference(lp):
        ratio = np.exp(lp - behavior)
        return -np.sum(
            weight * np.minimum(ratio * advantage, np.clip(ratio, 0.8, 1.2) * advantage)
        )

    def miles(lp):
        ratio = np.exp(lp - behavior)
        return -np.sum(np.minimum(ratio * a, np.clip(ratio, 0.8, 1.2) * a))

    assert miles(current) == pytest.approx(reference(current))
    epsilon = 1e-5
    for i in range(5):
        direction = np.zeros(5)
        direction[i] = epsilon
        left = (reference(current + direction) - reference(current - direction)) / (
            2 * epsilon
        )
        right = (miles(current + direction) - miles(current - direction)) / (
            2 * epsilon
        )
        assert right == pytest.approx(
            left, rel=2e-7, abs=1e-8
        )  # SDK stores advantages as float32


def test_recipe_is_full_context_four_slots_and_same_model():
    tree = ast.parse(
        (
            ROOT / "src/lilo/providers/modal/definitions/qwen3_5_9b_miles_lora_64k.py"
        ).read_text()
    )
    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
    }
    assert constants["MODEL_NAME"] == "Qwen/Qwen3.5-9B"
    assert constants["MAX_CONTEXT_LENGTH"] == 65536
    assert constants["GPU_TYPE"] == "H200"
    assert constants["GPUS"] == constants["TENSOR_MODEL_PARALLEL_SIZE"] == 8
    assert constants["MAX_LORA_SLOTS"] == 4
