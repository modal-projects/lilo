import pytest

from codegolf.reward import advantages, datum, extract_code, score


def test_correctness_dominates_length():
    assert score(False, "x") == 0
    assert score(True, "x" * 100000) > score(False, "")
    assert score(True, "print(1)") > score(True, "print(1)\n# longer")
    assert score(True, "") == 0


def test_utf8_and_reasoning():
    assert score(True, "é") == score(True, "xx")
    assert (
        extract_code("<think>reasoning</think>```python\nprint(1)\n```") == "print(1)"
    )


def test_group_normalization():
    assert advantages([0, 0, 0]) == [0, 0, 0]
    assert advantages([0, 1]) == [-1, 1]
    assert abs(sum(advantages([0, 1, 1.2]))) < 1e-12


def test_datum_alignment_and_mask():
    d = datum([10, 11, 12], [13, 14], [-0.2, -0.3], 2.0, 0.5)
    assert d.model_input.to_ints() == [10, 11, 12, 13]
    assert d.loss_fn_inputs["target_tokens"].data == [11, 12, 13, 14]
    assert d.loss_fn_inputs["weights"].data == [0, 0, 0.5, 0.5]
    assert d.loss_fn_inputs["advantages"].data == [0, 0, 2, 2]
    with pytest.raises(ValueError):
        datum([1], [2, 3], [0], 1)


def test_golf_reward_strength_and_output_penalty():
    import dataclasses

    from codegolf.config import Config
    from codegolf.reward import row_score

    config = dataclasses.asdict(Config())

    def reward(passed, size, tokens):
        return row_score(
            {"passed": passed, "code": "x" * size, "tokens": [1] * tokens}, config
        )

    # Halving realistic source length now gives a meaningful bounded signal.
    assert 0.03 < reward(True, 900, 4000) - reward(True, 1800, 4000) < 0.04
    assert reward(True, 1800, 4000) - reward(True, 1800, 12000) == pytest.approx(
        0.0390625
    )
    assert reward(True, 100000, 16384) >= 0.92
    assert reward(False, 1, 0) == 0
    assert reward(False, 1, 16384) == -0.08
    assert reward(False, 1, 32768) == -0.08
