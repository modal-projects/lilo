import itertools
import math

import pytest

from codegolf.reward import advantages, tailrl_advantages


def test_released_code_optimization_example_and_unsorting():
    # Upstream's worked example [0, 1, 3] -> [-3, -1.5, 4.5], permuted.
    assert tailrl_advantages([3.0, 0.0, 1.0]) == pytest.approx([4.5, -3.0, -1.5])


@pytest.mark.parametrize("n", [1, 2, 8, 16])
def test_binary_rewards_reduce_to_maxrl(n):
    for successes in range(n + 1):
        rewards = [1.0] * successes + [0.0] * (n - successes)
        expected = (
            [n / successes - 1.0] * successes + [-1.0] * (n - successes)
            if successes
            else [0.0] * n
        )
        assert tailrl_advantages(rewards) == pytest.approx(expected)


@pytest.mark.parametrize("rewards", [[], [-0.08], [-0.08] * 8, [1.15] * 8])
def test_degenerate_groups_have_no_signal(rewards):
    assert tailrl_advantages(rewards) == [0.0] * len(rewards)


def test_ties_permutation_and_signed_reward_range():
    rewards = [-0.08, 0.0, 1.02, 1.02, 1.12]
    expected = tailrl_advantages(rewards)
    assert expected[2] == expected[3]
    assert math.fsum(expected) == pytest.approx(0.0, abs=1e-14)
    for order in itertools.permutations(range(len(rewards))):
        actual = tailrl_advantages([rewards[i] for i in order])
        assert actual == pytest.approx([expected[i] for i in order])
    assert tailrl_advantages([r + 10 for r in rewards]) == pytest.approx(expected)
    assert tailrl_advantages([3 * r for r in rewards]) == pytest.approx(
        [3 * a for a in expected]
    )


def test_dispatch_does_not_standardize_tailrl():
    rewards = [0.0, 0.0, 1e-8]
    assert advantages(rewards, 100, estimator="tailrl") == pytest.approx(
        [-1e-8, -1e-8, 2e-8], abs=1e-16
    )
    assert advantages([0, 0, 1], estimator="grpo") == advantages([0, 0, 1])
    with pytest.raises(ValueError, match="Unknown advantage estimator"):
        advantages(rewards, estimator="typo")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_rewards_fail_before_training(bad):
    with pytest.raises(ValueError, match="finite"):
        tailrl_advantages([0.0, bad, 1.0])


def test_expected_gradient_matches_harmonic_best_of_k_objective():
    # Independently enumerate a tiny categorical policy. Centered N-sample
    # advantages, averaged over samples, target sum_{k=1}^{N-1} Best-of-k / k.
    # This checks the N scaling and the same-group baseline, not just the sort.
    n = 4
    rewards = [-0.08, 1.0, 1.12]
    logits = [-0.1, 0.2, -1.0]

    def probabilities(values):
        exp = [math.exp(v) for v in values]
        return [v / sum(exp) for v in exp]

    def objective(values):
        p = probabilities(values)
        return math.fsum(
            math.prod(p[i] for i in draw) * max(rewards[i] for i in draw) / k
            for k in range(1, n)
            for draw in itertools.product(range(len(p)), repeat=k)
        )

    p = probabilities(logits)
    gradient = [0.0] * len(p)
    for draw in itertools.product(range(len(p)), repeat=n):
        probability = math.prod(p[i] for i in draw)
        adv = tailrl_advantages([rewards[i] for i in draw])
        for parameter in range(len(p)):
            gradient[parameter] += probability * math.fsum(
                a * ((i == parameter) - p[parameter]) / n
                for i, a in zip(draw, adv, strict=True)
            )
    for parameter in range(len(p)):
        upper, lower = logits.copy(), logits.copy()
        upper[parameter] += 1e-5
        lower[parameter] -= 1e-5
        numerical = (objective(upper) - objective(lower)) / 2e-5
        assert gradient[parameter] == pytest.approx(numerical, abs=1e-10)
