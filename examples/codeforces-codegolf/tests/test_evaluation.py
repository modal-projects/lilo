import itertools
import math
import statistics

import pytest

from codegolf.evaluation import best_of_k, pass_at_k, sampling_metrics


@pytest.mark.parametrize("rewards", [[-0.08, 0.0, 1.0, 1.0, 1.15], [-0.1] * 5])
def test_best_of_k_matches_all_subsets(rewards):
    for k in range(1, len(rewards) + 1):
        expected = statistics.fmean(max(s) for s in itertools.combinations(rewards, k))
        assert best_of_k(rewards, k) == pytest.approx(expected)


def test_pass_at_k_matches_all_subsets_and_binary_best_of_k():
    for successes in range(6):
        rewards = [1.0] * successes + [0.0] * (5 - successes)
        for k in range(1, 6):
            expected = statistics.fmean(
                any(s) for s in itertools.combinations(rewards, k)
            )
            assert pass_at_k(5, successes, k) == pytest.approx(expected)
            assert best_of_k(rewards, k) == pytest.approx(expected)


def test_large_sampling_budget_does_not_overflow():
    rewards = [i / 4096 for i in range(4096)]
    curve = [best_of_k(rewards, k) for k in [1, 2, 16, 1024, 4096]]
    assert all(math.isfinite(value) for value in curve)
    assert curve == sorted(curve)
    assert curve[0] == statistics.fmean(rewards)
    assert curve[-1] == max(rewards)


@pytest.mark.parametrize("k", [0, -1, 4])
def test_no_extrapolation_beyond_sample_count(k):
    with pytest.raises(ValueError):
        best_of_k([0.0, 1.0, 2.0], k)
    with pytest.raises(ValueError):
        pass_at_k(3, 1, k)


def test_invalid_pools_are_rejected():
    with pytest.raises(ValueError):
        best_of_k([], 1)
    with pytest.raises(ValueError, match="finite"):
        best_of_k([float("nan"), 1], 1)
    with pytest.raises(ValueError):
        pass_at_k(3, 4, 1)
    with pytest.raises(ValueError):
        pass_at_k(3, -1, 1)
    with pytest.raises(ValueError, match="nonempty"):
        sampling_metrics([])
    with pytest.raises(ValueError, match="nonempty"):
        sampling_metrics([{"rows": []}])
    with pytest.raises(ValueError, match="same sample count"):
        sampling_metrics([{"rows": [{}]}, {"rows": [{}, {}]}])


def test_metrics_average_over_problems_and_use_verdict_for_success():
    # A zero-penalty failure has reward zero, but still fails the judge.
    records = [
        {"rows": [{"reward": r, "passed": r > 0} for r in [0, 0, 1.0]]},
        {"rows": [{"reward": r, "passed": r > 0} for r in [1.0, 1.1, 1.2]]},
    ]
    metrics = sampling_metrics(records)
    assert metrics["eval_samples"] == 3
    assert metrics["eval_problems"] == 2
    assert metrics["pass_at_k"] == pytest.approx({"1": 2 / 3, "2": 5 / 6, "3": 1})
    assert metrics["best_of_k"]["1"] == pytest.approx((1 / 3 + 1.1) / 2)
    assert metrics["best_of_k"]["3"] == pytest.approx(1.1)
