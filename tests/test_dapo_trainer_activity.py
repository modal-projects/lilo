"""Reject incomplete timing evidence and verify window accounting."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "trainer_activity",
    Path(__file__).parents[1] / "scripts/plot_dapo_trainer_activity.py",
)
activity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(activity)


def event(second, phase, name="forward_backward"):
    return (
        f"[2026-09-18 00:00:{second:06.3f} main] cell.py:227 - "
        f"ft op=execute phase={phase} cell=trainer-actor-0 fn={name} ok=true\n"
    )


def test_embedded_timestamps_clipping_and_delivery_order():
    start = activity.timestamp("2026-09-18 00:00:02.000")
    # Delivery order is reversed; clipping removes work before/after the window.
    log = event(7, "end") + event(1, "start")
    assert activity.operation_intervals(log, start, start + 3) == [
        {"operation": "forward_backward", "start": 0, "end": 3}
    ]


@pytest.mark.parametrize(
    "log",
    [
        event(1, "start"),
        event(2, "end"),
        event(1, "start") + event(2, "start") + event(3, "end"),
        event(1, "start") + event(3, "end").replace("ok=true", "ok=false"),
        event(1, "start")
        + event(2, "start", "optim_step")
        + event(3, "end")
        + event(4, "end", "optim_step"),
    ],
)
def test_reject_missing_failed_duplicate_or_overlapping_calls(log):
    start = activity.timestamp("2026-09-18 00:00:00.000")
    with pytest.raises(ValueError):
        activity.operation_intervals(log, start, start + 10)


def test_occupancy_counts_only_overlap_with_window():
    intervals = [{"start": 1, "end": 4}, {"start": 8, "end": 12}]
    assert activity.window_occupancy(intervals, 2, 10) == 0.5
