import json

from lilo.telemetry.critical_path import (
    ALL_MODELS,
    MAX_SERIES,
    CriticalPath,
    flatten,
)


def test_series_aggregate_per_model_and_across_models() -> None:
    timings = CriticalPath()
    timings.record("forward_backward.execute", 2.0, model_id="a", batch=2)
    timings.record("forward_backward.execute", 4.0, model_id="b", batch=1)

    per_model = timings.snapshot(model_id="a")["phases"]
    assert per_model == {
        "forward_backward.execute": {
            "count": 1,
            "total_s": 2.0,
            "mean_s": 2.0,
            "max_s": 2.0,
            "mean_batch": 2.0,
        }
    }
    combined = timings.snapshot()["phases"]["forward_backward.execute"]
    assert combined["count"] == 2
    assert combined["total_s"] == 6.0
    assert combined["max_s"] == 4.0


def test_aggregate_record_counts_once() -> None:
    timings = CriticalPath()
    timings.record("forward_backward.execute", 2.0, model_id=ALL_MODELS)
    assert timings.snapshot()["phases"]["forward_backward.execute"]["count"] == 1


def test_forget_model_evicts_series_and_pending() -> None:
    timings = CriticalPath()
    timings.register_model("a", {})
    timings.record("optim_step.execute", 1.0, model_id="a")
    timings.record("optim_step.execute", 1.0, model_id="b")
    timings.forget_model("a")
    assert timings.snapshot(model_id="a")["phases"] == {}
    assert timings.snapshot(model_id="b")["phases"]["optim_step.execute"]["count"] == 1
    assert timings.snapshot()["phases"]["optim_step.execute"]["count"] == 2
    assert timings.pending == {}


def test_queue_wait_and_execute_stay_separate() -> None:
    timings = CriticalPath()
    timings.record("forward_backward.queue_wait", 9.0, model_id="a")
    timings.record("forward_backward.execute", 1.0, model_id="a")

    phases = timings.snapshot(model_id="a")["phases"]
    assert phases["forward_backward.queue_wait"]["total_s"] == 9.0
    assert phases["forward_backward.execute"]["total_s"] == 1.0


def test_reset_clears_only_the_requested_model() -> None:
    timings = CriticalPath()
    timings.record("optim_step.execute", 1.0, model_id="a")
    timings.record("optim_step.execute", 1.0, model_id="b")

    timings.snapshot(model_id="a", reset=True)
    assert timings.snapshot(model_id="a")["phases"] == {}
    assert timings.snapshot(model_id="b")["phases"]["optim_step.execute"]["count"] == 1


def test_negative_durations_are_ignored() -> None:
    timings = CriticalPath()
    timings.record("optim_step.execute", -1.0, model_id="a")
    assert timings.snapshot(model_id="a")["phases"] == {}


def test_gauges_can_be_written_once() -> None:
    timings = CriticalPath()
    timings.gauge("trainer.backend_ready_s", 12.0, once=True)
    timings.gauge("trainer.backend_ready_s", 90.0, once=True)
    timings.gauge("trainer.first_model_ready_s", 30.0)
    timings.gauge("trainer.first_model_ready_s", 31.0)

    assert timings.snapshot()["gauges"] == {
        "trainer.backend_ready_s": 12.0,
        "trainer.first_model_ready_s": 31.0,
    }


def test_series_are_bounded() -> None:
    timings = CriticalPath()
    for index in range(MAX_SERIES * 2):
        timings.record("optim_step.execute", 1.0, model_id=f"model-{index}")
    assert len(timings.series) <= MAX_SERIES


def test_emit_prints_one_json_line(capsys) -> None:
    timings = CriticalPath()
    timings.record("optim_step.execute", 1.5, model_id="a")
    timings.report()

    line = capsys.readouterr().out.strip()
    payload = json.loads(line)
    assert payload["event"] == "lilo_critical_path"
    assert payload["phases"]["optim_step.execute"]["mean_s"] == 1.5


def test_flatten_produces_scalar_metrics() -> None:
    timings = CriticalPath()
    timings.record("forward_backward.execute", 2.0, model_id="a", batch=4)
    timings.gauge("trainer.backend_ready_s", 12.0)

    metrics = flatten(timings.snapshot(model_id="a"))
    assert metrics["lilo/forward_backward.execute.mean_s"] == 2.0
    assert metrics["lilo/forward_backward.execute.mean_batch"] == 4.0
    assert metrics["lilo/trainer.backend_ready_s"] == 12.0
    assert all(isinstance(value, float) for value in metrics.values())
