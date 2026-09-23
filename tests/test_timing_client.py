import json
import sys
import threading
import types

import httpx

from lilo.timing import (
    critical_path_metrics,
    flush_critical_path,
    log_critical_path,
)

SNAPSHOT = {
    "model_id": "model-a",
    "uptime_s": 300.0,
    "phases": {
        "forward_backward.execute": {
            "count": 2,
            "total_s": 4.0,
            "mean_s": 2.0,
            "max_s": 3.0,
            "mean_batch": 1.0,
        }
    },
    "gauges": {"trainer.backend_ready_s": 120.0},
}


def _transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def test_metrics_query_the_owning_trainer() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-api-key")
        return httpx.Response(200, json=SNAPSHOT)

    metrics = critical_path_metrics(
        "model-a",
        base_url="https://lilo.invalid/",
        api_key="tml-test",
        transport=_transport(handler),
    )

    assert seen["url"] == (
        "https://lilo.invalid/api/v1/timing?model_id=model-a&reset=true"
    )
    assert seen["key"] == "tml-test"
    assert metrics["lilo/forward_backward.execute.mean_s"] == 2.0
    assert metrics["lilo/trainer.backend_ready_s"] == 120.0
    assert metrics["lilo/uptime_s"] == 300.0


def test_logging_falls_back_to_stdout_without_wandb(capsys) -> None:
    metrics = log_critical_path(
        "model-a",
        step=7,
        base_url="https://lilo.invalid",
        api_key="tml-test",
        transport=_transport(lambda request: httpx.Response(200, json=SNAPSHOT)),
        background=False,
    )

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["step"] == 7
    assert payload["metrics"] == metrics


class FakeRun:
    def __init__(self) -> None:
        self.logged: list[tuple[dict, int | None, bool | None]] = []
        self.defined: list[tuple[str, dict]] = []

    def log(self, data, step=None, commit=None):
        self.logged.append((data, step, commit))

    def define_metric(self, name, **kwargs):
        self.defined.append((name, kwargs))


def test_logging_uses_the_active_wandb_run(monkeypatch, capsys) -> None:
    run = FakeRun()
    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(run=run))

    log_critical_path(
        "model-a",
        step=3,
        base_url="https://lilo.invalid",
        api_key="tml-test",
        transport=_transport(lambda request: httpx.Response(200, json=SNAPSHOT)),
        background=False,
    )

    [(data, step, commit)] = run.logged
    assert step is None
    assert commit is False
    assert data["lilo/step"] == 3
    assert data["lilo/forward_backward.execute.count"] == 2.0
    assert ("lilo/*", {"step_metric": "lilo/step"}) in run.defined
    assert capsys.readouterr().out == ""


def test_logging_never_raises_into_the_training_step(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "unavailable"})

    assert (
        log_critical_path(
            "model-a",
            base_url="https://lilo.invalid",
            api_key="tml-test",
            transport=_transport(handler),
            background=False,
        )
        == {}
    )

    def explode(*args, **kwargs):
        raise RuntimeError("wandb is unhappy")

    run = types.SimpleNamespace(log=explode, define_metric=explode)
    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(run=run))
    log_critical_path(
        "model-a",
        step=1,
        base_url="https://lilo.invalid",
        api_key="tml-test",
        transport=_transport(lambda request: httpx.Response(200, json=SNAPSHOT)),
        background=False,
    )


def test_background_logging_does_not_block_and_drops_overlapping_calls(
    monkeypatch, capsys
) -> None:
    release = threading.Event()
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        release.wait(timeout=5)
        return httpx.Response(200, json=SNAPSHOT)

    run = FakeRun()
    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(run=run))
    kwargs = {
        "base_url": "https://lilo.invalid",
        "api_key": "tml-test",
        "transport": _transport(handler),
    }

    assert log_critical_path("model-a", step=1, **kwargs) is None
    assert log_critical_path("model-a", step=2, **kwargs) is None
    assert run.logged == []
    release.set()
    assert flush_critical_path(timeout=5)

    assert len(calls) == 1
    [(data, _, commit)] = run.logged
    assert data["lilo/step"] == 1
    assert commit is False
    assert log_critical_path("model-a", step=3, **kwargs) is None
    assert flush_critical_path(timeout=5)
    assert [data["lilo/step"] for data, _, _ in run.logged] == [1, 3]
