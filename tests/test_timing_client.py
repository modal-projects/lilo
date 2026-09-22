import json
import sys
import types

import httpx

from lilo.timing import critical_path_metrics, log_critical_path

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
    )

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["step"] == 7
    assert payload["metrics"] == metrics


def test_logging_uses_the_active_wandb_run(monkeypatch, capsys) -> None:
    logged = {}

    run = types.SimpleNamespace(
        log=lambda metrics, step=None: logged.update(metrics=metrics, step=step)
    )
    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(run=run))

    log_critical_path(
        "model-a",
        step=3,
        base_url="https://lilo.invalid",
        api_key="tml-test",
        transport=_transport(lambda request: httpx.Response(200, json=SNAPSHOT)),
    )

    assert logged["step"] == 3
    assert logged["metrics"]["lilo/forward_backward.execute.count"] == 2.0
    assert capsys.readouterr().out == ""


def test_logging_never_raises_into_the_training_step() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "unavailable"})

    assert (
        log_critical_path(
            "model-a",
            base_url="https://lilo.invalid",
            api_key="tml-test",
            transport=_transport(handler),
        )
        == {}
    )
