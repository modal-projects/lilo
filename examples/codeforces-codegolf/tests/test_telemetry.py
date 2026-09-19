import asyncio

from codegolf.store import Store
from codegolf.telemetry import RunTelemetry


def test_export_failure_does_not_change_durable_writes(tmp_path):
    class FailingTelemetry(RunTelemetry):
        @property
        def run_id(self):
            raise RuntimeError("export failed")

    t = FailingTelemetry.__new__(FailingTelemetry)
    s = Store(tmp_path, observer=t.observe)
    asyncio.run(s.write("checkpoint.json", {"step": 50, "path": "saved"}))
    assert s.read("checkpoint.json")["step"] == 50


def test_step_metrics_use_run_identity_and_omit_sensitive_fields(monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    t = RunTelemetry("golf")
    t.attempt_id = "restore-2"
    seen = []

    class Gauge:
        def __init__(self, name):
            self.name = name

        def set(self, value, attrs):
            seen.append((self.name, value, attrs))

    class Meter:
        def create_gauge(self, name):
            return Gauge(name)

    t.meter = Meter()
    t.observe(
        "metrics/0001.json",
        {
            "step": 1,
            "reward": 1.2,
            "prompt": "SECRET",
            "text": "SECRET",
            "model_id": "m",
        },
    )
    assert all(
        attrs
        == {"lilo.run_id": "golf", "lilo.run_attempt_id": "restore-2", "phase": "train"}
        for _, _, attrs in seen
    )
    assert any(name == "codegolf.reward" and value == 1.2 for name, value, _ in seen)
    assert "SECRET" not in str(seen)
