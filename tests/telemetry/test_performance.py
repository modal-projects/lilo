import asyncio

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.trace import StatusCode

from lilo.telemetry.performance import Stages


def test_stuck_wait_visible_before_completion_and_cancellation_cleans_up():
    reader = InMemoryMetricReader()
    metrics = MeterProvider(metric_readers=[reader])
    exporter = InMemorySpanExporter()
    tracer = TracerProvider()
    tracer.add_span_processor(SimpleSpanProcessor(exporter))
    stages = Stages("inference", metrics=metrics, tracer=tracer.get_tracer("test"))

    async def run():
        entered = asyncio.Event()

        async def wait():
            with stages.track(
                "adapter_lock_wait", attributes={"lilo.request_id": "private-id"}
            ):
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(wait())
        await entered.wait()
        assert not exporter.get_finished_spans()
        points = {
            m.name: m.data.data_points
            for r in reader.get_metrics_data().resource_metrics
            for s in r.scope_metrics
            for m in s.metrics
        }
        assert points["lilo.stage.inflight"][0].value == 1
        assert points["lilo.stage.oldest_age"][0].value > 0
        assert "private-id" not in str(points)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stages.observe("inflight", None)[0].value == 0
        assert stages.observe("oldest_age", None)[0].value == 0

    asyncio.run(run())
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert not span.events
    metrics.shutdown()
    tracer.shutdown()


def test_exception_text_is_not_exported_and_export_failure_does_not_mask_it():
    reader = InMemoryMetricReader()
    metrics = MeterProvider(metric_readers=[reader])
    exporter = InMemorySpanExporter()
    tracer = TracerProvider()
    tracer.add_span_processor(SimpleSpanProcessor(exporter))
    stages = Stages("inference", metrics=metrics, tracer=tracer.get_tracer("test"))

    class BrokenHistogram:
        def record(self, *args):
            raise ValueError("export failure")

    stages.duration = BrokenHistogram()
    with pytest.raises(RuntimeError, match="PRIVATE PROMPT"):
        with stages.track("adapter_load"):
            raise RuntimeError("PRIVATE PROMPT")
    (span,) = exporter.get_finished_spans()
    assert span.attributes["error.type"] == "RuntimeError"
    assert "PRIVATE" not in str(span.attributes)
    assert not span.events
    assert stages.observe("inflight", None)[0].value == 0
    metrics.shutdown()
    tracer.shutdown()


def test_multiple_components_keep_live_callbacks_and_dropped_spans_work():
    reader = InMemoryMetricReader()
    metrics = MeterProvider(metric_readers=[reader])
    tracer = TracerProvider(sampler=ALWAYS_OFF)
    a = Stages("inference", metrics=metrics, tracer=tracer.get_tracer("test"))
    b = Stages("sampling", metrics=metrics, tracer=tracer.get_tracer("test"))
    with a.track("lock"), b.track("http"):
        points = [
            p
            for r in reader.get_metrics_data().resource_metrics
            for s in r.scope_metrics
            for m in s.metrics
            if m.name == "lilo.stage.inflight"
            for p in m.data.data_points
        ]
        assert {p.attributes["lilo.component"] for p in points} == {
            "inference",
            "sampling",
        }
        assert all(p.value == 1 for p in points)
    metrics.shutdown()
    tracer.shutdown()
