"""Opt-in stage spans and live age gauges; metric labels never contain request IDs."""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import socket
import threading
import time
from contextlib import contextmanager
from itertools import count

from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.metrics import Observation
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import StatusCode

from . import otlp

log = logging.getLogger(__name__)


def metrics_enabled():
    return os.getenv("OTEL_SDK_DISABLED", "").lower() != "true" and bool(
        os.getenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT")
        or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    )


@functools.cache
def meter_provider():
    if not metrics_enabled():
        return None
    try:
        protocol = os.getenv(
            "OTEL_EXPORTER_OTLP_METRICS_PROTOCOL",
            os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf"),
        )
        if protocol != "http/protobuf":
            raise ValueError("Metrics require http/protobuf")
        return MeterProvider(
            resource=Resource.create(
                {"service.name": os.getenv("OTEL_SERVICE_NAME", "lilo")}
            ),
            metric_readers=[
                PeriodicExportingMetricReader(
                    OTLPMetricExporter(),
                    export_interval_millis=5000,
                    export_timeout_millis=4000,
                )
            ],
        )
    except Exception:
        log.warning("Performance metric initialization failed")
        return None


class Stages:
    def __init__(self, component, *, metrics=None, tracer=None):
        self.attrs = {
            "lilo.component": component,
            "lilo.instance_id": os.getenv("MODAL_TASK_ID") or socket.gethostname(),
            "process.pid": os.getpid(),
        }
        self.tracer = tracer
        self.metrics = metrics if metrics is not None else meter_provider()
        self.lock = threading.Lock()
        self.ids = count()
        self.active = {}
        self.progress = {}
        self.duration = self.events = None
        if self.metrics is not None:
            meter = self.metrics.get_meter("lilo.performance." + component)
            self.duration = meter.create_histogram(
                "lilo.stage.duration",
                unit="s",
                description="Wall time in a host stage, including any nested waits",
            )
            self.events = meter.create_counter("lilo.stage.completed", unit="1")
            for name, unit in (
                ("inflight", "1"),
                ("oldest_age", "s"),
                ("last_progress_age", "s"),
            ):
                meter.create_observable_gauge(
                    f"lilo.stage.{name}",
                    unit=unit,
                    callbacks=[functools.partial(self.observe, name)],
                )

    def wrap(self, stage):
        """Instrument an async handoff without changing its arguments or result."""

        def decorate(function):
            @functools.wraps(function)
            async def call(*args, **kwargs):
                with self.track(stage):
                    return await function(*args, **kwargs)

            return call

        return decorate

    def observe(self, field, options):
        now = time.monotonic()
        with self.lock:
            result = []
            for stage, active in self.active.items():
                if field == "inflight":
                    value = len(active)
                elif field == "oldest_age":
                    value = max(0, now - min(active.values())) if active else 0
                else:
                    value = max(0, now - self.progress[stage])
                result.append(Observation(value, {**self.attrs, "lilo.stage": stage}))
            return result

    @contextmanager
    def track(self, stage, *, attributes=None, context=None):
        provider = otlp.provider()
        tracer = self.tracer or (
            provider.get_tracer("lilo.performance") if provider else None
        )
        if tracer is None and self.metrics is None:
            yield
            return
        started = time.monotonic()
        key = next(self.ids)
        with self.lock:
            self.active.setdefault(stage, {})[key] = started
            self.progress.setdefault(stage, started)
        status = "ok"
        try:
            if tracer is None:
                yield None
            else:
                # Exception messages can contain request data. Record only status/type.
                with tracer.start_as_current_span(
                    f"lilo.{self.attrs['lilo.component']}.{stage}",
                    context=context,
                    attributes={**self.attrs, **(attributes or {})},
                    record_exception=False,
                    set_status_on_exception=False,
                ) as span:
                    try:
                        yield span
                        if span.is_recording():
                            if span.status.status_code == StatusCode.ERROR:
                                status = "error"
                            elif span.status.status_code == StatusCode.UNSET:
                                span.set_status(StatusCode.OK)
                    except BaseException as exc:
                        span.set_attribute("error.type", type(exc).__name__)
                        span.set_status(StatusCode.ERROR)
                        raise
        except BaseException as exc:
            status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "error"
            raise
        finally:
            ended = time.monotonic()
            with self.lock:
                self.active[stage].pop(key, None)
                self.progress[stage] = ended
            if self.duration is not None:
                attrs = {**self.attrs, "lilo.stage": stage, "lilo.status": status}
                try:
                    self.duration.record(ended - started, attrs)
                    self.events.add(1, attrs)
                except Exception:
                    log.warning("Performance metric recording failed")


@functools.cache
def stages(component):
    return Stages(component)
