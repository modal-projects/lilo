"""Best-effort experiment telemetry. Durable run files remain authoritative."""

import logging
import math
import os
import time

log = logging.getLogger(__name__)


class RunTelemetry:
    def __init__(self, run_id):
        self.run_id = run_id
        self.attempt_id = None
        self.traces = self.metrics = self.tracer = self.meter = None
        self.gauges = {}
        try:
            if os.getenv("OTEL_SDK_DISABLED", "").lower() == "true":
                return
            from opentelemetry.sdk.resources import Resource

            resource = Resource.create(
                {
                    "service.name": "codegolf",
                    "deployment.environment.name": os.getenv("MODAL_ENVIRONMENT", ""),
                }
            )
            if os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.getenv(
                "OTEL_EXPORTER_OTLP_ENDPOINT"
            ):
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter,
                )
                from opentelemetry.sdk.trace import TracerProvider
                from opentelemetry.sdk.trace.export import BatchSpanProcessor

                self.traces = TracerProvider(resource=resource)
                self.traces.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
                self.tracer = self.traces.get_tracer("codegolf")
            if os.getenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT") or os.getenv(
                "OTEL_EXPORTER_OTLP_ENDPOINT"
            ):
                from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                    OTLPMetricExporter,
                )
                from opentelemetry.sdk.metrics import MeterProvider
                from opentelemetry.sdk.metrics.export import (
                    PeriodicExportingMetricReader,
                )

                self.metrics = MeterProvider(
                    resource=resource,
                    metric_readers=[
                        PeriodicExportingMetricReader(
                            OTLPMetricExporter(), export_interval_millis=5000
                        )
                    ],
                )
                self.meter = self.metrics.get_meter("codegolf")
        except Exception:
            log.warning("Experiment telemetry initialization failed")

    def observe(self, name, value):
        try:
            self._observe(name, value)
        except Exception:
            log.warning("Experiment telemetry export failed")

    def close(self):
        for provider in (self.metrics, self.traces):
            if provider:
                try:
                    provider.shutdown()
                except Exception:
                    log.warning("Experiment telemetry shutdown failed")

    def _observe(self, name, value):
        identity = {"lilo.run_id": self.run_id}
        if self.attempt_id:
            identity["lilo.run_attempt_id"] = self.attempt_id
        if name.startswith(("metrics/", "eval/")):
            phase = "eval" if name.startswith("eval/") else "train"
            values = {
                k: value.get(k)
                for k in (
                    "step",
                    "reward",
                    "pass_rate",
                    "passing_bytes",
                    "completion_tokens",
                    "sampled_entropy",
                    "truncated",
                    "samples",
                    "seconds",
                )
            }
            values.update(value.get("pipeline", {}))
            values["grad_norm"] = value.get("optimizer", {}).get("grad_norm:mean")
            if self.meter:
                for key, number in values.items():
                    if isinstance(number, (int, float)) and math.isfinite(number):
                        if key not in self.gauges:
                            self.gauges[key] = self.meter.create_gauge(
                                "codegolf." + key
                            )
                        self.gauges[key].set(number, {**identity, "phase": phase})
            return
        if not name.startswith(("events/", "attempts/")) or not self.tracer:
            return
        kind = value.get("kind", "attempt")
        attrs = dict(identity)
        for key in ("step", "model_id", "recovery", "seconds"):
            v = value.get(key)
            if isinstance(v, (str, int, float)):
                attrs["codegolf." + key] = v
        # Point events describe durable receipts, not invented operation durations.
        stamp = int(time.time() * 1e9)
        span = self.tracer.start_span(
            "codegolf." + kind, attributes=attrs, start_time=stamp
        )
        span.end(end_time=stamp)
