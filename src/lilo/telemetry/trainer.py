"""Explicit trainer command tracing and sampled operation state; no payload capture."""

from __future__ import annotations

import functools
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from opentelemetry.context import Context, get_value, set_value
from opentelemetry.propagate import extract, inject
from opentelemetry.trace import (
    Link,
    NonRecordingSpan,
    Span,
    SpanKind,
    StatusCode,
    set_span_in_context,
)

from lilo.telemetry.otlp import _attributes, provider

from . import backend
from .metadata import common_tags, experiment_tags

incoming = ContextVar("lilo_command_context", default=None)
accepted_root = ContextVar("lilo_accepted_root", default=None)
log = logging.getLogger(__name__)
OPERATIONS = (
    "idle",
    "accept",
    "unload",
    "forward",
    "forward_backward",
    "optim_step",
    "save_weights",
    "load_weights",
    "save_weights_for_sampler",
    "skip",
)


def best_effort(method):
    @functools.wraps(method)
    def call(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except Exception:  # noqa: BLE001 - telemetry cannot fail training
            log.warning("Trainer telemetry operation failed: %s", method.__name__)
            return None

    return call


def workload(payload):
    """Count input examples and text tokens; never infer loss tokens from input length."""
    data = getattr(payload, "data", None)
    if data is None:
        return {}
    counts = {"lilo.example_count": len(data)}
    tokens = 0
    for example in data:
        for chunk in example.model_input.chunks:
            values = getattr(chunk, "tokens", None)
            if values is None:
                return (
                    counts  # Image/other inputs do not have a known token count here.
                )
            tokens += len(values)
    counts["lilo.input_tokens"] = tokens
    return counts


def headers():
    value = incoming.get()
    if value is None:
        return {}
    context, started = value
    carrier = {"x-lilo-command-start": str(started)}
    inject(carrier, context=context)
    return carrier


class CommandMiddleware:
    """Trace submissions, excluding noisy future polling. Forward context explicitly."""

    def __init__(self, app, *, receiver=False):
        self.app, self.receiver = app, receiver

    async def __call__(self, scope, receive, send):
        name = scope.get("path", "").rsplit("/", 1)[-1]
        if scope["type"] != "http" or name not in (
            *OPERATIONS[1:],
            "accept_model",
            "unload_model",
            "create_model",
        ):
            return await self.app(scope, receive, send)
        p = provider()
        if p is None:
            return await self.app(scope, receive, send)
        now = time.time()
        carrier = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope.get("headers", [])}
        if self.receiver:
            try:
                started = float(carrier.get("x-lilo-command-start", now))
                if not 0 < started <= now:
                    started = now
            except ValueError:
                started = now
            token = incoming.set((extract(carrier), started))
            accepted_token = accepted_root.set(None)

            async def respond(message):
                if (
                    message["type"] == "http.response.start"
                    and accepted_root.get() is not None
                ):
                    propagated = {}
                    inject(propagated, context=accepted_root.get())
                    tags = get_value("lilo.command.tags", accepted_root.get()) or {}
                    message = {
                        **message,
                        "headers": [
                            *message.get("headers", []),
                            (b"x-lilo-command-tags", json.dumps(tags).encode()),
                            (
                                b"x-lilo-command-traceparent",
                                propagated["traceparent"].encode(),
                            ),
                        ],
                    }
                await send(message)

            try:
                return await self.app(scope, receive, respond)
            finally:
                incoming.reset(token)
                accepted_root.reset(accepted_token)
        token = incoming.set((Context(), now))
        accepted_token = accepted_root.set(None)
        status = 500
        error_type = None

        async def respond(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            return await self.app(scope, receive, respond)
        except BaseException as exc:
            error_type = type(exc).__name__
            raise
        finally:
            canonical = accepted_root.get()
            incoming.reset(token)
            accepted_root.reset(accepted_token)
            # The trainer returns the canonical identity on duplicate submissions too.
            # Emit the HTTP span afterwards so retries join the existing command.
            span = p.get_tracer("lilo.control").start_span(
                "lilo.control.submit" if canonical else "lilo.control." + name,
                context=canonical or Context(),
                kind=SpanKind.SERVER,
                start_time=int(now * 1e9),
                attributes={
                    "lilo.operation": name,
                    "lilo.component": "control",
                    "http.response.status_code": status,
                },
            )
            if canonical is not None:
                _attributes(span, get_value("lilo.command.tags", canonical) or {})
            if error_type:
                span.set_attribute("error.type", error_type)
            span.set_status(
                StatusCode.ERROR if status >= 400 or error_type else StatusCode.OK
            )
            span.end()


@dataclass
class CommandTrace:
    span: Span
    attributes: dict[str, Any]


class TrainerTelemetry:
    def __init__(
        self, instance_id, definition_id, boot_id, *, metric_reader=None, scoped=False
    ):
        self.attrs = {
            "lilo.trainer_instance_id": instance_id,
            "lilo.definition_id": definition_id,
            "lilo.boot_id": boot_id,
            "lilo.component": "trainer",
        }
        self.metric_tags = {}
        self.model_tags = {}
        self.commands: dict[str, CommandTrace] = {}
        self.identities = OrderedDict()
        self.lock = threading.Lock()
        self.activity = {"execution": "idle", "checkpoint": "idle", "sampler": "idle"}
        self.closed = False
        self.meter_provider = None
        if metric_reader is not None or (
            os.getenv("OTEL_SDK_DISABLED", "").lower() != "true"
            and (
                os.getenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT")
                or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
            )
        ):
            try:
                from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                    OTLPMetricExporter,
                )
                from opentelemetry.sdk.metrics import MeterProvider
                from opentelemetry.sdk.metrics.export import (
                    PeriodicExportingMetricReader,
                )
                from opentelemetry.sdk.resources import Resource

                protocol = os.getenv(
                    "OTEL_EXPORTER_OTLP_METRICS_PROTOCOL",
                    os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf"),
                )
                if protocol != "http/protobuf":
                    raise ValueError("Trainer metrics require http/protobuf")
                reader = metric_reader or PeriodicExportingMetricReader(
                    OTLPMetricExporter(),
                    export_interval_millis=5000,
                    export_timeout_millis=4000,
                )
                resource = Resource.create(
                    {"service.name": os.getenv("OTEL_SERVICE_NAME", "lilo")}
                )
                if scoped:
                    # Direct OTLP intake may not promote custom resource fields
                    # to metric tags. This deployment has one experiment owner.
                    self.metric_tags = experiment_tags(
                        {"run_id": resource.attributes.get("lilo.run_id")}
                    )
                self.meter_provider = MeterProvider(
                    resource=resource,
                    metric_readers=[reader],
                )
                self.meter_provider.get_meter("lilo.trainer").create_observable_gauge(
                    "lilo.trainer.state",
                    callbacks=[self.observe],
                    description="Sampled trainer operation activity, one-hot per lane",
                    unit="1",
                )
            except Exception:  # noqa: BLE001 - optional telemetry
                log.warning("Trainer state metric initialization failed")

    def observe(self, options):
        from opentelemetry.metrics import Observation

        with self.lock:
            if self.closed:
                return []
            activity = dict(self.activity)
        return [
            Observation(
                int(current == operation),
                {
                    **self.attrs,
                    **self.metric_tags,
                    "lilo.lane": lane,
                    "lilo.operation": operation,
                },
            )
            for lane, current in activity.items()
            for operation in (
                OPERATIONS
                if lane == "execution"
                else (
                    "idle",
                    "save_weights"
                    if lane == "checkpoint"
                    else "save_weights_for_sampler",
                )
            )
        ]

    def set_activity(self, lane, operation):
        with self.lock:
            self.activity[lane] = operation

    @best_effort
    def register_model(self, model_id, spec):
        self.model_tags[model_id] = experiment_tags(
            spec.get("user_metadata") if isinstance(spec, dict) else None
        )

    @best_effort
    def forget_model(self, model_id):
        self.model_tags.pop(model_id, None)
        for request_id in list(self.identities):
            if request_id.startswith(model_id + ":"):
                self.identities.pop(request_id, None)

    @best_effort
    def begin(self, operation):
        p = provider()
        if p is None:
            return
        _context, started = incoming.get() or (Context(), time.time())
        attributes = {
            **self.attrs,
            "lilo.component": "command",
            "lilo.model_id": operation.model_id,
            "lilo.request_id": operation.request_id,
            "lilo.seq_id": operation.seq_id,
            "lilo.operation": operation.kind.value,
            **workload(getattr(operation, "payload", None)),
            **self.model_tags.get(operation.model_id, {}),
        }
        span = p.get_tracer("lilo.trainer").start_span(
            "lilo.command." + operation.kind.value,
            context=Context(),
            start_time=int(started * 1e9),
            attributes=attributes,
        )
        self.commands[operation.request_id] = CommandTrace(span, attributes)
        canonical = set_span_in_context(NonRecordingSpan(span.get_span_context()))
        canonical = set_value(
            "lilo.command.tags",
            {
                k: v
                for k, v in attributes.items()
                if k
                in (
                    "lilo.run_id",
                    "lilo.run_attempt_id",
                    "lilo.model_id",
                    "lilo.request_id",
                )
            },
            canonical,
        )
        self.identities[operation.request_id] = canonical
        accepted_root.set(canonical)
        while len(self.identities) > 2048:
            self.identities.popitem(last=False)

    @best_effort
    def reuse(self, request_id):
        context = self.identities.get(request_id)
        if context is not None:
            accepted_root.set(context)

    @best_effort
    def finish(self, operation, state):
        entry = self.commands.pop(operation.request_id, None)
        if entry:
            span = entry.span
            ok = state.status.value == "complete"
            self._span(
                "result_ready",
                "execution",
                time.time(),
                time.time(),
                [entry],
                {
                    "lilo.model_id": operation.model_id,
                    "lilo.request_id": operation.request_id,
                },
                ok=ok,
            )
            span.set_status(StatusCode.OK if ok else StatusCode.ERROR)
            span.end()

    def _span(
        self,
        name,
        lane,
        t0,
        t1,
        parents,
        attrs,
        ok=True,
        independent=False,
        prefix="lilo.trainer.",
        execution_context=None,
        parent_context=None,
    ):
        p = provider()
        if p is None:
            return
        # A batch shared by several commands has links to all of them; no arbitrary parent.
        context = (
            set_span_in_context(parents[0].span)
            if len(parents) == 1 and not independent
            else Context()
        )
        links = (
            [Link(s.span.get_span_context()) for s in parents]
            if len(parents) > 1 or independent
            else []
        )
        if parent_context is not None:
            context = set_span_in_context(NonRecordingSpan(parent_context))
        if execution_context is not None:
            links.append(Link(execution_context))
        span = p.get_tracer("lilo.trainer").start_span(
            prefix + name.replace(":", "."),
            context=context,
            links=links,
            start_time=int(t0 * 1e9),
        )
        _attributes(
            span,
            {
                **self.attrs,
                **common_tags([p.attributes for p in parents]),
                "lilo.lane": lane,
                **attrs,
            },
        )
        span.set_status(StatusCode.OK if ok else StatusCode.ERROR)
        span.end(end_time=int(t1 * 1e9))
        return span.get_span_context()

    @best_effort
    def span(self, model, name, lane, t0, t1=None, **attrs):
        evidence = backend.received.get() or {}
        backend.received.set(None)
        models = (model,) if isinstance(model, str) else tuple(model)
        seq_ids = attrs.get("seq_ids", [])
        parents = [
            self.commands[f"{m}:{s}"]
            for m, s in zip(models, seq_ids)
            if f"{m}:{s}" in self.commands
        ]
        safe = {
            "lilo.operation": name.split(":")[-1],
            "lilo.command_count": attrs.get("n", len(seq_ids)),
        }
        counts = backend.numeric_attributes(evidence.get("attributes"))
        model_counts = evidence.get("models", {})
        if not isinstance(model_counts, dict):
            model_counts = {}
        for parent in parents:
            per_command = backend.numeric_attributes(
                model_counts.get(parent.attributes["lilo.model_id"])
            )
            if "lilo.loss_tokens" in per_command:
                value = per_command["lilo.loss_tokens"]
                parent.attributes["lilo.loss_tokens"] = value
                parent.span.set_attribute("lilo.loss_tokens", value)
            if len(parents) == 1 and "lilo.checkpoint_bytes" in counts:
                parent.attributes["lilo.checkpoint_bytes"] = counts[
                    "lilo.checkpoint_bytes"
                ]
                parent.span.set_attribute(
                    "lilo.checkpoint_bytes", counts["lilo.checkpoint_bytes"]
                )
        if parents and all("lilo.loss_tokens" in p.attributes for p in parents):
            counts["lilo.loss_tokens"] = sum(
                p.attributes["lilo.loss_tokens"] for p in parents
            )
        safe.update(counts)
        safe.update(common_tags([p.attributes for p in parents]))
        if not parents:
            safe.update(common_tags([self.model_tags.get(m, {}) for m in models]))
        if parents and all("lilo.example_count" in p.attributes for p in parents):
            safe["lilo.example_count"] = sum(
                p.attributes["lilo.example_count"] for p in parents
            )
        if parents and all("lilo.input_tokens" in p.attributes for p in parents):
            safe["lilo.input_tokens"] = sum(
                p.attributes["lilo.input_tokens"] for p in parents
            )
        if len(models) == 1:
            safe["lilo.model_id"] = models[0]
            if len(seq_ids) == 1:
                safe["lilo.request_id"] = f"{models[0]}:{seq_ids[0]}"
        lane = "execution" if lane == "gpu" else lane
        ended = t1 if t1 is not None else time.time()
        execution_context = self._span(
            name,
            lane,
            t0,
            ended,
            parents,
            safe,
            ok=attrs.get("ok", True),
            independent=True,
        )
        phases = evidence.get("phases", [])
        if execution_context is not None and isinstance(phases, list):
            for phase in phases[: backend.MAX_PHASES]:
                if (
                    not isinstance(phase, dict)
                    or phase.get("name") not in backend.PHASES
                ):
                    continue
                start, end = phase.get("start_ns"), phase.get("end_ns")
                if type(start) is not int or type(end) is not int:
                    continue
                if not int(t0 * 1e9) <= start <= end <= int(ended * 1e9):
                    continue
                self._span(
                    phase["name"],
                    lane,
                    start / 1e9,
                    end / 1e9,
                    [],
                    {**safe, "lilo.component": "backend", "lilo.rank": 0},
                    ok=phase.get("ok") is True,
                    prefix="lilo.backend.",
                    parent_context=execution_context,
                )
        # Command-local participation makes waiting visible as gaps. Physical
        # batches remain independent traces with aggregate workload counts.
        if name.startswith("wait_persistence:") or execution_context is None:
            return
        phase = name.split(":", 1)[0] if ":" in name else "execute"
        for parent in parents:
            self._span(
                phase,
                lane,
                t0,
                ended,
                [parent],
                parent.attributes,
                ok=attrs.get("ok", True),
                prefix="lilo.command.",
                execution_context=execution_context,
            )

    @best_effort
    def state(self, model, state, **detail):
        operation = state.rsplit(":", 1)[-1]
        if operation in OPERATIONS:
            self.set_activity("execution", operation)

    @best_effort
    def close(self):
        with self.lock:
            self.closed = True
        for command in self.commands.values():
            span = command.span
            span.set_attribute("lilo.incomplete", True)
            span.set_status(StatusCode.ERROR)
            span.end()
        self.commands.clear()
        self.identities.clear()
        self.model_tags.clear()
        if self.meter_provider:
            self.meter_provider.shutdown(timeout_millis=5000)
        from lilo.telemetry.otlp import flush

        flush()
