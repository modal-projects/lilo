"""Opt-in sampling traces. No metrics, automatic instrumentation, or payload capture."""

from __future__ import annotations

import functools
import logging
import math
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar

from opentelemetry.context import Context
from opentelemetry.trace import SpanKind, StatusCode, set_span_in_context

logger = logging.getLogger(__name__)
_sample = ContextVar("tune_otlp_sample", default=None)


@functools.cache
def provider():
    """Use a private provider so embedding Tune never reconfigures application tracing."""
    if os.getenv("OTEL_SDK_DISABLED", "").lower() == "true" or not (
        os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    ):
        return None
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        protocol = os.getenv(
            "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL",
            os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf"),
        )
        if protocol != "http/protobuf":
            raise ValueError("Tune sampling traces require http/protobuf")
        result = TracerProvider(
            resource=Resource.create(
                {"service.name": os.getenv("OTEL_SERVICE_NAME", "tune")}
            )
        )
        result.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        return result
    except Exception:  # noqa: BLE001 - telemetry must not prevent sampling
        # Do not log config/headers or export errors into the sampling response.
        logger.warning("Tune OTLP initialization failed; tracing disabled")
        return None


def flush(timeout_millis=5000):
    """For graceful shutdown and diagnostics; never flush in the sampling hot path."""
    p = provider()
    return p.force_flush(timeout_millis) if p is not None else True


def _attributes(span, attrs):
    for key, value in attrs.items():
        if isinstance(value, (str, bool, int)) or (
            isinstance(value, float) and math.isfinite(value)
        ):
            span.set_attribute(key, value)


@contextmanager
def sample_trace(task, stats):
    p = provider()
    if p is None:
        yield
        return
    now = time.time()
    accepted = task.get("accepted_at")
    start = (
        accepted if isinstance(accepted, (int, float)) and 0 < accepted <= now else now
    )
    span = p.get_tracer("tune.sampling").start_span(
        "tune.sample",
        context=Context(),
        kind=SpanKind.SERVER,
        start_time=int(start * 1e9),
    )
    from .metadata import METADATA_KEYS

    tags = {
        k: v
        for k, v in (task.get("telemetry_tags") or {}).items()
        if k in METADATA_KEYS.values() and isinstance(v, str) and 0 < len(v) <= 256
    }
    _attributes(span, tags)
    count = int((task.get("payload") or {}).get("num_samples", 1))
    _attributes(
        span,
        {
            "tune.request_id": task.get("request_id"),
            "tune.model_id": task.get("model_id")
            or "base:" + str(task.get("engine_definition_id")),
            "tune.base_model": task.get("base_model"),
            "tune.num_samples": count,
            "tune.version_requested": task.get("publish_version"),
            "tune.latest": bool(task.get("latest")),
            "tune.start_boundary": "accepted" if start == accepted else "worker",
        },
    )
    state = {
        "span": span,
        "tracer": p.get_tracer("tune.sampling"),
        "tags": tags,
        "attempts": 0,
        "retries": 0,
    }
    token = _sample.set(state)
    try:
        yield
    except BaseException as exc:
        span.set_status(StatusCode.ERROR)
        span.set_attribute("error.type", type(exc).__name__)
        raise
    else:
        span.set_status(StatusCode.OK)
    finally:
        _sample.reset(token)
        _attributes(
            span,
            {
                "tune.input_tokens": stats.get("prompt_tokens"),
                "tune.output_tokens": stats.get("generated_tokens"),
                "tune.attempt_count": state["attempts"],
                "tune.retry_count": state["retries"],
            },
        )
        # Cache/version evidence belongs to each sequence, not an arbitrary first one.
        if count == 1:
            _attributes(
                span,
                {
                    "tune.version_served_start": stats.get("version_served_start"),
                    "tune.version_served_end": stats.get("version_served_end"),
                },
            )
        span.end()


def record_attempt(event):
    """Bridge only the approved terminal attempt event, using an explicit allowlist."""
    state = _sample.get()
    if (
        state is None
        or event.get("ev") != "span"
        or event.get("name") != "sample_attempt"
    ):
        return
    attrs = event.get("attrs", {})
    state["attempts"] += 1
    state["retries"] += int(attrs.get("attempt_number", 1) > 1)
    span = state["tracer"].start_span(
        "tune.sample.attempt",
        context=set_span_in_context(state["span"]),
        kind=SpanKind.CLIENT,
        start_time=int(event["t0"] * 1e9),
    )
    raw = attrs.get("backend_timing") or {}
    _attributes(
        span,
        {
            **state["tags"],
            "tune.request_id": attrs.get("request_id"),
            "tune.attempt_id": attrs.get("attempt_id"),
            "tune.sequence_index": attrs.get("sequence_index"),
            "tune.attempt_number": attrs.get("attempt_number"),
            "tune.input_tokens": attrs.get("prompt_tokens"),
            "tune.output_tokens": attrs.get("generated_tokens"),
            "http.response.status_code": attrs.get("http_status"),
            "error.type": attrs.get("error_type"),
            "sglang.request_id": attrs.get("sglang_request_id"),
            "sglang.queue_s": attrs.get("scheduler_queue_s"),
            "sglang.prefill_s": attrs.get("prefill_elapsed_s"),
            # This is explicitly NOT pure decode or GPU time.
            "sglang.post_prefill_to_finish_s": attrs.get("decode_to_finish_s"),
            "sglang.cached_tokens": raw.get("cached_tokens"),
            "sglang.prompt_tokens": raw.get("prompt_tokens"),
            "sglang.completion_tokens": raw.get("completion_tokens"),
            "tune.version_served_start": raw.get("weight_version_start"),
            "tune.version_served_end": raw.get("weight_version_end"),
        },
    )
    span.set_status(StatusCode.OK if attrs.get("ok") else StatusCode.ERROR)
    span.end(end_time=int(event["t1"] * 1e9))
