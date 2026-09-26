"""Sampling telemetry normalization and request lookup contract."""

from __future__ import annotations

import logging
import math
import time
import uuid


def event_sink(task, callback):
    model = task.get("model_id") or "base:" + str(
        task.get("engine_definition_id", "unknown")
    )

    def emit(event):
        from .otlp import record_attempt

        try:
            record_attempt(event)
        except Exception:  # noqa: BLE001 - telemetry must not fail sampling
            logging.getLogger(__name__).warning("sample OTLP event failed")
        if callback is not None:
            try:
                callback({"model": model, **event})
            except Exception:
                logging.getLogger(__name__).exception("sample telemetry")

    return emit


class SampleAttempt:
    """Measurement lifecycle is separate from routing, retries and response conversion."""

    def __init__(self, emit, attrs):
        self.emit = emit
        self.id = uuid.uuid4().hex
        self.attrs = {**attrs, "attempt_id": self.id, "ok": False}
        self.started, self.clock = time.time(), time.monotonic()
        self.emit(
            {
                "ev": "lifecycle",
                "name": "sample_attempt.started",
                "t": self.started,
                "attrs": dict(self.attrs),
            }
        )

    def finish(self):
        finished = time.time()
        self.emit(
            {
                "ev": "span",
                "name": "sample_attempt",
                "lane": "inference",
                "t": finished,
                "t0": self.started,
                "t1": finished,
                "attrs": {**self.attrs, "duration_s": time.monotonic() - self.clock},
            }
        )


TIMING_KEYS = (
    "queue_time",
    "forward_entry_time",
    "prefill_finished_time",
    "request_received_ts",
    "api_server_dispatch_finish_ts",
    "request_finished_ts",
    "response_sent_to_client_ts",
    "e2e_latency",
    "decode_throughput",
    "num_retractions",
    "cached_tokens",
    "lilo_admission_retries",
    "completion_tokens",
    "prompt_tokens",
    "weight_version_start",
    "weight_version_end",
)


def timing_metadata(meta: dict) -> dict:
    """Retain backend evidence, without presenting buffered output rate as decode speed."""
    raw = {
        k: meta[k]
        for k in TIMING_KEYS
        if isinstance(meta.get(k), (int, float))
        and not isinstance(meta[k], bool)
        and math.isfinite(meta[k])
    }
    out = {
        "backend_timing": raw,
        "timing_source": "sglang.generate.meta_info",
        "decode_timing_validated": False,
    }
    if isinstance(meta.get("id"), str):
        out["sglang_request_id"] = meta["id"]
    if raw.get("queue_time", -1) >= 0:
        out["scheduler_queue_s"] = raw["queue_time"]
    start, end = raw.get("forward_entry_time"), raw.get("prefill_finished_time")
    if start is not None and end is not None and 0 < start <= end:
        out["prefill_elapsed_s"] = end - start
    finished = raw.get("request_finished_ts")
    if end is not None and finished is not None and 0 < end <= finished:
        # Scheduler prefill completion -> API server request completion. This
        # includes decode plus final delivery/processing, not pure GPU duration.
        out["decode_to_finish_s"] = finished - end
    rate, tokens = raw.get("decode_throughput"), raw.get("completion_tokens")
    if rate is not None and rate > 0 and tokens is not None and tokens > 1:
        # SGLang's pinned req_time_stats.py defines throughput using N-1 tokens.
        out["backend_reported_decode_s"] = (tokens - 1) / rate
    return out
