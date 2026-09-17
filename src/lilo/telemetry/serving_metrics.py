"""Forward local serving counters/histograms and sampled GPU activity over OTLP."""

from __future__ import annotations

import logging
import math
import os
import socket
import subprocess
import threading
import time
from collections import defaultdict

import httpx
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics.export import (
    AggregationTemporality,
    Gauge,
    Histogram,
    HistogramDataPoint,
    Metric,
    MetricsData,
    NumberDataPoint,
    ResourceMetrics,
    ScopeMetrics,
    Sum,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from prometheus_client.parser import text_string_to_metric_families

from .performance import metrics_enabled

log = logging.getLogger(__name__)
# Drop unrecognized labels rather than exporting prompts, adapter versions or IDs.
# A family with unknown labels is skipped, not collapsed into duplicate series.
LABELS = {
    "model_name",
    "engine_type",
    "tp_rank",
    "dp_rank",
    "pp_rank",
    "moe_ep_rank",
    "is_streaming",
    "cache_source",
    "speculative_algorithm",
}
FAMILIES = {
    "lora_pool_slots_used",
    "lora_pool_slots_total",
    "lora_pool_utilization",
    "num_retracted_requests",
    "num_aborted_requests",
    "inter_token_latency_seconds",
    "kv_available_tokens",
    "kv_evictable_tokens",
    "kv_used_tokens",
    "mamba_usage",
    "full_token_usage",
    "num_running_reqs",
    "num_waiting_reqs",
    "num_queue_reqs",
    "num_used_tokens",
    "token_usage",
    "cache_hit_rate",
    "gen_throughput",
    "num_retracted_reqs",
    "prompt_tokens",
    "generation_tokens",
    "cached_tokens",
    "num_requests",
    "time_to_first_token_seconds",
    "time_per_output_token_seconds",
    "e2e_request_latency_seconds",
    "queue_time_seconds",
}


class PrometheusDeltas:
    """Keep exact bucket counts. The first observation establishes a baseline."""

    def __init__(self, attrs):
        self.attrs = attrs
        self.previous = {}

    def convert(self, text, now):
        output = []
        current = {}
        for family in text_string_to_metric_families(text):
            if not family.name.startswith("sglang:") or family.name[7:] not in FAMILIES:
                continue
            groups = defaultdict(dict)
            for sample in family.samples:
                labels = dict(sample.labels)
                bound = labels.pop("le", None)
                if set(labels) - LABELS or not math.isfinite(sample.value):
                    continue
                key = tuple(sorted(labels.items()))
                groups[key][(sample.name, bound)] = sample.value
            points = []
            for labels, samples in groups.items():
                attrs = {**self.attrs, **dict(labels)}
                key = (family.name, labels)
                if family.type == "gauge":
                    value = samples.get((family.name, None))
                    if value is not None:
                        points.append(NumberDataPoint(attrs, now, now, value))
                elif family.type == "counter":
                    value = samples.get((family.name + "_total", None))
                    if value is None:
                        continue
                    current[key] = (now, value)
                    previous = self.previous.get(key)
                    if previous is not None and value >= previous[1]:
                        points.append(
                            NumberDataPoint(
                                attrs, previous[0], now, value - previous[1]
                            )
                        )
                elif family.type == "histogram":
                    buckets = sorted(
                        (float(bound), int(value))
                        for (name, bound), value in samples.items()
                        if name == family.name + "_bucket" and bound is not None
                    )
                    count = samples.get((family.name + "_count", None))
                    total = samples.get((family.name + "_sum", None))
                    if (
                        not buckets
                        or buckets[-1][0] != math.inf
                        or count is None
                        or total is None
                    ):
                        continue
                    bounds = tuple(b for b, _ in buckets[:-1])
                    cumulative = tuple(c for _, c in buckets)
                    counts = tuple(
                        c - (cumulative[i - 1] if i else 0)
                        for i, c in enumerate(cumulative)
                    )
                    if any(c < 0 for c in counts) or sum(counts) != count:
                        continue
                    current[key] = (now, bounds, counts, int(count), total)
                    previous = self.previous.get(key)
                    if previous is None or bounds != previous[1]:
                        continue
                    delta = tuple(
                        a - b for a, b in zip(counts, previous[2], strict=True)
                    )
                    if any(c < 0 for c in delta) or total < previous[4]:
                        continue  # Process restart/reset: use this scrape as the baseline.
                    points.append(
                        HistogramDataPoint(
                            attrs,
                            previous[0],
                            now,
                            int(count) - previous[3],
                            total - previous[4],
                            delta,
                            bounds,
                            None,
                            None,
                        )
                    )
            if points:
                if family.type == "gauge":
                    data = Gauge(points)
                elif family.type == "counter":
                    data = Sum(points, AggregationTemporality.DELTA, True)
                else:
                    data = Histogram(points, AggregationTemporality.DELTA)
                unit = "s" if family.name.endswith("_seconds") else "1"
                output.append(
                    Metric(
                        family.name.replace(":", "."), family.documentation, unit, data
                    )
                )
        self.previous = (
            current  # Missing/disappearing series do not accumulate forever.
        )
        return output


class ServingMetrics:
    """One bounded background sampler per trainer container or inference sidecar."""

    def __init__(self, component, upstream_url=None):
        self.upstream_url = upstream_url
        self.attrs = {
            "lilo.component": component,
            "lilo.instance_id": os.getenv("MODAL_TASK_ID") or socket.gethostname(),
        }
        self.resource = Resource.create(
            {"service.name": os.getenv("OTEL_SERVICE_NAME", "lilo")}
        )
        self.converter = PrometheusDeltas(self.attrs)
        self.stop = threading.Event()
        self.thread = None
        self.exporter = None
        self.last_gpu = None

    def start(self):
        if not metrics_enabled():
            return
        protocol = os.getenv(
            "OTEL_EXPORTER_OTLP_METRICS_PROTOCOL",
            os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf"),
        )
        if protocol != "http/protobuf":
            return
        try:
            self.exporter = OTLPMetricExporter(timeout=2)
            self.thread = threading.Thread(
                target=self.run, name="lilo-serving-metrics", daemon=True
            )
            self.thread.start()
        except Exception:
            log.warning("Serving metric initialization failed")

    def close(self):
        self.stop.set()
        # Never block an async request loop on an exporter. Lifespan calls via to_thread.
        if self.thread is not None:
            self.thread.join(timeout=7)
        if self.exporter is not None and (
            self.thread is None or not self.thread.is_alive()
        ):
            self.exporter.shutdown()

    def gpu_metrics(self, now):
        raw = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        ).stdout
        points = defaultdict(list)
        devices = 0
        fields = (
            ("activity", "%", 1),
            ("memory_activity", "%", 1),
            ("memory_used", "By", 1024**2),
            ("memory_total", "By", 1024**2),
            ("power", "W", 1),
        )
        for row in raw.splitlines():
            values = [v.strip() for v in row.split(",")]
            if len(values) != 6:
                continue
            devices += 1
            attrs = {**self.attrs, "gpu.index": values[0]}
            for (name, unit, scale), value in zip(fields, values[1:], strict=True):
                try:
                    value = float(value) * scale
                except ValueError:
                    continue
                if math.isfinite(value):
                    points[(name, unit)].append(NumberDataPoint(attrs, now, now, value))
        output = [
            Metric(
                "lilo.gpu." + name,
                "Sampled by nvidia-smi; not FLOP utilization",
                unit,
                Gauge(rows),
            )
            for (name, unit), rows in points.items()
        ]
        if devices and self.last_gpu is not None:
            elapsed = max(0, (now - self.last_gpu) / 1e9)
            output.append(
                Metric(
                    "lilo.gpu.capacity",
                    "Visible GPU seconds in this container",
                    "s",
                    Sum(
                        [
                            NumberDataPoint(
                                self.attrs, self.last_gpu, now, devices * elapsed
                            )
                        ],
                        AggregationTemporality.DELTA,
                        True,
                    ),
                )
            )
        self.last_gpu = now if devices else None
        return output

    def run(self):
        with httpx.Client(timeout=2, trust_env=False) as client:
            while not self.stop.is_set():
                now = time.time_ns()
                metrics = []
                if self.upstream_url is not None:
                    try:
                        response = client.get(
                            self.upstream_url.rstrip("/") + "/metrics"
                        )
                        response.raise_for_status()
                        metrics.extend(self.converter.convert(response.text, now))
                    except Exception:
                        log.warning("SGLang metric scrape failed")
                try:
                    metrics.extend(self.gpu_metrics(now))
                except Exception:
                    self.last_gpu = None
                    log.warning("GPU metric sampling failed")
                if metrics:
                    try:
                        self.exporter.export(
                            MetricsData(
                                [
                                    ResourceMetrics(
                                        self.resource,
                                        [
                                            ScopeMetrics(
                                                InstrumentationScope("lilo.serving"),
                                                metrics,
                                                "",
                                            )
                                        ],
                                        "",
                                    )
                                ]
                            )
                        )
                    except Exception:
                        log.warning("Serving metric export failed")
                self.stop.wait(5)
