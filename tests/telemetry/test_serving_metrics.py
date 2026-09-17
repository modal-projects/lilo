import math
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

from opentelemetry.exporter.otlp.proto.common.metrics_encoder import encode_metrics
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
    ExportMetricsServiceRequest,
)
from opentelemetry.sdk.metrics.export import (
    MetricExportResult,
    MetricsData,
    ResourceMetrics,
    ScopeMetrics,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.util.instrumentation import InstrumentationScope

from lilo.telemetry.serving_metrics import PrometheusDeltas, ServingMetrics


def sample(counter=100, low=2, high=3, total=1.2, extra=""):
    # Labels match the pinned SGLang scheduler and tokenizer collectors.
    return f"""# TYPE sglang:num_queue_reqs gauge
sglang:num_queue_reqs{{model_name="qwen",engine_type="unified",moe_ep_rank="0",tp_rank="0"}} 4
# TYPE sglang:generation_tokens_total counter
sglang:generation_tokens_total{{model_name="qwen",is_streaming="false"}} {counter}
# TYPE sglang:time_to_first_token_seconds histogram
sglang:time_to_first_token_seconds_bucket{{model_name="qwen",le="0.5"}} {low}
sglang:time_to_first_token_seconds_bucket{{model_name="qwen",le="+Inf"}} {high}
sglang:time_to_first_token_seconds_count{{model_name="qwen"}} {high}
sglang:time_to_first_token_seconds_sum{{model_name="qwen"}} {total}
{extra}
"""


def test_native_histogram_and_counter_deltas_serialize_over_otlp():
    converter = PrometheusDeltas({"lilo.instance_id": "replica-1"})
    baseline = converter.convert(sample(), 10)
    assert [m.name for m in baseline] == ["sglang.num_queue_reqs"]
    result = converter.convert(sample(160, 3, 5, 2.7), 20)
    by_name = {m.name: m for m in result}
    counter = by_name["sglang.generation_tokens"].data.data_points[0]
    assert counter.value == 60
    point = by_name["sglang.time_to_first_token_seconds"].data.data_points[0]
    assert point.count == 2
    assert point.bucket_counts == (1, 1)
    assert point.explicit_bounds == (0.5,)
    assert math.isclose(point.sum, 1.5)
    assert (point.start_time_unix_nano, point.time_unix_nano) == (10, 20)
    encoded = encode_metrics(
        MetricsData(
            [
                ResourceMetrics(
                    Resource.create({}),
                    [ScopeMetrics(InstrumentationScope("test"), result, "")],
                    "",
                )
            ]
        )
    )
    assert encoded.SerializeToString()
    # A reset must not produce negative counts; the following scrape resumes deltas.
    assert len(converter.convert(sample(4, 0, 1, 0.2), 30)) == 1
    after = converter.convert(sample(9, 1, 2, 0.8), 40)
    assert (
        next(m for m in after if m.name == "sglang.generation_tokens")
        .data.data_points[0]
        .value
        == 5
    )


def test_unknown_labels_and_removed_series_do_not_leak_or_accumulate():
    converter = PrometheusDeltas({})
    text = sample(extra='sglang:num_queue_reqs{request_id="PRIVATE"} 100')
    rows = converter.convert(text, 1)
    assert "PRIVATE" not in str(rows)
    assert len(rows[0].data.data_points) == 1
    converter.convert("", 2)
    assert converter.previous == {}
    assert len(converter.convert(text, 3)) == 1


def test_gpu_capacity_and_activity_have_distinct_units(monkeypatch):
    monkeypatch.setattr(
        "lilo.telemetry.serving_metrics.subprocess.run",
        lambda *a, **kw: SimpleNamespace(
            stdout="0, 20, 10, 100, 200, 50\n1, 40, 20, 110, 200, [N/A]\n"
        ),
    )
    collector = ServingMetrics("inference")
    assert not any(
        m.name == "lilo.gpu.capacity" for m in collector.gpu_metrics(1_000_000_000)
    )
    rows = {m.name: m for m in collector.gpu_metrics(6_000_000_000)}
    assert rows["lilo.gpu.capacity"].data.data_points[0].value == 10
    assert rows["lilo.gpu.activity"].unit == "%"
    assert rows["lilo.gpu.memory_used"].data.data_points[0].value == 100 * 1024**2
    assert len(rows["lilo.gpu.power"].data.data_points) == 1


def test_native_metrics_reach_http_otlp_receiver(monkeypatch):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = ExportMetricsServiceRequest()
            payload.ParseFromString(
                self.rfile.read(int(self.headers["Content-Length"]))
            )
            received.append((self.path, self.headers.get("test-auth"), payload))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
        f"http://127.0.0.1:{server.server_port}/v1/metrics",
    )
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_HEADERS", "test-auth=local-test")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_COMPRESSION", "none")
    exporter = OTLPMetricExporter(timeout=2)
    try:
        converter = PrometheusDeltas({"lilo.instance_id": "test"})
        converter.convert(sample(), 10)
        metrics = converter.convert(sample(110, 3, 4, 1.6), 20)
        result = exporter.export(
            MetricsData(
                [
                    ResourceMetrics(
                        Resource.create({}),
                        [ScopeMetrics(InstrumentationScope("test"), metrics, "")],
                        "",
                    )
                ]
            )
        )
        assert result == MetricExportResult.SUCCESS
        assert received[0][:2] == ("/v1/metrics", "local-test")
        rows = {
            m.name: m
            for r in received[0][2].resource_metrics
            for scope in r.scope_metrics
            for m in scope.metrics
        }
        point = rows["sglang.generation_tokens"].sum.data_points[0]
        assert getattr(point, point.WhichOneof("value")) == 10
        assert (
            rows["sglang.time_to_first_token_seconds"].histogram.data_points[0].count
            == 1
        )
    finally:
        exporter.shutdown()
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)
