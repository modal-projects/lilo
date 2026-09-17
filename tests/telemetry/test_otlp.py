import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from lilo.inference.sampling import sample_task
from lilo.telemetry import otlp


def task(n=1):
    return {
        "request_id": "otlp-test",
        "engine_definition_id": "test",
        "model_id": None,
        "accepted_at": time.time() - 1,
        "payload": {
            "num_samples": n,
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]},
            "sampling_params": {"max_tokens": 1},
        },
    }


@pytest.fixture
def spans(monkeypatch):
    exporter = InMemorySpanExporter()
    p = TracerProvider()
    p.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(otlp, "provider", lambda: p)
    yield exporter
    p.shutdown()


def test_real_sampling_path_preserves_sequence_timing_and_omits_payloads(spans):
    request = task(2)
    request["telemetry_tags"] = {
        "lilo.run_id": "run",
        "lilo.run_attempt_id": "attempt",
        "secret": "PRIVATE",
    }
    calls = 0

    def handle(req):
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "text": "PRIVATE OUTPUT",
                "meta_info": {
                    "id": json.loads(req.content)["rid"],
                    "output_token_logprobs": [[-0.1, 3]],
                    "prompt_tokens": 2,
                    "cached_tokens": calls - 1,
                    "completion_tokens": 1,
                    "queue_time": 0.1,
                    "forward_entry_time": 10,
                    "prefill_finished_time": 10.2,
                    "request_finished_ts": 11,
                    "decode_throughput": 999,
                },
            },
        )

    async def run():
        stats = {}
        with otlp.sample_trace(request, stats):
            await sample_task(
                request,
                "https://example.invalid",
                transport=httpx.MockTransport(handle),
                stats=stats,
            )

    asyncio.run(run())
    rows = spans.get_finished_spans()
    parent = next(s for s in rows if s.name == "lilo.sample")
    attempts = [s for s in rows if s.name == "lilo.sample.attempt"]
    assert parent.start_time == int(request["accepted_at"] * 1e9)
    assert parent.attributes["lilo.output_tokens"] == 2
    assert parent.attributes["lilo.attempt_count"] == 2
    assert parent.attributes["lilo.retry_count"] == 0
    assert {s.attributes["lilo.sequence_index"] for s in attempts} == {0, 1}
    assert {s.attributes["sglang.cached_tokens"] for s in attempts} == {0, 1}
    for s in (parent, *attempts):
        assert s.attributes["lilo.run_id"] == "run"
        assert s.attributes["lilo.run_attempt_id"] == "attempt"
        assert "secret" not in s.attributes
    for s in attempts:
        assert s.parent.span_id == parent.context.span_id
        assert s.context.trace_id == parent.context.trace_id
        assert s.attributes["sglang.prefill_s"] == pytest.approx(0.2)
        assert s.attributes["sglang.post_prefill_to_finish_s"] == pytest.approx(0.8)
        assert "sglang.decode_s" not in s.attributes
        assert "decode_throughput" not in str(s.attributes)
        assert "PRIVATE" not in str(s.attributes)
    assert "sglang.cached_tokens" not in parent.attributes


def test_missing_evidence_and_failure_do_not_become_zero_or_leak_errors(spans):
    def handle(req):
        return httpx.Response(400, text="PRIVATE PROMPT in error body")

    async def run():
        with otlp.sample_trace(task(), {}):
            await sample_task(
                task(), "https://example.invalid", transport=httpx.MockTransport(handle)
            )

    with pytest.raises(RuntimeError):
        asyncio.run(run())
    for s in spans.get_finished_spans():
        if s.name in {
            "lilo.sample",
            "lilo.sample.attempt",
            "lilo.sampling.inference_http",
        }:
            assert s.status.status_code == StatusCode.ERROR
        assert "sglang.cached_tokens" not in s.attributes
        assert "sglang.prefill_s" not in s.attributes
        assert not s.events  # no automatic exception/stack/payload capture
        assert "PRIVATE" not in str(s.attributes)


def test_retry_is_a_separate_attempt_in_same_trace(spans, monkeypatch):
    from lilo.inference import sampling

    async def no_wait(*args):
        pass

    monkeypatch.setattr(sampling, "_backoff", no_wait)
    calls = 0

    def handle(req):
        nonlocal calls
        calls += 1
        return (
            httpx.Response(503)
            if calls == 1
            else httpx.Response(
                200, json={"meta_info": {"output_token_logprobs": [[-0.1, 3]]}}
            )
        )

    async def run():
        with otlp.sample_trace(task(), {}):
            await sample_task(
                task(), "https://example.invalid", transport=httpx.MockTransport(handle)
            )

    asyncio.run(run())
    rows = [
        s
        for s in spans.get_finished_spans()
        if s.name in {"lilo.sample", "lilo.sample.attempt"}
    ]
    assert [s.status.status_code for s in rows] == [
        StatusCode.ERROR,
        StatusCode.OK,
        StatusCode.OK,
    ]
    assert rows[-1].attributes["lilo.retry_count"] == 1


def test_http_protobuf_export_uses_configured_endpoint_and_header(monkeypatch):
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )

    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(
                (
                    self.path,
                    self.headers.get("test-auth"),
                    self.rfile.read(int(self.headers["content-length"])),
                )
            )
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    otlp.provider.cache_clear()
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        f"http://127.0.0.1:{server.server_port}/v1/traces",
    )
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_HEADERS", "test-auth=local-test")
    try:
        with otlp.sample_trace(task(), {}):
            pass
        assert otlp.flush()
        path, auth, body = received[0]
        assert (path, auth) == ("/v1/traces", "local-test")
        decoded = ExportTraceServiceRequest.FromString(body)
        assert decoded.resource_spans[0].scope_spans[0].spans[0].name == "lilo.sample"
    finally:
        otlp.provider().shutdown()
        otlp.provider.cache_clear()
        server.shutdown()
        server.server_close()
        thread.join()


def test_no_endpoint_is_noop(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    otlp.provider.cache_clear()
    try:
        assert otlp.provider() is None
        with otlp.sample_trace(task(), {}):
            pass
    finally:
        otlp.provider.cache_clear()
