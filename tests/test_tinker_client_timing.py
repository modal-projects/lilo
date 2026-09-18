"""Tests for scripts/tinker_client_timing.py and lilo.request_timing."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import tinker_client_timing
from tinker.lib.public_interfaces.api_future import AwaitableConcurrentFuture
from tinker_client_timing import TinkerClientTimer

from lilo import request_timing


def test_mark_noop_when_disabled(capsys: pytest.CaptureFixture[str]) -> None:
    assert not request_timing.enabled({})
    request_timing.mark("test.mark", value=1)
    assert capsys.readouterr().out == ""


def test_mark_emits_json_when_enabled(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert request_timing.enabled({"LILO_REQUEST_TIMING": "1"})
    monkeypatch.setattr(request_timing, "_ENABLED", True)
    request_timing.mark("test.mark", request_id="m:1", bytes=42)
    line = capsys.readouterr().out.strip()
    payload = json.loads(line)
    assert payload["event"] == "lilo_request_mark"
    assert payload["mark"] == "test.mark"
    assert payload["request_id"] == "m:1"
    assert payload["bytes"] == 42
    assert payload["ts"] > 0


def test_flush_accumulates_and_resets() -> None:
    timer = TinkerClientTimer()
    timer.record_call("forward_backward", 0.5)
    timer.record_call("forward_backward", 0.25)
    timer.record_result("forward_backward", 1.0, 1.5)
    timer.record_http(
        "retrieve_future", 0.1, request_bytes=100, response_bytes=2048, status=200
    )
    timer.record_http(
        "retrieve_future", 0.2, request_bytes=100, response_bytes=None, status=408
    )
    metrics = timer.flush()
    assert metrics["client/forward_backward_calls"] == 2.0
    assert metrics["client/forward_backward_submit_s"] == pytest.approx(0.75)
    assert metrics["client/forward_backward_wait_s"] == pytest.approx(1.0)
    assert metrics["client/forward_backward_submit_to_result_s"] == pytest.approx(1.5)
    assert metrics["client/forward_backward_submit_ts"] > 0
    assert metrics["client/forward_backward_result_ts"] > 0
    assert metrics["client/http_retrieve_future_calls"] == 2.0
    assert metrics["client/http_retrieve_future_s"] == pytest.approx(0.3)
    assert metrics["client/http_retrieve_future_request_bytes"] == 200.0
    assert metrics["client/http_retrieve_future_response_bytes"] == 2048.0
    assert metrics["client/http_status_408_calls"] == 1.0
    assert metrics["client/http_calls"] == 2.0
    assert timer.flush() == {}


def test_timed_future_records_result() -> None:
    async def run() -> dict[str, float]:
        timer = TinkerClientTimer()
        inner = AwaitableConcurrentFuture(concurrent.futures.Future())

        def _resolve() -> None:
            inner.future().set_result("ok")

        loop = asyncio.get_running_loop()
        loop.call_later(0.01, _resolve)
        timed = tinker_client_timing._TimedFuture(
            inner,
            lambda wait_s, ts: timer.record_result("optim_step", wait_s, ts),
        )
        assert await timed.result_async() == "ok"
        return timer.flush()

    metrics = asyncio.run(run())
    assert metrics["client/optim_step_wait_s"] > 0
    assert metrics["client/optim_step_submit_to_result_s"] > 0


def test_http_send_wrapper_counts_request() -> None:
    timer = TinkerClientTimer()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.content == b"{}"
        return httpx.Response(200, json={"result": 1})

    async def run() -> None:
        try:
            timer.install()
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler), base_url="http://t"
            ) as client:
                await client.post("/api/v1/retrieve_future", content=b"{}")
        finally:
            timer.uninstall()

    asyncio.run(run())
    metrics = timer.flush()
    assert metrics["client/http_retrieve_future_calls"] == 1.0
    assert metrics["client/http_retrieve_future_request_bytes"] == 2.0
    assert metrics["client/http_retrieve_future_response_bytes"] == len(b'{"result":1}')
    assert metrics["client/http_calls"] == 1.0


def test_install_is_idempotent() -> None:
    timer = TinkerClientTimer()
    timer.install()
    first_send = httpx.AsyncClient.send
    timer.install()
    assert httpx.AsyncClient.send is first_send
    timer.uninstall()
