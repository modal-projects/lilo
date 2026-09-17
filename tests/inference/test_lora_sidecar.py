import asyncio
import json

import httpx
from fastapi.testclient import TestClient
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from stitch.types import VersionRef

from lilo.inference import lora_sidecar
from lilo.inference.bulletin import SnapshotBulletin
from lilo.inference.lora_sidecar import create_app
from lilo.telemetry.performance import Stages


def _publish(tmp_path, version):
    source = tmp_path / f"source-{version}"
    source.mkdir()
    (source / "adapter_model.safetensors").write_bytes(f"v{version}".encode())
    (source / "adapter_config.json").write_text(
        json.dumps({"r": 8}),
        encoding="utf-8",
    )
    bulletin = SnapshotBulletin(tmp_path / "bulletin")
    bulletin.publish(VersionRef("model-a", version), source)
    return bulletin


def test_sidecar_resolves_exact_adapter_and_stamps_response(tmp_path) -> None:
    bulletin = _publish(tmp_path, 3)
    adapter = VersionRef("model-a", 3)

    def upstream(request):
        payload = json.loads(request.content)
        if request.url.path == "/load_lora_adapter":
            assert payload == {
                "lora_name": adapter.identity,
                "lora_path": str(bulletin.resolve(adapter)),
                "pinned": False,
            }
            return httpx.Response(200, json={"success": True})
        assert request.url.path == "/generate"
        assert payload["lora_path"] == adapter.identity
        assert payload["rid"].startswith("lilo-")
        assert "weight_run_id" not in payload
        assert "weight_version" not in payload
        return httpx.Response(200, json={"meta_info": {}, "text": "ok"})

    app = create_app(
        bulletin,
        "http://sglang",
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post(
            "/generate",
            json={
                "input_ids": [1],
                "weight_run_id": "model-a",
                "weight_version": {"exact_version": 3, "min_version": None},
            },
        )

    assert response.status_code == 200
    assert response.json()["meta_info"] == {
        "weight_version_start": 3,
        "weight_version_end": 3,
    }


def test_sidecar_uses_latest_adapter_satisfying_minimum(tmp_path) -> None:
    bulletin = _publish(tmp_path, 4)
    adapter = VersionRef("model-a", 4)

    def upstream(request):
        payload = json.loads(request.content)
        if request.url.path == "/load_lora_adapter":
            assert payload["lora_name"] == adapter.identity
            assert payload["lora_path"] == str(bulletin.resolve(adapter))
            return httpx.Response(200, json={"success": True})
        assert payload["lora_path"] == adapter.identity
        return httpx.Response(200, json={"meta_info": {}})

    app = create_app(
        bulletin,
        "http://sglang",
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post(
            "/generate",
            json={
                "input_ids": [1],
                "weight_run_id": "model-a",
                "weight_version": {"exact_version": None, "min_version": 3},
            },
        )

    assert response.status_code == 200
    assert response.json()["meta_info"]["weight_version_start"] == 4


def test_sidecar_returns_retryable_conflict_when_adapter_is_missing(tmp_path) -> None:
    called = False

    def upstream(_request):
        nonlocal called
        called = True
        return httpx.Response(200, json={})

    app = create_app(
        SnapshotBulletin(tmp_path / "bulletin"),
        "http://sglang",
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post(
            "/generate",
            json={
                "input_ids": [1],
                "weight_run_id": "model-a",
                "weight_version": {"exact_version": 2, "min_version": None},
            },
        )

    assert response.status_code == 409
    assert not called


def test_sidecar_passes_base_model_requests_without_lora(tmp_path) -> None:
    def upstream(request):
        payload = json.loads(request.content)
        assert "lora_path" not in payload
        return httpx.Response(200, json={"meta_info": {}})

    app = create_app(
        SnapshotBulletin(tmp_path / "bulletin"),
        "http://sglang",
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post("/generate", json={"input_ids": [1]})

    assert response.status_code == 200
    assert response.json()["meta_info"] == {}


def test_sidecar_registers_each_adapter_once(tmp_path) -> None:
    bulletin = _publish(tmp_path, 5)
    loads = 0

    def upstream(request):
        nonlocal loads
        if request.url.path == "/load_lora_adapter":
            loads += 1
            return httpx.Response(200, json={"success": True})
        return httpx.Response(200, json={"meta_info": {}})

    app = create_app(
        bulletin,
        "http://sglang",
        transport=httpx.MockTransport(upstream),
    )
    payload = {
        "input_ids": [1],
        "weight_run_id": "model-a",
        "weight_version": {"exact_version": 5, "min_version": None},
    }
    with TestClient(app) as client:
        assert client.post("/generate", json=payload).status_code == 200
        assert client.post("/generate", json=payload).status_code == 200

    assert loads == 1


def test_sidecar_propagates_adapter_registration_failure(tmp_path) -> None:
    bulletin = _publish(tmp_path, 6)

    def upstream(request):
        assert request.url.path == "/load_lora_adapter"
        return httpx.Response(400, json={"success": False, "error": "invalid adapter"})

    app = create_app(
        bulletin,
        "http://sglang",
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post(
            "/generate",
            json={
                "input_ids": [1],
                "weight_run_id": "model-a",
                "weight_version": {"exact_version": 6, "min_version": None},
            },
        )

    assert response.status_code == 400
    assert response.json()["error"] == "invalid adapter"


def test_volume_refresh_cannot_overlap_adapter_loading(tmp_path) -> None:
    bulletin = _publish(tmp_path, 7)

    async def scenario():
        loading = asyncio.Event()
        release = asyncio.Event()
        refreshes = []

        async def refresh():
            assert not loading.is_set(), "volume refreshed during SGLang file reads"
            refreshes.append(True)

        bulletin._refresh = refresh

        async def upstream(request):
            if request.url.path == "/load_lora_adapter":
                loading.set()
                await release.wait()
                loading.clear()
            return httpx.Response(200, json={"meta_info": {}})

        app = create_app(
            bulletin, "http://sglang", transport=httpx.MockTransport(upstream)
        )
        payload = {
            "input_ids": [1],
            "weight_run_id": "model-a",
            "weight_version": {"exact_version": 7},
        }
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://sidecar"
            ) as client,
        ):
            first = asyncio.create_task(client.post("/generate", json=payload))
            await asyncio.wait_for(loading.wait(), 2)
            second = asyncio.create_task(
                client.post(
                    "/generate",
                    json={
                        **payload,
                        "weight_version": {"min_version": 7},
                    },
                )
            )
            await asyncio.sleep(0.05)
            assert len(refreshes) == 0
            release.set()
            results = await asyncio.wait_for(asyncio.gather(first, second), 2)
            assert all(result.status_code == 200 for result in results)
            assert len(refreshes) == 1

    asyncio.run(scenario())


def test_loaded_exact_version_bypasses_refresh_and_registration_lock(tmp_path) -> None:
    bulletin = _publish(tmp_path, 8)

    async def scenario():
        refreshes = []
        loads = []

        async def refresh():
            refreshes.append(True)

        bulletin._refresh = refresh

        def upstream(request):
            if request.url.path == "/load_lora_adapter":
                loads.append(True)
            return httpx.Response(200, json={"meta_info": {}})

        app = create_app(
            bulletin, "http://sglang", transport=httpx.MockTransport(upstream)
        )
        payload = {
            "input_ids": [1],
            "weight_run_id": "model-a",
            "weight_version": {"exact_version": 8},
        }
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://sidecar"
            ) as client,
        ):
            assert (await client.post("/generate", json=payload)).status_code == 200
            async with app.state.adapter_lock:
                response = await asyncio.wait_for(
                    client.post("/generate", json=payload), 2
                )
            assert response.status_code == 200
            assert response.json()["meta_info"]["weight_version_start"] == 8
            assert len(refreshes) == 0
            assert len(loads) == 1
            payload["weight_version"] = {"min_version": 8}
            assert (await client.post("/generate", json=payload)).status_code == 200
            assert len(refreshes) == 1
            assert len(loads) == 1

    asyncio.run(scenario())


def test_latest_validates_each_version_once_and_observes_publications(
    tmp_path, monkeypatch
):
    bulletin = _publish(tmp_path, 8)
    resolved = []
    loaded = []
    original = bulletin.resolve

    def resolve(ref):
        resolved.append(ref.version)
        return original(ref)

    monkeypatch.setattr(bulletin, "resolve", resolve)

    def upstream(request):
        if request.url.path == "/load_lora_adapter":
            loaded.append(json.loads(request.content)["lora_name"])
        return httpx.Response(200, json={"meta_info": {}})

    app = create_app(bulletin, "http://sglang", transport=httpx.MockTransport(upstream))
    payload = {
        "input_ids": [1],
        "weight_run_id": "model-a",
        "weight_version": {"min_version": 8},
    }
    with TestClient(app) as client:
        for _ in range(3):
            response = client.post("/generate", json=payload)
            assert response.status_code == 200
            assert response.json()["meta_info"]["weight_version_start"] == 8
        _publish(tmp_path, 9)
        for _ in range(2):
            response = client.post("/generate", json=payload)
            assert response.json()["meta_info"]["weight_version_start"] == 9
        payload["weight_version"] = {"min_version": 10}
        assert client.post("/generate", json=payload).status_code == 409
    assert resolved == [8, 9]
    assert loaded == [VersionRef("model-a", v).identity for v in (8, 9)]


def test_concurrent_latest_requests_share_one_refresh(tmp_path):
    bulletin = _publish(tmp_path, 12)

    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()
        refreshes = 0
        loads = 0

        async def refresh():
            nonlocal refreshes
            refreshes += 1
            entered.set()
            await release.wait()

        bulletin._refresh = refresh

        def upstream(request):
            nonlocal loads
            if request.url.path == "/load_lora_adapter":
                loads += 1
            return httpx.Response(200, json={"meta_info": {}})

        app = create_app(
            bulletin, "http://sglang", transport=httpx.MockTransport(upstream)
        )
        payload = {
            "input_ids": [1],
            "weight_run_id": "model-a",
            "weight_version": {"min_version": 12},
        }
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://sidecar"
            ) as client,
        ):
            requests = [
                asyncio.create_task(client.post("/generate", json=payload))
                for _ in range(8)
            ]
            await asyncio.wait_for(entered.wait(), 1)
            await asyncio.sleep(0.01)
            release.set()
            responses = await asyncio.wait_for(asyncio.gather(*requests), 2)
            assert all(r.status_code == 200 for r in responses)
            assert refreshes == 1
            assert loads == 1
            # A later request still checks for a newer publication.
            _publish(tmp_path, 13)
            response = await client.post("/generate", json=payload)
            assert response.json()["meta_info"]["weight_version_start"] == 13
            assert refreshes == 2

    asyncio.run(scenario())


def test_cancelled_waiter_does_not_cancel_shared_adapter_registration(tmp_path):
    bulletin = _publish(tmp_path, 15)

    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()
        loads = 0

        async def upstream(request):
            nonlocal loads
            if request.url.path == "/load_lora_adapter":
                loads += 1
                entered.set()
                await release.wait()
            return httpx.Response(200, json={"meta_info": {}})

        app = create_app(
            bulletin, "http://sglang", transport=httpx.MockTransport(upstream)
        )
        payload = {
            "input_ids": [1],
            "weight_run_id": "model-a",
            "weight_version": {"min_version": 15},
        }
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://sidecar"
            ) as client,
        ):
            first = asyncio.create_task(client.post("/generate", json=payload))
            await asyncio.wait_for(entered.wait(), 1)
            second = asyncio.create_task(client.post("/generate", json=payload))
            await asyncio.sleep(0.01)
            first.cancel()
            try:
                await first
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("expected cancelled request")
            assert not second.done()
            release.set()
            assert (await asyncio.wait_for(second, 1)).status_code == 200
            assert loads == 1
            assert not app.state.adapter_resolutions

    asyncio.run(scenario())


def test_shared_resolution_failure_is_retryable_and_minimums_are_distinct(tmp_path):
    bulletin = _publish(tmp_path, 16)

    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def refresh():
            nonlocal calls
            calls += 1
            if calls == 1:
                entered.set()
                await release.wait()
                raise RuntimeError("refresh failed")

        bulletin._refresh = refresh
        app = create_app(
            bulletin,
            "http://sglang",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"meta_info": {}})
            ),
        )
        payload = {
            "input_ids": [1],
            "weight_run_id": "model-a",
            "weight_version": {"min_version": 16},
        }
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://sidecar"
            ) as client,
        ):
            first = asyncio.create_task(client.post("/generate", json=payload))
            await asyncio.wait_for(entered.wait(), 1)
            second = asyncio.create_task(client.post("/generate", json=payload))
            await asyncio.sleep(0.01)
            release.set()
            results = await asyncio.gather(first, second, return_exceptions=True)
            assert all(isinstance(r, RuntimeError) for r in results)
            assert calls == 1
            assert not app.state.adapter_resolutions
            current, future = await asyncio.gather(
                client.post("/generate", json=payload),
                client.post(
                    "/generate",
                    json={**payload, "weight_version": {"min_version": 17}},
                ),
            )
            assert current.status_code == 200
            assert future.status_code == 409
            assert calls == 3

    asyncio.run(scenario())


def test_exact_version_refreshes_only_when_snapshot_is_missing(tmp_path):
    bulletin = SnapshotBulletin(tmp_path / "bulletin")
    refreshes = []

    def refresh():
        refreshes.append(True)
        _publish(tmp_path, 21)

    bulletin._refresh = refresh
    app = create_app(
        bulletin,
        "http://sglang",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"meta_info": {}})
        ),
    )
    with TestClient(app) as client:
        for _ in range(2):
            response = client.post(
                "/generate",
                json={
                    "input_ids": [1],
                    "weight_run_id": "model-a",
                    "weight_version": {"exact_version": 21},
                },
            )
            assert response.status_code == 200
            assert response.json()["meta_info"]["weight_version_start"] == 21
    assert refreshes == [True]


def test_live_metrics_show_adapter_lock_wait_and_trace_context(tmp_path, monkeypatch):
    bulletin = _publish(tmp_path, 3)
    _publish(tmp_path, 4)
    reader = InMemoryMetricReader()
    metrics = MeterProvider(metric_readers=[reader])
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    stages = Stages("inference", metrics=metrics, tracer=provider.get_tracer("test"))
    monkeypatch.setattr(lora_sidecar, "stages", lambda _: stages)

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()

        async def handle(req):
            if req.url.path == "/load_lora_adapter":
                entered.set()
                await release.wait()
            return httpx.Response(200, json={"meta_info": {}})

        app = create_app(
            bulletin, "http://sglang", transport=httpx.MockTransport(handle)
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://sidecar"
            ) as client:

                def request(version):
                    return client.post(
                        "/generate",
                        json={
                            "input_ids": [1],
                            "weight_run_id": "model-a",
                            "weight_version": {"exact_version": version},
                        },
                        headers={
                            "traceparent": "00-11111111111111111111111111111111-2222222222222222-01"
                        },
                    )

                first = asyncio.create_task(request(3))
                await asyncio.wait_for(entered.wait(), 1)
                second = asyncio.create_task(request(4))

                async def wait_for_lock():
                    while not any(
                        p.value == 1
                        and p.attributes["lilo.stage"] == "adapter_lock_wait"
                        for p in stages.observe("inflight", None)
                    ):
                        await asyncio.sleep(0.001)

                await asyncio.wait_for(wait_for_lock(), 1)
                points = {
                    p.attributes["lilo.stage"]: p.value
                    for p in stages.observe("inflight", None)
                }
                assert points["adapter_load"] == 1
                assert points["adapter_lock_wait"] == 1
                assert points["request"] == 2
                assert not any(
                    s.name.endswith(".adapter_load")
                    for s in exporter.get_finished_spans()
                )
                release.set()
                responses = await asyncio.gather(first, second)
                assert all(r.status_code == 200 for r in responses)
                assert all(p.value == 0 for p in stages.observe("inflight", None))

    try:
        asyncio.run(run())
        spans = exporter.get_finished_spans()
        assert all(s.context.trace_id == int("1" * 32, 16) for s in spans)
        roots = [s for s in spans if s.name == "lilo.inference.request"]
        assert all(s.parent.span_id == int("2" * 16, 16) for s in roots)
    finally:
        metrics.shutdown()
        provider.shutdown()
