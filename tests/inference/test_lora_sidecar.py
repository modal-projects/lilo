import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from stitch.types import VersionRef

from lilo.inference.bulletin import SnapshotBulletin
from lilo.inference.lora_sidecar import create_app


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
            async with app.state.snapshot_access._condition:
                response = await asyncio.wait_for(
                    client.post("/generate", json=payload), 2
                )
            assert response.status_code == 200
            assert response.json()["meta_info"]["weight_version_start"] == 8
            assert len(refreshes) == 0
            assert len(loads) == 1
            payload["weight_version"] = {"min_version": 8}
            assert (await client.post("/generate", json=payload)).status_code == 200
            assert len(refreshes) == 0
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
        "weight_version": {},
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
            "weight_version": {},
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
            assert calls == 2  # The distinct constraints share the mount refresh.

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


def test_independent_adapters_load_concurrently_and_refresh_waits_for_all(tmp_path):
    bulletin = _publish(tmp_path, 1)
    _publish(tmp_path, 2)

    async def scenario():
        entered = [asyncio.Event(), asyncio.Event()]
        release = [asyncio.Event(), asyncio.Event()]
        refreshed = asyncio.Event()

        async def refresh():
            assert all(event.is_set() for event in release)
            refreshed.set()

        bulletin._refresh = refresh

        async def upstream(request):
            if request.url.path == "/load_lora_adapter":
                ref = VersionRef.parse(json.loads(request.content)["lora_name"])
                index = ref.version - 1
                entered[index].set()
                await release[index].wait()
            return httpx.Response(200, json={"meta_info": {}})

        app = create_app(
            bulletin, "http://sglang", transport=httpx.MockTransport(upstream)
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://sidecar"
            ) as client,
        ):

            def request(version):
                return client.post(
                    "/generate",
                    json={
                        "input_ids": [1],
                        "weight_run_id": "model-a",
                        "weight_version": {"exact_version": version},
                    },
                )

            requests = [asyncio.create_task(request(v)) for v in (1, 2)]
            try:
                await asyncio.wait_for(asyncio.gather(*(e.wait() for e in entered)), 2)
                refresh_task = asyncio.create_task(app.state.snapshot_access.refresh())
                release[0].set()
                await requests[0]
                await asyncio.sleep(0.01)
                assert not refreshed.is_set()
                release[1].set()
                await asyncio.wait_for(refresh_task, 2)
                assert refreshed.is_set()
            finally:
                for event in release:
                    event.set()
                await asyncio.gather(*requests)

    asyncio.run(scenario())


def test_exact_and_latest_share_registration_of_same_snapshot(tmp_path):
    bulletin = _publish(tmp_path, 1)

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
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://sidecar"
            ) as client,
        ):
            payload = {"input_ids": [1], "weight_run_id": "model-a"}
            first = asyncio.create_task(client.post("/generate", json=payload))
            try:
                await asyncio.wait_for(entered.wait(), 2)
                second = asyncio.create_task(
                    client.post(
                        "/generate",
                        json={
                            **payload,
                            "weight_version": {"exact_version": 1},
                        },
                    )
                )
                await asyncio.sleep(0.01)
                assert loads == 1
            finally:
                release.set()
            responses = await asyncio.gather(first, second)
            assert all(r.status_code == 200 for r in responses)
            assert loads == 1

    asyncio.run(scenario())


def test_warm_minimum_bypasses_refresh_and_higher_watermark_discovers_new_version(
    tmp_path,
):
    bulletin = _publish(tmp_path, 1)
    refreshes = []
    bulletin._refresh = lambda: refreshes.append(True)
    app = create_app(
        bulletin,
        "http://sglang",
        refresh_interval_s=60,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"meta_info": {}})
        ),
    )
    with TestClient(app) as client:
        payload = {
            "input_ids": [1],
            "weight_run_id": "model-a",
            "weight_version": {"min_version": 1},
        }
        assert (
            client.post("/generate", json=payload).json()["meta_info"][
                "weight_version_start"
            ]
            == 1
        )
        assert len(refreshes) == 1
        _publish(tmp_path, 2)
        for _ in range(3):
            assert (
                client.post("/generate", json=payload).json()["meta_info"][
                    "weight_version_start"
                ]
                == 1
            )
        assert len(refreshes) == 1  # No I/O on the satisfied-minimum path.
        payload["weight_version"] = {"min_version": 2}
        assert (
            client.post("/generate", json=payload).json()["meta_info"][
                "weight_version_start"
            ]
            == 2
        )
        assert len(refreshes) == 2
        payload["weight_version"] = {"min_version": 3}
        assert client.post("/generate", json=payload).status_code == 409


def test_background_discovery_advances_reused_minimum_sessions(tmp_path):
    bulletin = _publish(tmp_path, 1)

    async def scenario():
        advanced = asyncio.Event()

        async def upstream(request):
            if request.url.path == "/load_lora_adapter" and json.loads(request.content)[
                "lora_name"
            ].endswith("000002"):
                advanced.set()
            return httpx.Response(200, json={"meta_info": {}})

        app = create_app(
            bulletin,
            "http://sglang",
            refresh_interval_s=0.01,
            transport=httpx.MockTransport(upstream),
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://sidecar"
            ) as client,
        ):
            payload = {
                "input_ids": [1],
                "weight_run_id": "model-a",
                "weight_version": {"min_version": 1},
            }
            await client.post("/generate", json=payload)
            _publish(tmp_path, 2)
            await client.post("/generate", json=payload)
            await asyncio.wait_for(advanced.wait(), 2)
            # Wait for the load acknowledgment to update the cached high-watermark.
            await asyncio.sleep(0)
            response = await client.post("/generate", json=payload)
            assert response.json()["meta_info"]["weight_version_start"] == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("response_format", ["fastapi", "sglang", "stream"])
def test_transient_queue_rejection_retries_locally_with_fresh_id(
    tmp_path, response_format
):
    rids = []
    stream = response_format == "stream"

    def upstream(request):
        rids.append(json.loads(request.content)["rid"])
        if len(rids) == 1:
            if stream:
                body = {
                    "meta_info": {
                        "completion_tokens": 0,
                        "finish_reason": {
                            "type": "abort",
                            "status_code": 503,
                            "message": "The request queue is full.",
                        },
                    }
                }
                return httpx.Response(
                    200,
                    text="data: " + json.dumps(body) + "\n\ndata: [DONE]\n\n",
                    headers={"content-type": "text/event-stream"},
                )
            if response_format == "sglang":
                return httpx.Response(
                    503,
                    json={
                        "object": "error",
                        "message": "The request queue is full.",
                        "type": "503",
                        "param": None,
                        "code": 503,
                    },
                )
            return httpx.Response(503, json={"detail": "The request queue is full."})
        return httpx.Response(200, json={"meta_info": {"completion_tokens": 64}})

    app = create_app(
        SnapshotBulletin(tmp_path),
        "http://sglang",
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        r = client.post(
            "/generate", json={"input_ids": [1], "rid": "original", "stream": stream}
        )
        assert r.status_code == 200
        assert r.json()["meta_info"]["completion_tokens"] == 64
        assert r.json()["meta_info"]["lilo_admission_retries"] == 1
    assert rids == ["original", "original:admit-1"]


@pytest.mark.parametrize(
    "rejection",
    [
        {"detail": "The request queue is full."},
        {
            "object": "error",
            "message": "The request queue is full.",
            "type": "503",
            "param": None,
            "code": 503,
        },
    ],
)
def test_sustained_queue_rejection_is_returned_for_rerouting(tmp_path, rejection):
    calls = []

    def upstream(request):
        calls.append(True)
        return httpx.Response(503, json=rejection)

    app = create_app(
        SnapshotBulletin(tmp_path),
        "http://sglang",
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        assert client.post("/generate", json={"input_ids": [1]}).status_code == 503
    assert len(calls) == 3


def test_non_admission_failure_and_partial_output_are_never_retried(tmp_path):
    from lilo.inference.lora_sidecar import _queue_rejected

    assert not _queue_rejected(
        httpx.Response(503, json={"detail": "engine restarting"})
    )

    assert not _queue_rejected(
        httpx.Response(200, json={"meta_info": {"completion_tokens": 64}})
    )
    rejection = {
        "object": "error",
        "message": "The request queue is full.",
        "type": "503",
        "param": None,
        "code": 503,
    }
    for changes in [
        {"message": "engine restarting"},
        {"code": 500},
        {"text": "partial output"},
        {"meta_info": {"completion_tokens": 1}},
        {"meta_info": None},
    ]:
        assert not _queue_rejected(httpx.Response(503, json={**rejection, **changes}))
    body = {
        "meta_info": {
            "completion_tokens": 1,
            "finish_reason": {
                "type": "abort",
                "status_code": 503,
                "message": "The request queue is full.",
            },
        }
    }
    assert not _queue_rejected(
        httpx.Response(
            200,
            text="data: " + json.dumps(body) + "\n\n",
            headers={"content-type": "text/event-stream"},
        )
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"input_ids": [[1], [2]]},
        {"text": ["first", "second"]},
        {"input_ids": [1], "sampling_params": {"n": 2}},
        {"input_ids": [1], "sampling_params": [{}, {}]},
        {"input_ids": [1], "rid": ["first", "second"]},
    ],
)
@pytest.mark.parametrize(
    "rejection",
    [
        {"detail": "The request queue is full."},
        {
            "object": "error",
            "message": "The request queue is full.",
            "type": "503",
            "param": None,
            "code": 503,
        },
    ],
)
def test_batch_level_queue_errors_are_not_replayed(tmp_path, payload, rejection):
    calls = []

    def upstream(request):
        calls.append(True)
        return httpx.Response(503, json=rejection)

    app = create_app(
        SnapshotBulletin(tmp_path),
        "http://sglang",
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        assert client.post("/generate", json=payload).status_code == 503
    assert len(calls) == 1


def test_implicit_reload_is_coalesced_and_excludes_volume_refresh(tmp_path):
    bulletin = _publish(tmp_path, 1)

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        loads = 0
        refreshed = []

        async def upstream(request):
            nonlocal loads
            if request.url.path == "/load_lora_adapter":
                loads += 1
                if loads == 2:
                    entered.set()
                    await release.wait()
            return httpx.Response(200, json={"meta_info": {}})

        app = create_app(
            bulletin, "http://sglang", transport=httpx.MockTransport(upstream)
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://sidecar"
            ) as client,
        ):
            payload = {
                "input_ids": [1],
                "weight_run_id": "model-a",
                "weight_version": {"exact_version": 1},
            }
            await client.post("/generate", json=payload)

            async def refresh():
                assert release.is_set()
                refreshed.append(True)

            bulletin._refresh = refresh
            body = {"lora_name": VersionRef("model-a", 1).identity}
            first = asyncio.create_task(
                client.post("/internal/reload_lora_adapter", json=body)
            )
            try:
                await asyncio.wait_for(entered.wait(), 2)
                second = asyncio.create_task(
                    client.post("/internal/reload_lora_adapter", json=body)
                )
                refresh_task = asyncio.create_task(app.state.snapshot_access.refresh())
                await asyncio.sleep(0.01)
                assert not refreshed
                assert loads == 2
            finally:
                release.set()
            responses = await asyncio.gather(first, second, refresh_task)
            assert all(r.status_code == 200 for r in responses[:2])
            assert loads == 2 and refreshed == [True]
            assert (
                await client.post(
                    "/internal/reload_lora_adapter", json={"lora_name": "unknown"}
                )
            ).status_code == 404
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app, client=("192.0.2.1", 123)),
                base_url="http://sidecar",
            ) as external:
                assert (
                    await external.post("/internal/reload_lora_adapter", json=body)
                ).status_code == 404

    asyncio.run(scenario())
