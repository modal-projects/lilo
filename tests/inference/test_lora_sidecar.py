import json

import httpx
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
