import asyncio
from types import SimpleNamespace

import httpx
import pytest

from lilo.control_plane import ControlPlane, create_control_plane_app
from lilo.proto import tinker_public_pb2
from lilo.providers.local import (
    InMemoryKeyValueStore,
    LocalEnginePlatform,
)
from tests.support import EchoExecutor

BASE_MODEL = "Qwen/Qwen3-8B"
DEFINITION = "qwen3_8b"
DEFINITIONS = (
    SimpleNamespace(
        DEFINITION_ID=DEFINITION,
        MODEL_NAME=BASE_MODEL,
        PARAMETERIZATION="lora",
        CATALOG_VISIBLE=True,
        MAX_CONTEXT_LENGTH=16_384,
    ),
    SimpleNamespace(
        DEFINITION_ID=f"{DEFINITION}_full",
        MODEL_NAME=BASE_MODEL,
        PARAMETERIZATION="full",
        CATALOG_VISIBLE=True,
        MAX_CONTEXT_LENGTH=65_536,
    ),
)


def http_client(api_key: str | None = "tml-test") -> httpx.AsyncClient:
    plane = ControlPlane(
        InMemoryKeyValueStore(),
        LocalEnginePlatform(DEFINITION, EchoExecutor),
    )
    app = create_control_plane_app(
        plane,
        DEFINITIONS,
        api_key="tml-test",
        retrieve_window=1.0,
    )
    headers = {"X-API-Key": api_key} if api_key is not None else {}
    return httpx.AsyncClient(
        base_url="http://control-plane",
        headers=headers,
        transport=httpx.ASGITransport(app=app),
    )


async def created_model(client: httpx.AsyncClient) -> tuple[str, str]:
    created = await client.post(
        "/api/v1/create_session",
        json={"tags": [], "user_metadata": None, "sdk_version": "0.5.0"},
    )
    session_id = created.json()["session_id"]
    response = await client.post(
        "/api/v1/create_model",
        json={
            "session_id": session_id,
            "model_seq_id": 0,
            "base_model": BASE_MODEL,
            "lora_config": {"rank": 32},
        },
    )
    body = response.json()
    creation = await client.post(
        "/api/v1/retrieve_future",
        json={"request_id": body["request_id"]},
    )
    assert creation.json() == {
        "type": "create_model",
        "model_id": body["model_id"],
    }
    return session_id, body["model_id"]


def test_training_round_trip_over_http() -> None:
    async def run() -> None:
        client = http_client()
        _, model_id = await created_model(client)

        submitted = await client.post(
            "/api/v1/forward_backward",
            json={
                "model_id": model_id,
                "seq_id": 1,
                "forward_backward_input": {
                    "data": [
                        {
                            "model_input": {"chunks": [{"tokens": [1, 2]}]},
                            "loss_fn_inputs": {},
                        }
                    ],
                    "loss_fn": "cross_entropy",
                },
            },
        )
        request_id = submitted.json()["request_id"]

        retrieved = await client.post(
            "/api/v1/retrieve_future",
            json={"request_id": request_id},
        )
        assert retrieved.status_code == 200
        result = retrieved.json()
        assert result["kind"] == "forward_backward"
        assert result["payload"] == {
            "data": [
                {
                    "loss_fn_inputs": {},
                    "model_input": {"chunks": [{"tokens": [1, 2]}]},
                }
            ],
            "loss_fn": "cross_entropy",
        }
        await client.aclose()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("path", "body"),
    (
        (
            "/api/v1/save_weights_for_sampler",
            {"seq_id": 1, "path": "invalid/name"},
        ),
        (
            "/api/v1/load_weights",
            {"seq_id": 1, "path": "tinker://run/sampler_weights/checkpoint"},
        ),
        (
            "/api/v1/load_weights",
            {
                "seq_id": 1,
                "path": "tinker://run/weights/checkpoint",
                "weights_access_token": "token",
            },
        ),
        (
            "/api/v1/save_weights_for_sampler",
            {"seq_id": 1, "path": "checkpoint", "ttl_seconds": "invalid"},
        ),
        (
            "/api/v1/forward_backward",
            {
                "seq_id": 1,
                "forward_backward_input": {
                    "data": "invalid",
                    "loss_fn": "cross_entropy",
                },
            },
        ),
    ),
)
def test_rejected_operation_does_not_stall_model(path: str, body: dict) -> None:
    async def run() -> None:
        client = http_client()
        _, model_id = await created_model(client)
        rejected = await client.post(path, json={**body, "model_id": model_id})
        assert rejected.status_code == 400
        submitted = await client.post(
            "/api/v1/optim_step",
            json={"model_id": model_id, "seq_id": 2, "adam_params": {}},
        )
        resolved = await client.post(
            "/api/v1/retrieve_future",
            json={"request_id": submitted.json()["request_id"]},
        )
        assert resolved.status_code == 200
        assert resolved.json()["kind"] == "optim_step"
        await client.aclose()

    asyncio.run(run())


def test_rejected_protobuf_operation_does_not_stall_model() -> None:
    async def run() -> None:
        client = http_client()
        _, model_id = await created_model(client)
        message = tinker_public_pb2.ForwardBackwardRequest(
            model_id=model_id,
            seq_id=1,
            loss_fn="cross_entropy",
        )
        message.data.add().model_input.add()
        rejected = await client.post(
            "/api/v1/forward_backward",
            content=message.SerializeToString(),
            headers={"Content-Type": "application/x-protobuf"},
        )
        assert rejected.status_code == 400
        submitted = await client.post(
            "/api/v1/optim_step",
            json={"model_id": model_id, "seq_id": 2, "adam_params": {}},
        )
        resolved = await client.post(
            "/api/v1/retrieve_future",
            json={"request_id": submitted.json()["request_id"]},
        )
        assert resolved.status_code == 200
        assert resolved.json()["kind"] == "optim_step"
        await client.aclose()

    asyncio.run(run())


def test_pending_future_returns_try_again_408() -> None:
    async def run() -> None:
        client = http_client()
        _, model_id = await created_model(client)
        submitted = await client.post(
            "/api/v1/optim_step",
            json={"model_id": model_id, "seq_id": 2, "adam_params": {}},
        )
        request_id = submitted.json()["request_id"]
        retrieved = await client.post(
            "/api/v1/retrieve_future",
            json={"request_id": request_id},
        )
        assert retrieved.status_code == 408
        assert retrieved.json() == {
            "type": "try_again",
            "request_id": request_id,
            "queue_state": "active",
        }
        await client.aclose()

    asyncio.run(run())


def test_service_endpoints() -> None:
    async def run() -> None:
        client = http_client()
        assert (await client.get("/api/v1/healthz")).json() == {"status": "ok"}
        capabilities = await client.get("/api/v1/get_server_capabilities")
        assert capabilities.json() == {
            "supported_models": [
                {"model_name": BASE_MODEL, "max_context_length": 16_384}
            ],
        }
        config = await client.post("/api/v1/client/config", json={})
        assert config.json()["use_pyqwest_transport"] is False
        telemetry = await client.post("/api/v1/telemetry", json={})
        assert telemetry.json() == {"status": "accepted"}
        await client.aclose()

    asyncio.run(run())


def test_model_info_and_unload() -> None:
    async def run() -> None:
        client = http_client()
        _, model_id = await created_model(client)

        info = await client.post("/api/v1/get_info", json={"model_id": model_id})
        body = info.json()
        assert body["model_id"] == model_id
        assert body["model_data"]["model_name"] == BASE_MODEL
        assert body["is_lora"] is True
        assert body["lora_rank"] == 32
        assert body["parameterization"] == {"type": "lora"}

        unloaded = await client.post(
            "/api/v1/unload_model",
            json={"model_id": model_id},
        )
        request_id = unloaded.json()["request_id"]
        retrieved = await client.post(
            "/api/v1/retrieve_future",
            json={"request_id": request_id},
        )
        assert retrieved.json() == {
            "type": "unload_model",
            "model_id": model_id,
        }
        await client.aclose()

    asyncio.run(run())


def test_base_sampling_session_prefers_full_definition() -> None:
    async def run() -> None:
        plane = ControlPlane(
            InMemoryKeyValueStore(),
            LocalEnginePlatform(DEFINITION, EchoExecutor),
        )
        app = create_control_plane_app(plane, DEFINITIONS, api_key=None)
        client = httpx.AsyncClient(
            base_url="http://control-plane",
            transport=httpx.ASGITransport(app=app),
        )
        session = await client.post("/api/v1/create_session", json={})
        created = await client.post(
            "/api/v1/create_sampling_session",
            json={
                "session_id": session.json()["session_id"],
                "sampling_session_seq_id": 0,
                "base_model": BASE_MODEL,
            },
        )
        stored = await plane.get_sampling_session(
            created.json()["sampling_session_id"]
        )
        assert stored.engine_definition_id == f"{DEFINITION}_full"
        await client.aclose()

    asyncio.run(run())


def test_full_model_info_is_not_lora() -> None:
    async def run() -> None:
        client = http_client()
        session = await client.post("/api/v1/create_session", json={})
        created = await client.post(
            "/api/v1/create_model",
            json={
                "session_id": session.json()["session_id"],
                "model_seq_id": 0,
                "base_model": BASE_MODEL,
                "parameterization": {"type": "full"},
            },
        )
        model_id = created.json()["model_id"]
        await client.post(
            "/api/v1/retrieve_future",
            json={"request_id": created.json()["request_id"]},
        )

        info = await client.post("/api/v1/get_info", json={"model_id": model_id})
        assert info.json()["is_lora"] is False
        assert info.json()["lora_rank"] is None
        assert info.json()["parameterization"] == {"type": "full"}
        await client.aclose()

    asyncio.run(run())


def test_errors_map_to_status_codes() -> None:
    async def run() -> None:
        client = http_client()
        session_id, _ = await created_model(client)

        missing = await client.post(
            "/api/v1/session_heartbeat",
            json={"session_id": "no-such-session"},
        )
        assert missing.status_code == 404

        unsupported = await client.post(
            "/api/v1/create_model",
            json={
                "session_id": session_id,
                "model_seq_id": 1,
                "base_model": "not-a-model",
            },
        )
        assert unsupported.status_code == 400

        invalid_full = await client.post(
            "/api/v1/create_model",
            json={
                "session_id": session_id,
                "model_seq_id": 2,
                "base_model": BASE_MODEL,
                "parameterization": {"type": "full"},
                "lora_config": {"rank": 32},
            },
        )
        assert invalid_full.status_code == 400
        assert invalid_full.json()["error"] == "invalid_request"
        assert "cannot both be provided" in invalid_full.json()["message"]
        await client.aclose()

    asyncio.run(run())


def test_bad_api_key_is_rejected() -> None:
    async def run() -> None:
        client = http_client(api_key="tml-wrong")
        response = await client.post("/api/v1/create_session", json={})
        assert response.status_code == 401
        assert response.json() == {"error": "unauthorized", "message": "unauthorized"}
        await client.aclose()

    asyncio.run(run())


def test_rollout_pool_config_is_full_only_and_validated() -> None:
    async def run() -> None:
        client = http_client()
        session_id, _ = await created_model(client)

        def create(seq: int, **body):
            return client.post(
                "/api/v1/create_model",
                json={
                    "session_id": session_id,
                    "model_seq_id": seq,
                    "base_model": BASE_MODEL,
                    **body,
                },
            )

        full = {"parameterization": {"type": "full"}}
        accepted = await create(
            1, **full, rollout={"min_containers": 2, "max_containers": 4}
        )
        assert accepted.status_code == 200
        lora = await create(2, lora_config={"rank": 8}, rollout={"max_containers": 4})
        assert lora.status_code == 400
        assert "only configurable for full" in lora.json()["message"]
        inverted = await create(
            3, **full, rollout={"min_containers": 4, "max_containers": 2}
        )
        assert inverted.status_code == 400
        assert inverted.json()["error"] == "invalid_request"
        unknown = await create(4, **full, rollout={"target_concurrency": 8})
        assert unknown.status_code == 400
        assert unknown.json()["error"] == "invalid_request"
        await client.aclose()

    asyncio.run(run())
