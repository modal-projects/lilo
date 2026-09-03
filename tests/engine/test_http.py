import asyncio
import json

import httpx
import pytest
from tinker import types
from tinker.proto.request_conv import forward_backward_request_to_proto
from tinker.types.forward_backward_input import ForwardBackwardInput
from tinker.types.forward_backward_request import ForwardBackwardRequest

from lilo.engine import (
    EngineServer,
    FutureStatus,
    HttpEngineClient,
    create_engine_app,
)
from lilo.errors import RecordNotFound, SequenceConflict
from tests.support import EchoExecutor


def http_client(
    server: EngineServer,
    *,
    token: str | None = "secret",
) -> HttpEngineClient:
    app = create_engine_app(server, token="secret")
    return HttpEngineClient(
        "http://engine",
        token=token,
        transport=httpx.ASGITransport(app=app),
    )


def forward_backward_json(seq_id: int, data: object = None) -> bytes:
    tokens = list(data if data is not None else [seq_id])
    return json.dumps(
        {
            "model_id": "model-a",
            "seq_id": seq_id,
            "forward_backward_input": {
                "data": [
                    {
                        "model_input": {"chunks": [{"tokens": tokens}]},
                        "loss_fn_inputs": {},
                    }
                ],
                "loss_fn": "cross_entropy",
            },
        }
    ).encode()


def test_full_round_trip_over_http() -> None:
    async def run() -> None:
        server = EngineServer(EchoExecutor())
        client = http_client(server)

        assert await client.accept_model("model-a", {"rank": 8})
        assert await client.model_ids() == ("model-a",)
        request_id = await client.forward_backward(
            forward_backward_json(1, data=[1, 2]),
            "application/json",
        )
        assert request_id == "model-a:1"
        state = await client.retrieve_future(request_id, timeout=1.0)
        assert state.status == FutureStatus.COMPLETE
        assert state.result == {
            "model_id": "model-a",
            "kind": "forward_backward",
            "payload": {
                "data": [
                    {
                        "loss_fn_inputs": {},
                        "model_input": {
                            "chunks": [
                                {
                                    "tokens": [1, 2],
                                }
                            ]
                        },
                    }
                ],
                "loss_fn": "cross_entropy",
            },
        }

        request_id = await client.optim_step(
            {"model_id": "model-a", "seq_id": 2, "adam_params": {}}
        )
        state = await client.retrieve_future(request_id, timeout=1.0)
        assert state.status == FutureStatus.COMPLETE

        request_id = await client.skip_sequence("model-a", 3, "invalid request")
        state = await client.retrieve_future(request_id, timeout=1.0)
        assert state.status == FutureStatus.FAILED
        assert state.error == "invalid request"

        assert await client.retrieve_future("model-a:9") is None
        await client.unload_model("model-a")
        assert await client.model_ids() == ()
        await client.close()
        await server.close()

    asyncio.run(run())


def test_proto_forward_backward_over_http() -> None:
    async def run() -> None:
        server = EngineServer(EchoExecutor())
        client = http_client(server)
        assert await client.accept_model("model-a", {})

        request = ForwardBackwardRequest(
            model_id="model-a",
            seq_id=1,
            forward_backward_input=ForwardBackwardInput(
                data=[
                    types.Datum(
                        model_input=types.ModelInput.from_ints([1, 2, 3]),
                        loss_fn_inputs={"target_tokens": [2, 3, 4]},
                    )
                ],
                loss_fn="cross_entropy",
            ),
        )
        body = forward_backward_request_to_proto(request).SerializeToString()
        request_id = await client.forward_backward(
            body,
            "application/x-protobuf",
        )
        state = await client.retrieve_future(request_id, timeout=1.0)
        assert state.status == FutureStatus.COMPLETE
        payload = state.result["payload"]
        assert payload["loss_fn"] == "cross_entropy"
        assert len(payload["data"]) == 1
        await client.close()
        await server.close()

    asyncio.run(run())


def test_typed_errors_cross_http() -> None:
    async def run() -> None:
        server = EngineServer(EchoExecutor())
        client = http_client(server)

        with pytest.raises(RecordNotFound):
            await client.forward_backward(
                forward_backward_json(1),
                "application/json",
            )

        assert await client.accept_model("model-a", {})
        await client.forward_backward(forward_backward_json(1), "application/json")
        await client.forward_backward(forward_backward_json(1), "application/json")
        with pytest.raises(SequenceConflict):
            await client.forward_backward(
                forward_backward_json(1, data=[999]),
                "application/json",
            )
        with pytest.raises(ValueError):
            await client.optim_step({"adam_params": {}})

        await client.close()
        await server.close()

    asyncio.run(run())


def test_bad_token_is_rejected() -> None:
    async def run() -> None:
        server = EngineServer(EchoExecutor())
        client = http_client(server, token="wrong")
        with pytest.raises(httpx.HTTPStatusError):
            await client.accept_model("model-a", {})
        await client.close()

    asyncio.run(run())
