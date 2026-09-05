import asyncio
import json
import socket
import subprocess
import sys
import time

import httpx
import pytest

from lilo.engine import EngineServer, FutureStatus
from lilo.engine.api import Execution, OperationKind
from lilo.engine.backend import HttpExecutor, create_backend_app
from lilo.engine.operations import parse_operation_payload
from tests.support import EchoExecutor

MODEL_SPEC = {"base_model": "test/model", "parameterization": "full"}


def http_executor(backend_executor) -> HttpExecutor:
    app = create_backend_app(backend_executor)
    return HttpExecutor(
        "http://backend",
        transport=httpx.ASGITransport(app=app),
    )


def test_engine_runs_operations_over_backend_http() -> None:
    async def run() -> None:
        executor = http_executor(EchoExecutor())
        server = EngineServer(executor)
        await server.accept_model("model-a", MODEL_SPEC)
        body = json.dumps(
            {
                "model_id": "model-a",
                "seq_id": 1,
                "forward_backward_input": {
                    "data": [
                        {
                            "model_input": {"chunks": [{"tokens": [1]}]},
                            "loss_fn_inputs": {},
                        }
                    ],
                    "loss_fn": "cross_entropy",
                },
            }
        ).encode()
        await server.forward_backward(body, "application/json")
        state = await server.retrieve_future("model-a:1", timeout=2.0)
        assert state.status == FutureStatus.COMPLETE
        assert state.result == {
            "model_id": "model-a",
            "kind": "forward_backward",
            "payload": {
                "data": [
                    {
                        "loss_fn_inputs": {},
                        "model_input": {"chunks": [{"tokens": [1]}]},
                    }
                ],
                "loss_fn": "cross_entropy",
            },
        }
        request_id = await server.save_weights_for_sampler(
            {
                "model_id": "model-a",
                "seq_id": 2,
                "publish_version": 1,
            }
        )
        state = await server.retrieve_future(request_id, timeout=2.0)
        assert state.status == FutureStatus.COMPLETE
        assert state.result == {"publish_version": 1}
        await server.close()
        await executor.close()

    asyncio.run(run())


def test_backend_errors_cross_http() -> None:
    async def run() -> None:
        class FailingExecutor(EchoExecutor):
            async def execute(self, model_id, kind, payload):
                raise RuntimeError("cuda out of memory")

        executor = http_executor(FailingExecutor())
        server = EngineServer(executor)
        await server.accept_model("model-a", MODEL_SPEC)
        await server.optim_step({"model_id": "model-a", "seq_id": 1, "adam_params": {}})
        state = await server.retrieve_future("model-a:1", timeout=2.0)
        assert state.status == FutureStatus.FAILED
        assert state.error == "RuntimeError: cuda out of memory"
        await server.close()
        await executor.close()

    asyncio.run(run())


def test_backend_read_timeout_fences_engine() -> None:
    async def run() -> None:
        fenced = []

        async def timeout(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("stalled", request=request)

        executor = HttpExecutor(
            "http://backend",
            transport=httpx.MockTransport(timeout),
            read_timeout=42,
            on_read_timeout=lambda: fenced.append(True),
        )
        with pytest.raises(TimeoutError, match="backend /execute exceeded 42s"):
            await executor.execute(
                "model-a",
                OperationKind.OPTIM_STEP,
                parse_operation_payload(OperationKind.OPTIM_STEP, {"adam_params": {}}),
            )
        assert fenced == [True]
        await executor.close()

    asyncio.run(run())


def test_backend_batches_cross_http() -> None:
    async def run() -> None:
        executor = http_executor(EchoExecutor())
        results = await executor.execute_batch(
            (
                Execution(
                    "model-a",
                    OperationKind.FORWARD_BACKWARD,
                    parse_operation_payload(
                        OperationKind.FORWARD_BACKWARD,
                        {
                            "data": [
                                {
                                    "model_input": {"chunks": [{"tokens": [1]}]},
                                    "loss_fn_inputs": {},
                                }
                            ],
                            "loss_fn": "cross_entropy",
                        },
                    ),
                ),
                Execution(
                    "model-b",
                    OperationKind.FORWARD_BACKWARD,
                    parse_operation_payload(
                        OperationKind.FORWARD_BACKWARD,
                        {
                            "data": [
                                {
                                    "model_input": {"chunks": [{"tokens": [2]}]},
                                    "loss_fn_inputs": {},
                                }
                            ],
                            "loss_fn": "cross_entropy",
                        },
                    ),
                ),
            )
        )

        assert [result["model_id"] for result in results] == ["model-a", "model-b"]
        await executor.close()

    asyncio.run(run())


def test_backend_shutdown_is_idempotent() -> None:
    class CloseableExecutor(EchoExecutor):
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1

    async def run() -> None:
        backend = CloseableExecutor()
        executor = http_executor(backend)

        await executor.shutdown_backend()
        await executor.shutdown_backend()

        assert backend.close_calls == 1
        await executor.close()

    asyncio.run(run())


def test_backend_runner_serves_executor_in_subprocess() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    backend = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "lilo.engine.backend",
            "tests.support:EchoExecutor",
            str(port),
        ]
    )

    async def run() -> None:
        executor = HttpExecutor(f"http://127.0.0.1:{port}")
        deadline = time.time() + 10
        while True:
            assert backend.poll() is None, "backend exited"
            try:
                if (await executor.http.get("/healthz")).is_success:
                    break
            except httpx.TransportError:
                assert time.time() < deadline, "backend never became healthy"
                await asyncio.sleep(0.1)
        await executor.accept_model("model-a", MODEL_SPEC)
        result = await executor.execute(
            "model-a",
            OperationKind.OPTIM_STEP,
            parse_operation_payload(
                OperationKind.OPTIM_STEP,
                {"adam_params": {}},
            ),
        )
        assert result == {
            "model_id": "model-a",
            "kind": "optim_step",
            "payload": {"adam_params": {}},
        }
        await executor.shutdown_backend()
        await executor.close()

    try:
        asyncio.run(run())
    finally:
        backend.terminate()
        backend.wait(timeout=10)
