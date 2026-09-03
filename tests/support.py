from __future__ import annotations

import contextlib
import socket
import threading
import time
from collections.abc import Iterator

import uvicorn
from tinker import AdamParams
from tinker.types.forward_backward_input import ForwardBackwardInput

from lilo.engine.api import EngineApi, Execution, OperationKind
from lilo.engine.operations import (
    OperationPayload,
    SaveWeightsForSamplerPayload,
    serialize_operation_payload,
)
from lilo.providers.contracts import (
    EngineInstance,
    SamplingTask,
)


class EchoExecutor:
    async def accept_model(self, model_id: str, spec: object) -> None:
        return None

    async def execute(
        self,
        model_id: str,
        kind: OperationKind,
        payload: OperationPayload,
    ) -> object:
        return {
            "model_id": model_id,
            "kind": kind.value,
            "payload": serialize_operation_payload(payload),
        }

    async def execute_batch(
        self,
        executions: tuple[Execution, ...],
    ) -> tuple[object, ...]:
        return tuple(
            [
                await self.execute(item.model_id, item.kind, item.payload)
                for item in executions
            ]
        )

    async def capture_operation(
        self,
        model_id: str,
        kind: OperationKind,
        payload: object,
    ) -> object:
        if kind == OperationKind.SAVE_WEIGHTS:
            return await self.capture_checkpoint(model_id, payload)
        if kind == OperationKind.SAVE_WEIGHTS_FOR_SAMPLER:
            publication = await self.execute(model_id, kind, payload)
            if (
                not isinstance(publication, dict)
                or "publish_version" not in publication
            ):
                publication = {
                    "publish_version": (
                        payload.get("publish_version", 1)
                        if isinstance(payload, dict)
                        else 1
                    )
                }
            return {
                "model_id": model_id,
                "capture_id": "sampler-snapshot",
                "publication": publication,
            }
        raise ValueError(f"{kind.value} has no persistence phase")

    async def persist_operation(
        self,
        model_id: str,
        kind: OperationKind,
        payload: object,
        capture: object,
    ) -> object:
        if kind == OperationKind.SAVE_WEIGHTS:
            return await self.persist_checkpoint(model_id, payload, capture)
        if kind == OperationKind.SAVE_WEIGHTS_FOR_SAMPLER:
            return capture["publication"]
        raise ValueError(f"{kind.value} has no persistence phase")

    async def capture_checkpoint(self, model_id: str, payload: object) -> object:
        return {"model_id": model_id, "snapshot_id": "snapshot"}

    async def persist_checkpoint(
        self,
        model_id: str,
        payload: object,
        snapshot: object,
    ) -> object:
        return {
            "path": f"/checkpoints/{model_id}/weights/snapshot",
            "type": "save_weights",
        }

    async def unload_model(self, model_id: str) -> None:
        return None

    async def close(self) -> None:
        return None


class TinkerStubExecutor(EchoExecutor):
    async def execute(
        self,
        model_id: str,
        kind: OperationKind,
        payload: OperationPayload,
    ) -> object:
        if kind == OperationKind.FORWARD_BACKWARD:
            assert isinstance(payload, ForwardBackwardInput)
            return {
                "loss_fn_output_type": "scalar",
                "loss_fn_outputs": [{} for _ in payload.data],
                "metrics": {"loss:sum": 1.25},
            }
        if kind == OperationKind.OPTIM_STEP:
            assert isinstance(payload, AdamParams)
            return {"metrics": {"lr": payload.learning_rate}}
        if kind == OperationKind.SAVE_WEIGHTS_FOR_SAMPLER:
            assert isinstance(payload, SaveWeightsForSamplerPayload)
            return {"publish_version": payload.publish_version}
        return {}


class TinkerStubSampler:
    async def __call__(self, task: SamplingTask) -> object:
        count = int(task.payload["num_samples"])
        max_tokens = int(task.payload["sampling_params"]["max_tokens"])
        return {
            "type": "sample",
            "sequences": [
                {
                    "stop_reason": "length",
                    "tokens": list(range(index, index + max_tokens)),
                }
                for index in range(count)
            ],
        }


class SingleEnginePlatform:
    def __init__(self, definition_id: str, client: EngineApi) -> None:
        self.definition_id = definition_id
        self._client = client
        self.instance = EngineInstance(definition_id, "instance-1", "running")

    async def ensure_instance(
        self,
        definition_id: str,
    ) -> EngineInstance:
        assert definition_id == self.definition_id
        return self.instance

    async def spawn_instance(self, definition_id: str) -> EngineInstance:
        return await self.ensure_instance(definition_id)

    async def active_instances(
        self,
        definition_id: str,
    ) -> tuple[EngineInstance, ...]:
        assert definition_id == self.definition_id
        if self.instance.terminal:
            return ()
        return (self.instance,)

    async def get_instance(self, instance_id: str) -> EngineInstance | None:
        if instance_id != self.instance.instance_id:
            return None
        return self.instance

    async def list_instances(self) -> tuple[EngineInstance, ...]:
        return (self.instance,)

    def client(self, instance_id: str) -> EngineApi:
        return self._client

    async def set_instance_state(self, instance_id: str, state):
        assert instance_id == self.instance.instance_id
        self.instance = EngineInstance(
            self.definition_id, instance_id, state, self.instance.revision
        )
        return self.instance

    async def stop_instance(self, instance_id: str) -> None:
        await self.set_instance_state(instance_id, "stopped")


@contextlib.contextmanager
def serve(app) -> Iterator[str]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started:
        assert time.time() < deadline, "server did not start"
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
