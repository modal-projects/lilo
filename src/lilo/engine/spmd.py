from __future__ import annotations

import asyncio
import logging
import os
import threading
import uuid
from collections.abc import Mapping
from dataclasses import asdict
from datetime import timedelta
from typing import Any

from tinker import AdamParams, ForwardBackwardOutput
from tinker.types.forward_backward_input import ForwardBackwardInput

from lilo.backends.contract import (
    CommandBackend,
    ForwardBatch,
    ForwardItem,
    ModelSpec,
)

from .api import Execution, OperationKind
from .operations import (
    LoadWeightsPayload,
    OperationPayload,
    SaveWeightsForSamplerPayload,
    SaveWeightsPayload,
)

COMMAND_LANE_TIMEOUT = timedelta(days=365)


def _serialize_forward_output(output: ForwardBackwardOutput) -> dict[str, Any]:
    loss_fn_outputs = []
    for record in output.loss_fn_outputs:
        serialized = {}
        for key, tensor in record.items():
            value = {
                "data": tensor.data,
                "dtype": tensor.dtype,
                "shape": tensor.shape,
            }
            if tensor.sparse_crow_indices is not None:
                value["sparse_crow_indices"] = tensor.sparse_crow_indices
            if tensor.sparse_col_indices is not None:
                value["sparse_col_indices"] = tensor.sparse_col_indices
            serialized[key] = value
        loss_fn_outputs.append(serialized)
    return {
        "loss_fn_output_type": output.loss_fn_output_type,
        "loss_fn_outputs": loss_fn_outputs,
        "metrics": dict(output.metrics),
    }


def initialize_distributed_runtime() -> tuple[Any, Any, Any]:
    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank)
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29541")
    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    command_group = (
        dist.new_group(backend="gloo", timeout=COMMAND_LANE_TIMEOUT)
        if world_size > 1
        else None
    )
    checkpoint_persistence_group = (
        dist.new_group(backend="gloo", timeout=COMMAND_LANE_TIMEOUT)
        if world_size > 1
        else None
    )
    sampler_persistence_group = (
        dist.new_group(backend="gloo", timeout=COMMAND_LANE_TIMEOUT)
        if world_size > 1
        else None
    )
    return command_group, checkpoint_persistence_group, sampler_persistence_group


class DistributedExecutor:
    def __init__(
        self,
        backend: CommandBackend,
        *,
        command_group=None,
        checkpoint_persistence_group=None,
        sampler_persistence_group=None,
    ) -> None:
        self.backend = backend
        self.command_group = command_group
        self.checkpoint_persistence_group = checkpoint_persistence_group
        self.sampler_persistence_group = sampler_persistence_group
        self._closed = False

    async def accept_model(self, model_id: str, spec: ModelSpec) -> None:
        await asyncio.to_thread(
            self._dispatch,
            self.command_group,
            "accept_model",
            (model_id, spec),
        )

    async def execute(
        self,
        model_id: str,
        kind: OperationKind,
        payload: OperationPayload,
    ) -> object:
        if kind == OperationKind.FORWARD_BACKWARD:
            (result,) = await self.execute_batch((Execution(model_id, kind, payload),))
            return result

        if kind == OperationKind.LOAD_WEIGHTS:
            if not isinstance(payload, LoadWeightsPayload):
                raise TypeError("load_weights requires a load payload")
            await asyncio.to_thread(
                self._dispatch,
                self.command_group,
                "load_checkpoint",
                (model_id, payload.uri),
                {
                    "restore_optimizer": payload.restore_optimizer,
                },
            )
            return {"path": payload.uri, "type": "load_weights"}

        if kind == OperationKind.FORWARD:
            if not isinstance(payload, ForwardBackwardInput):
                raise TypeError("forward requires a forward payload")
            batch = ForwardBatch(
                items=(ForwardItem(model_id, tuple(payload.data)),),
                loss_fn=payload.loss_fn,
                loss_fn_config=payload.loss_fn_config or {},
                forward_only=True,
            )
            outputs = await asyncio.to_thread(
                self._dispatch,
                self.command_group,
                "forward_backward",
                (batch,),
            )
            (output,) = outputs
            return _serialize_forward_output(output)

        if kind == OperationKind.OPTIM_STEP:
            if not isinstance(payload, AdamParams):
                raise TypeError("optim_step requires an optimizer payload")
            outputs = await asyncio.to_thread(
                self._dispatch,
                self.command_group,
                "optim_step",
                ((model_id,), payload),
            )
            (output,) = outputs
            return output.model_dump(mode="json")

        raise ValueError(f"unsupported operation: {kind.value}")

    async def execute_batch(
        self,
        executions: tuple[Execution, ...],
    ) -> tuple[object, ...]:
        if not executions:
            raise ValueError("forward_backward batch cannot be empty")
        if any(
            execution.kind != OperationKind.FORWARD_BACKWARD for execution in executions
        ):
            raise ValueError("batch contains a non-forward_backward operation")
        payloads: list[ForwardBackwardInput] = []
        for execution in executions:
            if not isinstance(execution.payload, ForwardBackwardInput):
                raise TypeError("forward_backward requires a forward payload")
            payloads.append(execution.payload)
        loss_fn = payloads[0].loss_fn
        loss_fn_config = payloads[0].loss_fn_config or {}
        if any(
            payload.loss_fn != loss_fn
            or (payload.loss_fn_config or {}) != loss_fn_config
            for payload in payloads[1:]
        ):
            raise ValueError("forward_backward batch has incompatible losses")

        batch = ForwardBatch(
            items=tuple(
                ForwardItem(
                    execution.model_id,
                    tuple(payload.data),
                )
                for execution, payload in zip(
                    executions,
                    payloads,
                    strict=True,
                )
            ),
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
        )
        outputs = await asyncio.to_thread(
            self._dispatch,
            self.command_group,
            "forward_backward",
            (batch,),
        )
        if len(outputs) != len(executions):
            raise RuntimeError("backend returned the wrong result count")
        return tuple(_serialize_forward_output(output) for output in outputs)

    async def capture_operation(
        self,
        model_id: str,
        kind: OperationKind,
        payload: OperationPayload,
    ) -> object:
        if kind == OperationKind.SAVE_WEIGHTS:
            if not isinstance(payload, SaveWeightsPayload):
                raise TypeError("save_weights requires a checkpoint payload")
            snapshot_id = uuid.uuid4().hex
            await asyncio.to_thread(
                self._dispatch,
                self.command_group,
                "capture_checkpoint",
                (model_id, snapshot_id),
                {
                    "destination": payload.destination,
                    "include_optimizer": payload.include_optimizer,
                },
            )
            return {
                "model_id": model_id,
                "snapshot_id": snapshot_id,
                "destination": payload.destination,
            }
        if kind == OperationKind.SAVE_WEIGHTS_FOR_SAMPLER:
            if not isinstance(payload, SaveWeightsForSamplerPayload):
                raise TypeError("sampler publication requires a sampler payload")
            capture_id = uuid.uuid4().hex
            publication = await asyncio.to_thread(
                self._dispatch,
                self.command_group,
                "capture_sampler_snapshot",
                (model_id, capture_id, payload.publish_version),
            )
            return {
                "capture_id": capture_id,
                "publication": asdict(publication),
            }
        raise ValueError(f"{kind.value} has no persistence phase")

    async def persist_operation(
        self,
        model_id: str,
        kind: OperationKind,
        payload: OperationPayload,
        capture: object,
    ) -> object:
        if kind == OperationKind.SAVE_WEIGHTS:
            if not isinstance(payload, SaveWeightsPayload):
                raise TypeError("save_weights requires a checkpoint payload")
            if not isinstance(capture, Mapping):
                raise ValueError("snapshot handle must be an object")
            return await asyncio.to_thread(
                self._persist_checkpoint,
                model_id,
                payload,
                capture,
            )
        if kind == OperationKind.SAVE_WEIGHTS_FOR_SAMPLER:
            if not isinstance(capture, Mapping):
                raise ValueError("sampler capture handle must be an object")
            capture_id = capture.get("capture_id")
            publication = capture.get("publication")
            if not isinstance(capture_id, str) or not capture_id:
                raise ValueError("sampler capture handle has no capture_id")
            if not isinstance(publication, Mapping):
                raise ValueError("sampler capture handle has no publication")
            await asyncio.to_thread(
                self._dispatch,
                self.sampler_persistence_group,
                "persist_sampler_snapshot",
                (capture_id,),
            )
            return dict(publication)
        raise ValueError(f"{kind.value} has no persistence phase")

    async def unload_model(self, model_id: str) -> None:
        await asyncio.to_thread(
            self._dispatch,
            self.command_group,
            "unload_model",
            (model_id,),
        )

    async def close(self) -> None:
        await asyncio.to_thread(self._shutdown)

    def follow(self, group=None) -> None:
        checkpoint_persistence = None
        sampler_persistence = None
        if group is None:
            if (
                self.command_group is None
                or self.checkpoint_persistence_group is None
                or self.sampler_persistence_group is None
            ):
                raise RuntimeError("follow requires distributed command groups")
            checkpoint_persistence = threading.Thread(
                target=self.follow,
                args=(self.checkpoint_persistence_group,),
                name="tinker-checkpoint-persistence-follower",
            )
            sampler_persistence = threading.Thread(
                target=self.follow,
                args=(self.sampler_persistence_group,),
                name="tinker-sampler-persistence-follower",
            )
            checkpoint_persistence.start()
            sampler_persistence.start()
            group = self.command_group

        import torch.distributed as dist

        try:
            while True:
                holder = [None]
                dist.broadcast_object_list(holder, src=0, group=group)
                method, args, kwargs = holder[0]
                if method == "_stop":
                    return
                self._dispatch(group, method, args, kwargs)
                if method == "close":
                    return
        except BaseException:
            logging.getLogger(__name__).exception(
                "follower %s exited", threading.current_thread().name
            )
            raise
        finally:
            if checkpoint_persistence is not None:
                checkpoint_persistence.join()
            if sampler_persistence is not None:
                sampler_persistence.join()

    def _persist_checkpoint(
        self,
        model_id: str,
        payload: SaveWeightsPayload,
        snapshot: Mapping[str, Any],
    ) -> dict[str, str]:
        if snapshot.get("model_id") != model_id:
            raise ValueError("snapshot handle belongs to a different model")
        snapshot_id = snapshot.get("snapshot_id")
        destination = snapshot.get("destination")
        if not isinstance(snapshot_id, str) or not snapshot_id:
            raise ValueError("snapshot handle has no snapshot_id")
        if not isinstance(destination, str) or not destination:
            raise ValueError("snapshot handle has no destination")
        overwrite = payload.overwrite
        uri = self._dispatch(
            self.checkpoint_persistence_group,
            "persist_checkpoint",
            (snapshot_id, destination),
            {"overwrite": overwrite},
        )
        return {"path": uri, "type": "save_weights"}

    def _shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.checkpoint_persistence_group is not None:
            import torch.distributed as dist

            dist.broadcast_object_list(
                [("_stop", (), {})],
                src=0,
                group=self.checkpoint_persistence_group,
            )
        if self.sampler_persistence_group is not None:
            import torch.distributed as dist

            dist.broadcast_object_list(
                [("_stop", (), {})],
                src=0,
                group=self.sampler_persistence_group,
            )
        self._dispatch(self.command_group, "close", ())

    def _dispatch(
        self,
        group,
        method: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = kwargs or {}
        current_rank = 0
        if group is not None:
            import torch.distributed as dist

            current_rank = dist.get_rank(group=group)
        if group is not None and current_rank == 0:
            dist.broadcast_object_list(
                [(method, args, kwargs)],
                src=0,
                group=group,
            )

        result = None
        error: Exception | None = None
        try:
            result = getattr(self.backend, method)(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - gather backend errors across ranks
            error = exc

        if group is None or method == "close":
            if error is not None and current_rank == 0:
                raise error
            return result

        local_error = f"{type(error).__name__}: {error}" if error else None
        errors = [None] * dist.get_world_size(group=group)
        dist.all_gather_object(errors, local_error, group=group)
        failed_rank = next(
            (rank for rank, rank_error in enumerate(errors) if rank_error is not None),
            None,
        )
        if failed_rank is not None and current_rank == 0:
            raise RuntimeError(f"rank {failed_rank}: {errors[failed_rank]}")
        return result
