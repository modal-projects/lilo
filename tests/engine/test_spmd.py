import asyncio
import os
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest
from tinker import ForwardBackwardOutput, OptimStepResponse, TensorData

from lilo.backends import ForwardBatch, ModelSpec, SamplerPublication
from lilo.engine import DistributedExecutor, OperationKind
from lilo.engine.api import Execution
from lilo.engine.operations import parse_model_spec, parse_operation_payload
from lilo.engine.spmd import initialize_distributed_runtime


class RecordingBackend:
    def __init__(self) -> None:
        self.calls = []

    def accept_model(self, model_id, spec):
        self.calls.append(("accept", model_id, spec))

    def forward_backward(self, batch: ForwardBatch):
        self.calls.append(("forward_backward", batch))
        return tuple(
            ForwardBackwardOutput(
                loss_fn_output_type="TestLoss",
                loss_fn_outputs=[
                    {"logprobs": TensorData(data=[-1.0], dtype="float32")}
                ],
                metrics={"loss:mean": 1.0},
            )
            for _ in batch.items
        )

    def optim_step(self, model_ids, adam):
        self.calls.append(("optim_step", model_ids, adam))
        return tuple(
            OptimStepResponse(metrics={"grad_norm:mean": 0.5}) for _ in model_ids
        )

    def capture_checkpoint(
        self,
        model_id,
        snapshot_id,
        *,
        destination,
        include_optimizer,
    ):
        self.calls.append(
            (
                "capture_checkpoint",
                model_id,
                snapshot_id,
                destination,
                include_optimizer,
            )
        )

    def persist_checkpoint(self, snapshot_id, destination, *, overwrite=False):
        self.calls.append(("persist_checkpoint", snapshot_id, destination, overwrite))
        return f"/checkpoints/{destination}"

    def load_checkpoint(
        self,
        model_id,
        uri,
        *,
        restore_optimizer=False,
    ):
        self.calls.append(
            (
                "load_checkpoint",
                model_id,
                uri,
                restore_optimizer,
            )
        )

    def capture_sampler_snapshot(self, model_id, capture_id, requested_version):
        self.calls.append(
            ("capture_sampler_snapshot", model_id, capture_id, requested_version)
        )
        return SamplerPublication(
            publish_version=requested_version,
            base_model="Qwen/Qwen3-4B",
            optimizer_step=1,
        )

    def persist_sampler_snapshot(self, capture_id):
        self.calls.append(("persist_sampler_snapshot", capture_id))

    def unload_model(self, model_id):
        self.calls.append(("unload", model_id))

    def close(self):
        self.calls.append(("close",))


def test_distributed_executor_normalizes_and_dispatches_lifecycle() -> None:
    async def run() -> None:
        backend = RecordingBackend()
        executor = DistributedExecutor(backend)

        await executor.accept_model(
            "model-a",
            parse_model_spec(
                {
                    "base_model": "Qwen/Qwen3-4B",
                    "parameterization": {"type": "lora"},
                    "lora_config": {"rank": 16, "seed": 7, "alpha": 32},
                }
            ),
        )
        spec = backend.calls[-1][2]
        assert spec == ModelSpec(
            base_model="Qwen/Qwen3-4B",
            parameterization="lora",
            lora_config=spec.lora_config,
        )
        assert spec.lora_config.rank == 16
        assert spec.lora_config.seed == 7

        forward = await executor.execute(
            "model-a",
            OperationKind.FORWARD_BACKWARD,
            parse_operation_payload(
                OperationKind.FORWARD_BACKWARD,
                {
                    "data": [
                        {
                            "model_input": {"chunks": [{"tokens": [1, 2, 3]}]},
                            "loss_fn_inputs": {
                                "target_tokens": [2, 3, 4],
                                "weights": [1.0, 1.0, 1.0],
                            },
                        },
                    ],
                    "loss_fn": "cross_entropy",
                },
            ),
        )
        assert forward["loss_fn_output_type"] == "TestLoss"
        batch = backend.calls[-1][1]
        assert batch.items[0].data[0].model_input.to_ints() == [1, 2, 3]
        assert batch.forward_only is False

        optim = await executor.execute(
            "model-a",
            OperationKind.OPTIM_STEP,
            parse_operation_payload(
                OperationKind.OPTIM_STEP,
                {"adam_params": {"learning_rate": 1e-4}},
            ),
        )
        assert optim == {"metrics": {"grad_norm:mean": 0.5}}

        checkpoint_payload = parse_operation_payload(
            OperationKind.SAVE_WEIGHTS,
            {"name": "smoke"},
        )
        checkpoint = await executor.capture_operation(
            "model-a",
            OperationKind.SAVE_WEIGHTS,
            checkpoint_payload,
        )
        saved = await executor.persist_operation(
            "model-a",
            OperationKind.SAVE_WEIGHTS,
            checkpoint_payload,
            checkpoint,
        )
        assert saved == {"path": "/checkpoints/smoke", "type": "save_weights"}
        captured, persisted = backend.calls[-2:]
        assert captured[:2] == ("capture_checkpoint", "model-a")
        assert captured[3:] == ("smoke", True)
        assert persisted == (
            "persist_checkpoint",
            captured[2],
            "smoke",
            False,
        )

        loaded = await executor.execute(
            "model-a",
            OperationKind.LOAD_WEIGHTS,
            parse_operation_payload(
                OperationKind.LOAD_WEIGHTS,
                {"path": saved["path"], "optimizer": True},
            ),
        )
        assert loaded == {"path": "/checkpoints/smoke", "type": "load_weights"}
        assert backend.calls[-1] == (
            "load_checkpoint",
            "model-a",
            "/checkpoints/smoke",
            True,
        )

        sampler_payload = parse_operation_payload(
            OperationKind.SAVE_WEIGHTS_FOR_SAMPLER,
            {"publish_version": 9},
        )
        capture = await executor.capture_operation(
            "model-a",
            OperationKind.SAVE_WEIGHTS_FOR_SAMPLER,
            sampler_payload,
        )
        published = await executor.persist_operation(
            "model-a",
            OperationKind.SAVE_WEIGHTS_FOR_SAMPLER,
            sampler_payload,
            capture,
        )
        assert published["publish_version"] == 9

        await executor.unload_model("model-a")
        await executor.close()
        await executor.close()
        assert backend.calls[-2:] == [("unload", "model-a"), ("close",)]

    asyncio.run(run())


def test_standard_checkpoint_parser_preserves_optimizer_semantics() -> None:
    loaded = parse_operation_payload(
        OperationKind.LOAD_WEIGHTS,
        {"path": "/checkpoint", "optimizer": True},
    )
    assert loaded.restore_optimizer is True

    saved = parse_operation_payload(OperationKind.SAVE_WEIGHTS, {})
    assert saved.include_optimizer is True


def test_model_spec_rejects_mismatched_parameterization() -> None:
    with pytest.raises(ValueError, match="lora_config must be absent"):
        parse_model_spec(
            {
                "base_model": "Qwen/Qwen3-4B",
                "parameterization": {"type": "full"},
                "lora_config": {"rank": 8},
            }
        )


def test_distributed_executor_batches_compatible_forward_backward() -> None:
    async def run() -> None:
        backend = RecordingBackend()
        executor = DistributedExecutor(backend)
        payload = parse_operation_payload(
            OperationKind.FORWARD_BACKWARD,
            {
                "data": [
                    {
                        "model_input": {"chunks": [{"tokens": [1, 2]}]},
                        "loss_fn_inputs": {
                            "target_tokens": [2, 3],
                            "weights": [1.0, 1.0],
                        },
                    },
                ],
                "loss_fn": "cross_entropy",
            },
        )

        outputs = await executor.execute_batch(
            (
                Execution("model-a", OperationKind.FORWARD_BACKWARD, payload),
                Execution("model-b", OperationKind.FORWARD_BACKWARD, payload),
                Execution("model-a", OperationKind.FORWARD_BACKWARD, payload),
            )
        )

        batch = backend.calls[-1][1]
        assert [item.model_id for item in batch.items] == [
            "model-a",
            "model-b",
            "model-a",
        ]
        assert len(outputs) == 3

    asyncio.run(run())


def test_checkpoint_handles_are_unique_across_models() -> None:
    async def run() -> None:
        executor = DistributedExecutor(RecordingBackend())

        first = await executor.capture_operation(
            "model-a",
            OperationKind.SAVE_WEIGHTS,
            parse_operation_payload(
                OperationKind.SAVE_WEIGHTS,
                {"name": "step-1"},
            ),
        )
        second = await executor.capture_operation(
            "model-b",
            OperationKind.SAVE_WEIGHTS,
            parse_operation_payload(
                OperationKind.SAVE_WEIGHTS,
                {"name": "step-1"},
            ),
        )

        assert first["snapshot_id"] != second["snapshot_id"]
        assert first["destination"] == second["destination"] == "step-1"

    asyncio.run(run())


def test_checkpoint_persist_failure_is_returned() -> None:
    class ExistingCheckpointBackend(RecordingBackend):
        def persist_checkpoint(self, snapshot_id, destination, *, overwrite=False):
            self.calls.append(
                ("persist_checkpoint", snapshot_id, destination, overwrite)
            )
            raise FileExistsError(destination)

    async def run() -> None:
        backend = ExistingCheckpointBackend()
        executor = DistributedExecutor(backend)
        payload = parse_operation_payload(
            OperationKind.SAVE_WEIGHTS,
            {"name": "step-1"},
        )
        snapshot = await executor.capture_operation(
            "model-a",
            OperationKind.SAVE_WEIGHTS,
            payload,
        )

        with pytest.raises(FileExistsError, match="step-1"):
            await executor.persist_operation(
                "model-a",
                OperationKind.SAVE_WEIGHTS,
                payload,
                snapshot,
            )

        assert backend.calls[-1][0] == "persist_checkpoint"

    asyncio.run(run())


def test_initializes_distributed_runtime_from_environment(monkeypatch) -> None:
    calls = []
    distributed = ModuleType("torch.distributed")
    distributed.init_process_group = lambda **kwargs: calls.append(("init", kwargs))
    distributed.new_group = lambda **kwargs: calls.append(("group", kwargs)) or "gloo"
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(set_device=lambda rank: calls.append(("device", rank)))
    torch.distributed = distributed
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", distributed)
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "2")

    (
        command_group,
        checkpoint_persistence_group,
        sampler_persistence_group,
    ) = initialize_distributed_runtime()

    assert command_group == "gloo"
    assert checkpoint_persistence_group == "gloo"
    assert sampler_persistence_group == "gloo"
    assert calls[0] == ("device", 3)
    assert calls[1][0] == "init"
    assert calls[2:] == [
        ("group", {"backend": "gloo"}),
        ("group", {"backend": "gloo"}),
        ("group", {"backend": "gloo"}),
    ]


def test_persistence_operations_use_separate_groups(monkeypatch) -> None:
    calls = []
    executor = DistributedExecutor(
        RecordingBackend(),
        checkpoint_persistence_group="checkpoints",
        sampler_persistence_group="samplers",
    )

    def dispatch(group, method, args, kwargs=None):
        calls.append((group, method))
        return "/checkpoints/step-1" if method == "persist_checkpoint" else None

    monkeypatch.setattr(executor, "_dispatch", dispatch)

    async def run() -> None:
        checkpoint_payload = parse_operation_payload(
            OperationKind.SAVE_WEIGHTS,
            {"name": "step-1"},
        )
        await executor.persist_operation(
            "model-a",
            OperationKind.SAVE_WEIGHTS,
            checkpoint_payload,
            {
                "model_id": "model-a",
                "snapshot_id": "snapshot-1",
                "destination": "step-1",
            },
        )
        sampler_payload = parse_operation_payload(
            OperationKind.SAVE_WEIGHTS_FOR_SAMPLER,
            {"publish_version": 1},
        )
        await executor.persist_operation(
            "model-a",
            OperationKind.SAVE_WEIGHTS_FOR_SAMPLER,
            sampler_payload,
            {
                "capture_id": "capture-1",
                "publication": {"publish_version": 1},
            },
        )

    asyncio.run(run())

    assert calls == [
        ("checkpoints", "persist_checkpoint"),
        ("samplers", "persist_sampler_snapshot"),
    ]


def test_initializes_single_process_runtime_without_launcher_env(
    monkeypatch,
) -> None:
    calls = []
    distributed = ModuleType("torch.distributed")
    distributed.init_process_group = lambda **kwargs: calls.append(kwargs)
    distributed.new_group = lambda **kwargs: pytest.fail("single rank needs no group")
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(set_device=lambda rank: calls.append(rank))
    torch.distributed = distributed
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", distributed)
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(name, raising=False)

    assert initialize_distributed_runtime() == (None, None, None)

    assert calls == [0, {"backend": "nccl", "world_size": 1, "rank": 0}]
    assert os.environ["LOCAL_RANK"] == "0"
    assert os.environ["MASTER_ADDR"] == "127.0.0.1"
    assert os.environ["MASTER_PORT"] == "29541"


def test_distributed_dispatch_reports_remote_rank_errors(monkeypatch) -> None:
    calls = []
    rank = [0]
    distributed = ModuleType("torch.distributed")
    distributed.broadcast_object_list = lambda value, **kwargs: calls.append(
        ("broadcast", value, kwargs)
    )
    distributed.get_rank = lambda **kwargs: rank[0]
    distributed.get_world_size = lambda **kwargs: 2

    def gather(errors, local_error, **kwargs):
        calls.append(("gather", local_error, kwargs))
        errors[:] = [local_error, "ValueError: remote boom"]

    distributed.all_gather_object = gather
    torch = ModuleType("torch")
    torch.distributed = distributed
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", distributed)
    executor = DistributedExecutor(RecordingBackend(), command_group="commands")

    with pytest.raises(RuntimeError, match="rank 1: ValueError: remote boom"):
        executor._dispatch("commands", "unload_model", ("model-a",))

    assert calls[0] == (
        "broadcast",
        [("unload_model", ("model-a",), {})],
        {"src": 0, "group": "commands"},
    )
    assert calls[1] == ("gather", None, {"group": "commands"})

    rank[0] = 1
    calls.clear()
    executor._dispatch("commands", "unload_model", ("model-a",))
    assert calls == [("gather", None, {"group": "commands"})]


def test_persistence_does_not_block_command_lane() -> None:
    async def run() -> None:
        started = threading.Event()
        release = threading.Event()

        class SlowPersistenceBackend(RecordingBackend):
            def persist_checkpoint(self, snapshot_id, destination, *, overwrite=False):
                started.set()
                if not release.wait(1.0):
                    raise TimeoutError("test did not release persistence")
                return super().persist_checkpoint(
                    snapshot_id,
                    destination,
                    overwrite=overwrite,
                )

        executor = DistributedExecutor(SlowPersistenceBackend())
        checkpoint_payload = parse_operation_payload(
            OperationKind.SAVE_WEIGHTS,
            {"name": "snapshot-1"},
        )
        snapshot = await executor.capture_operation(
            "model-a",
            OperationKind.SAVE_WEIGHTS,
            checkpoint_payload,
        )
        persisting = asyncio.create_task(
            executor.persist_operation(
                "model-a",
                OperationKind.SAVE_WEIGHTS,
                checkpoint_payload,
                snapshot,
            )
        )
        assert await asyncio.to_thread(started.wait, 1.0)

        forward = await asyncio.wait_for(
            executor.execute(
                "model-a",
                OperationKind.FORWARD,
                parse_operation_payload(
                    OperationKind.FORWARD,
                    {
                        "data": [
                            {
                                "model_input": {"chunks": [{"tokens": [1]}]},
                                "loss_fn_inputs": {
                                    "target_tokens": [1],
                                    "weights": [1.0],
                                },
                            },
                        ],
                        "loss_fn": "cross_entropy",
                    },
                ),
            ),
            0.5,
        )
        assert forward["loss_fn_output_type"] == "TestLoss"

        release.set()
        assert (await persisting)["path"] == "/checkpoints/snapshot-1"

    asyncio.run(run())


def test_parse_forward_backward_accepts_lilo_loss_fns() -> None:
    datum = {
        "model_input": {"chunks": [{"tokens": [1, 2]}]},
        "loss_fn_inputs": {
            "target_tokens": [2, 3],
            "logprobs": [-0.5, -0.5],
            "advantages": [1.0, -1.0],
        },
    }
    payload = parse_operation_payload(
        OperationKind.FORWARD_BACKWARD,
        {"data": [datum], "loss_fn": "dppo", "loss_fn_config": {"tv_threshold": 0.2}},
    )
    assert payload.loss_fn == "dppo"
    assert payload.loss_fn_config == {"tv_threshold": 0.2}
    with pytest.raises(ValueError, match="loss_fn"):
        parse_operation_payload(
            OperationKind.FORWARD_BACKWARD,
            {"data": [datum], "loss_fn": "nope"},
        )
