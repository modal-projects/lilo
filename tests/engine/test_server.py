import asyncio
import json

import pytest

from tests.support import EchoExecutor
from lilo.engine import EngineServer, FutureStatus, OperationKind
from lilo.errors import EngineSaturated, RecordNotFound, SequenceConflict


async def forward_backward(
    server: EngineServer,
    seq_id: int,
    data: object = None,
    model_id: str = "model-a",
) -> str:
    tokens = list(data if data is not None else [seq_id])
    body = json.dumps(
        {
            "model_id": model_id,
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
    return await server.forward_backward(body, "application/json")


def test_capacity_and_idempotent_accept() -> None:
    async def run() -> None:
        server = EngineServer(EchoExecutor(), max_models=1)
        assert await server.accept_model("model-a", {"rank": 8})
        assert await server.model_ids() == ("model-a",)
        assert await server.accept_model("model-a", {"rank": 8})
        assert not await server.accept_model("model-b", {"rank": 8})
        await server.unload_model("model-a")
        assert await server.model_ids() == ()

    asyncio.run(run())


def test_accept_loads_checkpoint_before_model_is_ready() -> None:
    async def run() -> None:
        calls = []

        class RecordingExecutor(EchoExecutor):
            async def accept_model(self, model_id, spec):
                calls.append(("accept", model_id))

            async def execute(self, model_id, kind, payload):
                calls.append((kind.value, model_id, payload))
                return await super().execute(model_id, kind, payload)

        server = EngineServer(RecordingExecutor())
        accepted = await server.accept_model(
            "session:train:0",
            {
                "checkpoint": {
                    "uri": "/checkpoints/snapshot",
                    "restore_optimizer": True,
                }
            },
        )

        assert accepted
        assert calls[0] == ("accept", "session:train:0")
        assert calls[1][0:2] == ("load_weights", "session:train:0")
        assert calls[1][2].uri == "/checkpoints/snapshot"
        assert calls[1][2].restore_optimizer
        request_id = await server.optim_step(
            {"model_id": "session:train:0", "seq_id": 1, "adam_params": {}}
        )
        assert (await server.retrieve_future(request_id, timeout=1)).status == (
            FutureStatus.COMPLETE
        )

    asyncio.run(run())


def test_accept_reports_checkpoint_load_failure() -> None:
    async def run() -> None:
        unloaded = []

        class FailingExecutor(EchoExecutor):
            async def execute(self, model_id, kind, payload):
                if kind == OperationKind.LOAD_WEIGHTS:
                    raise FileNotFoundError(payload.uri)
                return await super().execute(model_id, kind, payload)

            async def unload_model(self, model_id):
                unloaded.append(model_id)

        server = EngineServer(FailingExecutor())
        with pytest.raises(ValueError, match="accept model: /missing"):
            await server.accept_model(
                "session:train:0",
                {"checkpoint": {"uri": "/missing"}},
            )
        assert await server.model_ids() == ()
        assert unloaded == ["session:train:0"]

    asyncio.run(run())


def test_executes_in_order_across_gaps() -> None:
    async def run() -> None:
        server = EngineServer(EchoExecutor())
        await server.accept_model("model-a", {})
        assert await forward_backward(server, 2) == "model-a:2"
        assert (
            await server.retrieve_future("model-a:2", timeout=0.05)
        ).status == FutureStatus.PENDING
        await forward_backward(server, 1)
        assert (
            await server.retrieve_future("model-a:1", timeout=1.0)
        ).status == FutureStatus.COMPLETE
        assert (
            await server.retrieve_future("model-a:2", timeout=1.0)
        ).status == FutureStatus.COMPLETE
        assert await server.retrieve_future("model-a:3") is None
        await server.close()

    asyncio.run(run())


def test_skip_sequence_advances_across_rejected_operation() -> None:
    async def run() -> None:
        server = EngineServer(EchoExecutor())
        await server.accept_model("model-a", {})
        await forward_backward(server, 2)
        request_id = await server.skip_sequence("model-a", 1, "invalid request")
        assert request_id == "model-a:1"
        skipped = await server.retrieve_future(request_id, timeout=1.0)
        assert skipped.status == FutureStatus.FAILED
        assert skipped.error == "invalid request"
        assert (
            await server.retrieve_future("model-a:2", timeout=1.0)
        ).status == FutureStatus.COMPLETE
        await server.close()

    asyncio.run(run())


def test_retrieve_blocks_until_completion() -> None:
    async def run() -> None:
        server = EngineServer(EchoExecutor())
        await server.accept_model("model-a", {})
        await forward_backward(server, 2)

        waiter = asyncio.create_task(server.retrieve_future("model-a:2", timeout=5.0))
        await asyncio.sleep(0.01)
        await forward_backward(server, 1)
        assert (await waiter).status == FutureStatus.COMPLETE
        await server.close()

    asyncio.run(run())


def test_deduplicates_and_detects_conflicts() -> None:
    async def run() -> None:
        server = EngineServer(EchoExecutor())
        await server.accept_model("model-a", {})
        first = await forward_backward(server, 1)
        assert await forward_backward(server, 1) == first
        assert (
            await server.retrieve_future("model-a:1", timeout=1.0)
        ).status == FutureStatus.COMPLETE
        with pytest.raises(SequenceConflict):
            await forward_backward(server, 1, data=[999])
        with pytest.raises(RecordNotFound):
            await forward_backward(server, 1, model_id="model-b")
        with pytest.raises(ValueError):
            await server.optim_step({"adam_params": {}})
        await server.close()

    asyncio.run(run())


def test_forward_flattens_input_envelope() -> None:
    async def run() -> None:
        server = EngineServer(EchoExecutor())
        await server.accept_model("model-a", {})
        request_id = await server.forward(
            {
                "model_id": "model-a",
                "seq_id": 1,
                "forward_input": {
                    "data": [
                        {
                            "model_input": {"chunks": [{"tokens": [1]}]},
                            "loss_fn_inputs": {},
                        }
                    ],
                    "loss_fn": "cross_entropy",
                },
            }
        )
        state = await server.retrieve_future(request_id, timeout=1.0)
        assert state.status == FutureStatus.COMPLETE
        assert state.result == {
            "model_id": "model-a",
            "kind": "forward",
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
        await server.close()

    asyncio.run(run())


def test_operations_share_one_sequence_per_model() -> None:
    async def run() -> None:
        server = EngineServer(EchoExecutor())
        await server.accept_model("model-a", {})
        await forward_backward(server, 1)
        request_id = await server.optim_step(
            {"model_id": "model-a", "seq_id": 2, "adam_params": {"learning_rate": 1e-4}}
        )
        state = await server.retrieve_future(request_id, timeout=1.0)
        assert state.status == FutureStatus.COMPLETE
        assert state.result == {
            "model_id": "model-a",
            "kind": "optim_step",
            "payload": {"adam_params": {}},
        }
        await server.close()

    asyncio.run(run())


@pytest.mark.parametrize("name", ("", "../other", "/tmp/checkpoint", "nested/name"))
def test_save_weights_rejects_paths_outside_model_directory(name: str) -> None:
    async def run() -> None:
        server = EngineServer(EchoExecutor())
        await server.accept_model("model-a", {})
        with pytest.raises(ValueError, match="single path component"):
            await server.save_weights(
                {"model_id": "model-a", "seq_id": 1, "path": name}
            )

    asyncio.run(run())


def test_draining_refuses_new_work() -> None:
    async def run() -> None:
        server = EngineServer(EchoExecutor())
        await server.accept_model("model-a", {})
        server.draining = True
        assert not await server.accept_model("model-b", {})
        with pytest.raises(EngineSaturated):
            await forward_backward(server, 1)

    asyncio.run(run())


def test_unload_drops_model_futures() -> None:
    async def run() -> None:
        server = EngineServer(EchoExecutor())
        await server.accept_model("model-a", {})
        await forward_backward(server, 2)
        await server.unload_model("model-a")
        assert await server.retrieve_future("model-a:2") is None
        await server.close()

    asyncio.run(run())


def test_unload_runs_after_active_operation_and_drops_queued_work() -> None:
    async def run() -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        calls = []

        class BlockingExecutor(EchoExecutor):
            async def execute(self, model_id, kind, payload):
                calls.append(("execute", payload.data[0].model_input.to_ints()))
                started.set()
                await release.wait()
                return await super().execute(model_id, kind, payload)

            async def unload_model(self, model_id):
                calls.append(("unload", model_id))

        server = EngineServer(BlockingExecutor())
        await server.accept_model("model-a", {})
        await forward_backward(server, 1)
        await started.wait()
        await forward_backward(server, 2)

        unloading = asyncio.create_task(server.unload_model("model-a"))
        await asyncio.sleep(0)
        with pytest.raises(RecordNotFound):
            await forward_backward(server, 3)

        release.set()
        await unloading

        assert calls == [("execute", [1]), ("unload", "model-a")]
        assert await server.retrieve_future("model-a:1") is None
        assert await server.retrieve_future("model-a:2") is None
        assert await server.model_ids() == ()
        await server.close()

    asyncio.run(run())


def test_executor_errors_fail_the_future() -> None:
    async def run() -> None:
        class FailingExecutor(EchoExecutor):
            async def execute(self, model_id, kind, payload):
                raise RuntimeError("boom")

        server = EngineServer(FailingExecutor())
        await server.accept_model("model-a", {})
        await forward_backward(server, 1)
        state = await server.retrieve_future("model-a:1", timeout=1.0)
        assert state.status == FutureStatus.FAILED
        assert state.error == "boom"
        await server.close()

    asyncio.run(run())


def test_oldest_results_evicted_beyond_cap() -> None:
    async def run() -> None:
        server = EngineServer(EchoExecutor(), max_results=1)
        await server.accept_model("model-a", {})
        await server.accept_model("model-b", {})
        await forward_backward(server, 1, model_id="model-b")
        await forward_backward(server, 1)
        assert (
            await server.retrieve_future("model-a:1", timeout=1.0)
        ).status == FutureStatus.COMPLETE
        await forward_backward(server, 2)
        assert (
            await server.retrieve_future("model-a:2", timeout=1.0)
        ).status == FutureStatus.COMPLETE
        assert await server.retrieve_future("model-a:1") is None
        with pytest.raises(SequenceConflict):
            await forward_backward(server, 1)
        assert (
            await server.retrieve_future("model-b:1", timeout=1.0)
        ).status == FutureStatus.COMPLETE
        await server.close()

    asyncio.run(run())


def test_duplicate_accept_waits_for_registration() -> None:
    async def run() -> None:
        release = asyncio.Event()

        class SlowAcceptExecutor(EchoExecutor):
            registered = False

            async def accept_model(self, model_id: str, spec: object) -> None:
                await release.wait()
                self.registered = True

        executor = SlowAcceptExecutor()
        server = EngineServer(executor)
        first = asyncio.create_task(server.accept_model("model-a", {}))
        await asyncio.sleep(0.01)
        second = asyncio.create_task(server.accept_model("model-a", {}))
        await asyncio.sleep(0.01)
        assert not second.done()
        release.set()
        assert await first
        assert await second
        assert executor.registered

    asyncio.run(run())


def test_accept_runs_after_active_operation() -> None:
    async def run() -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        calls = []

        class BlockingExecutor(EchoExecutor):
            async def accept_model(self, model_id, spec):
                calls.append(("accept", model_id))

            async def execute(self, model_id, kind, payload):
                calls.append(("execute", model_id))
                started.set()
                await release.wait()
                return await super().execute(model_id, kind, payload)

        server = EngineServer(BlockingExecutor())
        await server.accept_model("model-a", {})
        calls.clear()
        await forward_backward(server, 1)
        await started.wait()

        accepting = asyncio.create_task(server.accept_model("model-b", {}))
        await asyncio.sleep(0)
        assert calls == [("execute", "model-a")]

        release.set()
        assert await accepting
        assert calls == [("execute", "model-a"), ("accept", "model-b")]
        await server.close()

    asyncio.run(run())


def test_builder_interleaves_models() -> None:
    async def run() -> None:
        server = EngineServer(EchoExecutor())
        await server.accept_model("model-a", {})
        await server.accept_model("model-b", {})
        await forward_backward(server, 1, model_id="model-a")
        await forward_backward(server, 1, model_id="model-b")
        for request_id in ("model-a:1", "model-b:1"):
            state = await server.retrieve_future(request_id, timeout=1.0)
            assert state.status == FutureStatus.COMPLETE
        await server.close()

    asyncio.run(run())


def test_batches_compatible_forward_backward_across_models() -> None:
    async def run() -> None:
        class RecordingExecutor(EchoExecutor):
            def __init__(self) -> None:
                self.batches = []

            async def execute_batch(self, executions):
                self.batches.append(executions)
                return await super().execute_batch(executions)

        executor = RecordingExecutor()
        server = EngineServer(executor)
        await server.accept_model("model-a", {})
        await server.accept_model("model-b", {})
        await asyncio.gather(
            forward_backward(server, 1, model_id="model-a"),
            forward_backward(server, 1, model_id="model-b"),
        )

        for request_id in ("model-a:1", "model-b:1"):
            state = await server.retrieve_future(request_id, timeout=1.0)
            assert state.status == FutureStatus.COMPLETE
        assert [[item.model_id for item in batch] for batch in executor.batches] == [
            ["model-a", "model-b"]
        ]
        await server.close()

    asyncio.run(run())


def test_batches_consecutive_forward_backward_for_one_model() -> None:
    async def run() -> None:
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        batches = []

        class RecordingExecutor(EchoExecutor):
            async def execute_batch(self, executions):
                batches.append(
                    [
                        (item.model_id, *item.payload.data[0].model_input.to_ints())
                        for item in executions
                    ]
                )
                if len(batches) == 1:
                    first_started.set()
                    await release_first.wait()
                return await super().execute_batch(executions)

        server = EngineServer(RecordingExecutor())
        await server.accept_model("model-a", {})
        await server.accept_model("model-b", {})

        await forward_backward(server, 1)
        await first_started.wait()
        await forward_backward(server, 2)
        await forward_backward(server, 3)
        await server.optim_step(
            {"model_id": "model-a", "seq_id": 4, "adam_params": {"learning_rate": 0.1}}
        )
        await forward_backward(server, 5)
        await forward_backward(server, 1, model_id="model-b")
        await forward_backward(server, 2, model_id="model-b")

        release_first.set()
        for request_id in ("model-a:5", "model-b:2"):
            state = await server.retrieve_future(request_id, timeout=1.0)
            assert state.status == FutureStatus.COMPLETE
        assert batches == [
            [("model-a", 1)],
            [("model-a", 2), ("model-b", 1), ("model-a", 3), ("model-b", 2)],
            [("model-a", 5)],
        ]
        await server.close()

    asyncio.run(run())


def test_batches_forward_backward_buffered_during_previous_batch() -> None:
    async def run() -> None:
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        second_started = asyncio.Event()
        batches = []

        class RecordingExecutor(EchoExecutor):
            async def execute_batch(self, executions):
                batches.append([item.model_id for item in executions])
                if len(batches) == 1:
                    first_started.set()
                    await release_first.wait()
                else:
                    second_started.set()
                return await super().execute_batch(executions)

        server = EngineServer(RecordingExecutor())
        for model_id in ("model-a", "model-b", "model-c"):
            await server.accept_model(model_id, {})

        await forward_backward(server, 1, model_id="model-a")
        await first_started.wait()
        await asyncio.gather(
            forward_backward(server, 1, model_id="model-b"),
            forward_backward(server, 1, model_id="model-c"),
        )

        release_first.set()
        await asyncio.wait_for(second_started.wait(), timeout=1.0)
        assert batches == [["model-a"], ["model-b", "model-c"]]
        await server.close()

    asyncio.run(run())


def test_close_waits_for_active_capture_and_persistence() -> None:
    async def run() -> None:
        capture_started = asyncio.Event()
        release_capture = asyncio.Event()
        calls = []

        class CheckpointExecutor(EchoExecutor):
            async def capture_checkpoint(self, model_id, payload):
                calls.append("capture")
                capture_started.set()
                await release_capture.wait()
                return {"snapshot_id": "snapshot-1"}

            async def persist_checkpoint(self, model_id, payload, snapshot):
                calls.append("persist")
                return {"path": "/checkpoint", "type": "save_weights"}

        server = EngineServer(CheckpointExecutor())
        await server.accept_model("model-a", {})
        await server.save_weights(
            {"model_id": "model-a", "seq_id": 1, "name": "snapshot-1"}
        )
        await capture_started.wait()

        closing = asyncio.create_task(server.close())
        await asyncio.sleep(0)
        assert not closing.done()
        release_capture.set()
        await closing

        assert calls == ["capture", "persist"]

    asyncio.run(run())


def test_checkpoint_persistence_overlaps_later_gpu_operations() -> None:
    async def run() -> None:
        persist_started = asyncio.Event()
        release_persist = asyncio.Event()
        executed = []

        class CheckpointExecutor(EchoExecutor):
            async def capture_checkpoint(self, model_id, payload):
                executed.append(("capture", model_id))
                return {"snapshot_id": "snapshot-1"}

            async def persist_checkpoint(self, model_id, payload, snapshot):
                executed.append(("persist", model_id))
                persist_started.set()
                await release_persist.wait()
                return {"path": "/checkpoints/snapshot-1", "type": "save_weights"}

            async def execute(self, model_id, kind, payload):
                executed.append((kind.value, model_id))
                return await super().execute(model_id, kind, payload)

        server = EngineServer(CheckpointExecutor())
        await server.accept_model("model-a", {})
        save_id = await server.save_weights(
            {"model_id": "model-a", "seq_id": 1, "name": "snapshot-1"}
        )
        forward_id = await forward_backward(server, 2)

        await asyncio.wait_for(persist_started.wait(), 1.0)
        forward = await server.retrieve_future(forward_id, timeout=1.0)
        assert forward.status == FutureStatus.COMPLETE
        assert (await server.retrieve_future(save_id)).status == FutureStatus.PENDING
        assert executed[0] == ("capture", "model-a")
        assert set(executed[1:3]) == {
            ("persist", "model-a"),
            ("forward_backward", "model-a"),
        }

        release_persist.set()
        saved = await server.retrieve_future(save_id, timeout=1.0)
        assert saved.status == FutureStatus.COMPLETE
        assert saved.result == {
            "path": "/checkpoints/snapshot-1",
            "type": "save_weights",
        }
        await server.close()

    asyncio.run(run())


def test_checkpoint_persistence_does_not_block_sampler_publication() -> None:
    async def run() -> None:
        checkpoint_started = asyncio.Event()
        release_checkpoint = asyncio.Event()
        sampler_completed = asyncio.Event()

        class SplitPersistenceExecutor(EchoExecutor):
            async def persist_operation(
                self,
                model_id,
                kind,
                payload,
                capture,
            ):
                if kind == OperationKind.SAVE_WEIGHTS:
                    checkpoint_started.set()
                    await release_checkpoint.wait()
                    return {
                        "path": "/checkpoints/snapshot-1",
                        "type": "save_weights",
                    }
                sampler_completed.set()
                return capture["publication"]

        server = EngineServer(SplitPersistenceExecutor())
        await server.accept_model("model-a", {})
        checkpoint_id = await server.save_weights(
            {"model_id": "model-a", "seq_id": 1, "name": "snapshot-1"}
        )
        sampler_id = await server.save_weights_for_sampler(
            {
                "model_id": "model-a",
                "seq_id": 2,
                "publish_version": 1,
            }
        )

        await asyncio.wait_for(checkpoint_started.wait(), timeout=1.0)
        await asyncio.wait_for(sampler_completed.wait(), timeout=1.0)
        assert (await server.retrieve_future(checkpoint_id)).status == (
            FutureStatus.PENDING
        )
        assert (await server.retrieve_future(sampler_id)).status == (
            FutureStatus.COMPLETE
        )

        release_checkpoint.set()
        await server.close()

    asyncio.run(run())


def test_next_capture_waits_for_previous_persistence() -> None:
    async def run() -> None:
        persist_started = asyncio.Event()
        release_persist = asyncio.Event()
        captures = []

        class CheckpointExecutor(EchoExecutor):
            async def capture_checkpoint(self, model_id, payload):
                captures.append(payload.destination)
                return {"snapshot_id": payload.destination}

            async def persist_checkpoint(self, model_id, payload, snapshot):
                if payload.destination == "first":
                    persist_started.set()
                    await release_persist.wait()
                return {"path": payload.destination, "type": "save_weights"}

        server = EngineServer(CheckpointExecutor())
        await server.accept_model("model-a", {})
        first = await server.save_weights(
            {"model_id": "model-a", "seq_id": 1, "name": "first"}
        )
        second = await server.save_weights(
            {"model_id": "model-a", "seq_id": 2, "name": "second"}
        )
        await persist_started.wait()
        await asyncio.sleep(0.01)
        assert captures == ["first"]

        release_persist.set()
        assert (await server.retrieve_future(first, timeout=1)).status == (
            FutureStatus.COMPLETE
        )
        assert (await server.retrieve_future(second, timeout=1)).status == (
            FutureStatus.COMPLETE
        )
        assert captures == ["first", "second"]
        await server.close()

    asyncio.run(run())


def test_load_waits_for_pending_persistence() -> None:
    async def run() -> None:
        persist_started = asyncio.Event()
        release_persist = asyncio.Event()
        executed = []

        class CheckpointExecutor(EchoExecutor):
            async def capture_checkpoint(self, model_id, payload):
                executed.append(("capture", model_id))
                return {"snapshot_id": "snapshot-1"}

            async def persist_checkpoint(self, model_id, payload, snapshot):
                executed.append(("persist", model_id))
                persist_started.set()
                await release_persist.wait()
                return {"path": "/checkpoints/snapshot-1", "type": "save_weights"}

            async def execute(self, model_id, kind, payload):
                executed.append((kind.value, model_id))
                return await super().execute(model_id, kind, payload)

        server = EngineServer(CheckpointExecutor())
        await server.accept_model("model-a", {})
        await server.save_weights(
            {"model_id": "model-a", "seq_id": 1, "name": "snapshot-1"}
        )
        load_id = await server.load_weights(
            {
                "model_id": "model-a",
                "seq_id": 2,
                "path": "/checkpoints/previous",
            }
        )

        await asyncio.wait_for(persist_started.wait(), 1.0)
        assert (await server.retrieve_future(load_id)).status == FutureStatus.PENDING
        assert executed == [("capture", "model-a"), ("persist", "model-a")]

        release_persist.set()
        loaded = await server.retrieve_future(load_id, timeout=1.0)
        assert loaded.status == FutureStatus.COMPLETE
        assert executed[-1] == ("load_weights", "model-a")
        await server.close()

    asyncio.run(run())


def test_unload_waits_for_pending_persistence() -> None:
    async def run() -> None:
        persist_started = asyncio.Event()
        release_persist = asyncio.Event()
        executed = []

        class CheckpointExecutor(EchoExecutor):
            async def capture_checkpoint(self, model_id, payload):
                executed.append(("capture", model_id))
                return {"snapshot_id": "snapshot-1"}

            async def persist_checkpoint(self, model_id, payload, snapshot):
                executed.append(("persist", model_id))
                persist_started.set()
                await release_persist.wait()
                return {"path": "/checkpoints/snapshot-1", "type": "save_weights"}

            async def unload_model(self, model_id):
                executed.append(("unload", model_id))

        server = EngineServer(CheckpointExecutor())
        await server.accept_model("model-a", {})
        await server.save_weights(
            {"model_id": "model-a", "seq_id": 1, "name": "snapshot-1"}
        )
        await asyncio.wait_for(persist_started.wait(), 1.0)

        unloading = asyncio.create_task(server.unload_model("model-a"))
        await asyncio.sleep(0.05)
        assert not unloading.done()
        assert executed == [("capture", "model-a"), ("persist", "model-a")]

        release_persist.set()
        await unloading
        assert executed[-1] == ("unload", "model-a")
        await server.close()

    asyncio.run(run())


def test_sampler_persistence_overlaps_later_gpu_operations() -> None:
    async def run() -> None:
        persist_started = asyncio.Event()
        release_persist = asyncio.Event()
        executed = []

        class SamplerExecutor(EchoExecutor):
            async def capture_operation(self, model_id, kind, payload):
                assert kind.value == "save_weights_for_sampler"
                executed.append(("capture_sampler", model_id))
                return {"capture_id": "sampler-1", "publish_version": 1}

            async def persist_operation(
                self,
                model_id,
                kind,
                payload,
                capture,
            ):
                assert kind.value == "save_weights_for_sampler"
                executed.append(("persist_sampler", model_id))
                persist_started.set()
                await release_persist.wait()
                return {"publish_version": capture["publish_version"]}

            async def execute(self, model_id, kind, payload):
                executed.append((kind.value, model_id))
                return await super().execute(model_id, kind, payload)

        server = EngineServer(SamplerExecutor())
        await server.accept_model("model-a", {})
        save_id = await server.save_weights_for_sampler(
            {
                "model_id": "model-a",
                "seq_id": 1,
                "publish_version": 1,
            }
        )
        forward_id = await forward_backward(server, 2)

        await asyncio.wait_for(persist_started.wait(), 1.0)
        forward = await server.retrieve_future(forward_id, timeout=1.0)
        assert forward.status == FutureStatus.COMPLETE
        assert (await server.retrieve_future(save_id)).status == FutureStatus.PENDING
        assert executed[0] == ("capture_sampler", "model-a")

        release_persist.set()
        saved = await server.retrieve_future(save_id, timeout=1.0)
        assert saved.status == FutureStatus.COMPLETE
        assert saved.result == {"publish_version": 1}
        await server.close()

    asyncio.run(run())
