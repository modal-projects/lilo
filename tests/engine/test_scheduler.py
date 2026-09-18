"""Scheduling regressions exercised through the public operation/future API."""

import asyncio
import json

import pytest

from lilo.engine import Engine, FutureStatus
from tests.engine.test_server import forward_backward
from tests.support import EchoExecutor


class GatedExecutor(EchoExecutor):
    """Hold the initial operation while competing clients fill their queues."""

    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = []
        self.batches = []

    async def execute(self, model_id, kind, payload):
        self.calls.append((model_id, kind))
        if model_id == "gate":
            self.started.set()
            await self.release.wait()
        return await super().execute(model_id, kind, payload)

    async def execute_forward_backward_batch(self, executions):
        self.batches.append([item.model_id for item in executions])
        return await super().execute_forward_backward_batch(executions)


async def setup(executor, models, **kwargs):
    server = Engine(executor, max_models=len(models) + 1, **kwargs)
    for model in ["gate", *models]:
        await server.accept_model(model, {})
    await optim(server, "gate", 1)
    await asyncio.wait_for(executor.started.wait(), 1)
    return server


async def optim(server, model, seq):
    return await server.optim_step(
        {"model_id": model, "seq_id": seq, "adam_params": {"learning_rate": 0.1}}
    )


async def complete(server, requests):
    for request in requests:
        state = await server.retrieve_future(request, timeout=1)
        assert state.status == FutureStatus.COMPLETE


def test_busy_first_client_cannot_starve_other_operations():
    async def run():
        executor = GatedExecutor()
        server = await setup(executor, ["a", "b", "c"])
        requests = [await optim(server, "a", seq) for seq in range(1, 9)]
        requests += [await optim(server, model, 1) for model in ["b", "c"]]
        executor.release.set()
        await complete(server, requests)
        assert [model for model, _ in executor.calls[:5]] == [
            "gate",
            "a",
            "b",
            "c",
            "a",
        ]
        await server.close()

    asyncio.run(run())


def test_bounded_batch_completes_without_waiting_for_remaining_clients():
    async def run():
        later_started, later_release = asyncio.Event(), asyncio.Event()

        class Executor(GatedExecutor):
            async def execute_forward_backward_batch(self, executions):
                if executions[0].model_id == "c":
                    later_started.set()
                    await later_release.wait()
                return await super().execute_forward_backward_batch(executions)

        executor = Executor()
        server = await setup(executor, ["a", "b", "c"], max_batch_tokens=8)
        requests = [
            await forward_backward(server, 1, data=[1] * 4, model_id=model)
            for model in ["a", "b", "c"]
        ]
        executor.release.set()
        await asyncio.wait_for(later_started.wait(), 1)
        await complete(server, requests[:2])
        assert (
            await server.retrieve_future(requests[2], timeout=0)
        ).status == FutureStatus.PENDING
        later_release.set()
        await complete(server, requests)
        assert executor.batches == [["a", "b"], ["c"]]
        await server.close()

    asyncio.run(run())


@pytest.mark.parametrize("fail_large", [False, True])
def test_oversized_request_runs_alone_and_does_not_block_later_turns(fail_large):
    async def run():
        class Executor(GatedExecutor):
            async def execute_forward_backward_batch(self, executions):
                result = await super().execute_forward_backward_batch(executions)
                if fail_large and executions[0].model_id == "a":
                    raise RuntimeError("oversized request failed")
                return result

        executor = Executor()
        server = await setup(executor, ["a", "b", "c"], max_batch_tokens=4)
        requests = [
            await forward_backward(server, 1, data=[1] * size, model_id=model)
            for model, size in [("a", 10), ("b", 2), ("c", 2)]
        ]
        executor.release.set()
        await complete(server, requests[1:])
        first = await server.retrieve_future(requests[0], timeout=1)
        assert first.status == (
            FutureStatus.FAILED if fail_large else FutureStatus.COMPLETE
        )
        assert executor.batches == [["a"], ["b", "c"]]
        if not fail_large:
            assert (
                len(
                    first.result["payload"]["data"][0]["model_input"]["chunks"][0][
                        "tokens"
                    ]
                )
                == 10
            )
        await server.close()

    asyncio.run(run())


def test_compatible_batching_does_not_skip_incompatible_client_next_turn():
    async def run():
        executor = GatedExecutor()
        server = await setup(executor, ["a", "b", "c"], max_batch_tokens=8)
        requests = []
        for seq in range(1, 5):
            for model in ["a", "c"]:
                requests.append(await forward_backward(server, seq, model_id=model))
        requests.append(
            await server.forward_backward(
                json.dumps(
                    {
                        "model_id": "b",
                        "seq_id": 1,
                        "forward_backward_input": {
                            "data": [
                                {
                                    "model_input": {"chunks": [{"tokens": [1]}]},
                                    "loss_fn_inputs": {},
                                }
                            ],
                            "loss_fn": "importance_sampling",
                        },
                    }
                ).encode(),
                "application/json",
            )
        )
        executor.release.set()
        await complete(server, requests)
        assert executor.batches[:2] == [["a", "c"], ["b"]]
        await server.close()

    asyncio.run(run())


def test_client_with_missing_sequence_does_not_block_ready_work():
    async def run():
        executor = GatedExecutor()
        server = await setup(executor, ["a", "b", "c"], max_batch_tokens=4)
        await forward_backward(server, 2, model_id="a")
        ready = await forward_backward(server, 1, model_id="b")
        executor.release.set()
        await complete(server, [ready])
        # Removing the previous anchor must leave the rotation usable.
        await server.unload_model("b")
        requests = [await forward_backward(server, 1, model_id=m) for m in ["a", "c"]]
        await complete(server, [*requests, "a:2"])
        await server.close()

    asyncio.run(run())


@pytest.mark.parametrize("budget", [0, -1])
def test_rejects_nonpositive_batch_budget(budget):
    with pytest.raises(ValueError, match="max_batch_tokens"):
        Engine(EchoExecutor(), max_batch_tokens=budget)
