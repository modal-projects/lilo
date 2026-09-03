import asyncio

import pytest

from tests.support import EchoExecutor, TinkerStubSampler
from lilo.control_plane import ControlPlane, FutureResolutionStatus
from lilo.control_plane.keys import (
    sample_task_key,
    session_key,
)
from lilo.errors import SequenceConflict
from lilo.providers import SamplingTask
from lilo.providers.local import (
    InMemoryKeyValueStore,
    InMemorySessionKeyValueStores,
    LocalEnginePlatform,
    LocalSamplingTaskPlatform,
)

DEFINITION = "qwen3_8b"
BASE_MODEL = "Qwen/Qwen3-8B"


async def sampling_plane(runner=None):
    if runner is None:
        runner = TinkerStubSampler()
    tasks = LocalSamplingTaskPlatform(runner)
    task_stores = InMemorySessionKeyValueStores()
    plane = ControlPlane(
        InMemoryKeyValueStore(),
        LocalEnginePlatform(DEFINITION, EchoExecutor),
        sampling_tasks=tasks,
        sampling_task_stores=task_stores,
    )
    session = await plane.create_session()
    sampling = await plane.create_sampling_session(
        session_id=session.session_id,
        sampling_session_seq_id=0,
        base_model=BASE_MODEL,
        engine_definition_id=DEFINITION,
    )
    return plane, tasks, session.session_id, sampling.sampling_session_id


def sample_request(sampling_session_id: str, seq_id: int = 0) -> dict:
    return {
        "type": "sample",
        "sampling_session_id": sampling_session_id,
        "seq_id": seq_id,
        "num_samples": 1,
        "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]},
        "sampling_params": {"max_tokens": 2},
    }


def test_sampling_sessions_and_submissions_are_idempotent() -> None:
    async def run() -> None:
        plane, tasks, session_id, sampling_session_id = await sampling_plane()
        duplicate = await plane.create_sampling_session(
            session_id=session_id,
            sampling_session_seq_id=0,
            base_model=BASE_MODEL,
        )
        assert duplicate.sampling_session_id == sampling_session_id

        request = sample_request(sampling_session_id)
        request_id = await plane.submit_sample(request)
        assert await plane.submit_sample(request) == request_id
        resolution = await plane.retrieve(request_id, timeout=1.0)
        assert resolution.status == FutureResolutionStatus.COMPLETE
        assert resolution.result["sequences"][0]["tokens"] == [0, 1]
        assert len(tasks._tasks) == 1

        changed = sample_request(sampling_session_id)
        changed["num_samples"] = 2
        with pytest.raises(SequenceConflict):
            await plane.submit_sample(changed)

    asyncio.run(run())


def test_session_collection_removes_task_store() -> None:
    async def run() -> None:
        plane, _, session_id, sampling_session_id = await sampling_plane()
        await plane.submit_sample(sample_request(sampling_session_id))
        task_stores = plane.sampling_task_stores
        assert isinstance(task_stores, InMemorySessionKeyValueStores)
        store = task_stores.for_session(session_id)
        assert await store.get(sample_task_key(sampling_session_id, 0)) is not None

        await plane.close_session(session_id, "done")
        await plane.sweep_idle_sessions(0)

        assert session_id not in task_stores.stores

    asyncio.run(run())


def test_task_store_delete_failure_retains_closed_session() -> None:
    class FailingStores(InMemorySessionKeyValueStores):
        async def delete_session(self, session_id: str) -> None:
            raise RuntimeError(session_id)

    async def run() -> None:
        kv = InMemoryKeyValueStore()
        stores = FailingStores()
        plane = ControlPlane(
            kv,
            LocalEnginePlatform(DEFINITION, EchoExecutor),
            sampling_tasks=LocalSamplingTaskPlatform(TinkerStubSampler()),
            sampling_task_stores=stores,
        )
        session = await plane.create_session()
        await plane.close_session(session.session_id, "done")

        with pytest.raises(RuntimeError, match=session.session_id):
            await plane.sweep_idle_sessions(0)

        assert await kv.get(session_key(session.session_id)) is not None

    asyncio.run(run())


def test_sampling_task_carries_engine_definition() -> None:
    async def run() -> None:
        seen = []

        async def capture(task: SamplingTask) -> object:
            seen.append(task)
            return {"type": "sample", "sequences": []}

        plane, _, _, sampling_session_id = await sampling_plane(capture)
        request_id = await plane.submit_sample(sample_request(sampling_session_id))
        await plane.retrieve(request_id, timeout=1.0)
        assert seen[0].engine_definition_id == DEFINITION
        assert seen[0].session_id

    asyncio.run(run())


def test_sampling_session_creation_detects_conflicts() -> None:
    async def run() -> None:
        plane, _, session_id, _ = await sampling_plane()
        with pytest.raises(SequenceConflict):
            await plane.create_sampling_session(
                session_id=session_id,
                sampling_session_seq_id=0,
                base_model="Qwen/Qwen3-4B",
            )

    asyncio.run(run())


def test_sample_future_pends_and_lost_tasks_are_retryable() -> None:
    async def run() -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def blocked(task: SamplingTask) -> object:
            started.set()
            await release.wait()
            return {"type": "sample", "sequences": []}

        plane, tasks, _, sampling_session_id = await sampling_plane(blocked)
        request_id = await plane.submit_sample(sample_request(sampling_session_id))
        await started.wait()
        resolution = await plane.retrieve(request_id)
        assert resolution.status == FutureResolutionStatus.PENDING

        task_id = next(iter(tasks._tasks))
        await tasks.forget(task_id)
        resolution = await plane.retrieve(request_id)
        assert resolution.status == FutureResolutionStatus.RETRYABLE

    asyncio.run(run())
