import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock, patch

import modal

from lilo.providers import SamplingTask, SamplingTaskStatus
from lilo.providers.local import InMemorySessionKeyValueStores
from lilo.providers.modal.kv import (
    SESSION_KV_NAME,
    ModalSessionKeyValueStores,
    app_store_name,
    session_task_store_name,
)
from lilo.providers.modal.sampling import (
    ModalSamplingTaskPlatform,
    call_key,
)


def task() -> SamplingTask:
    return SamplingTask(
        request_id="sample-a:0",
        session_id="session-a",
        sampling_session_id="sample-a",
        base_model="Qwen/Qwen3-8B",
        engine_definition_id="qwen3_8b",
        model_path=None,
        model_id=None,
        publish_version=None,
        payload={"seq_id": 0},
    )


def test_session_task_store_names_are_stable_and_bounded() -> None:
    first = session_task_store_name("session-a", "ap-test")
    assert first == session_task_store_name("session-a", "ap-test")
    assert first != session_task_store_name("session-b", "ap-test")
    assert first != session_task_store_name("session-a", "ap-other")
    assert len(first) <= 64


def test_shared_store_names_are_isolated_by_app() -> None:
    assert app_store_name(SESSION_KV_NAME, "ap-test") == "ap-test-sessions"
    assert app_store_name(SESSION_KV_NAME, "ap-test") != app_store_name(
        SESSION_KV_NAME,
        "ap-other",
    )


def test_session_task_store_deletes_named_dict() -> None:
    async def run() -> None:
        delete = AsyncMock()
        stores = ModalSessionKeyValueStores("ap-test", delete=delete)
        await stores.delete_session("session-a")
        delete.assert_awaited_once_with(
            session_task_store_name("session-a", "ap-test"),
            allow_missing=True,
        )

    asyncio.run(run())


def test_submit_persists_and_reuses_call_id() -> None:
    async def run() -> None:
        stores = InMemorySessionKeyValueStores()
        spawn = AsyncMock(return_value="fc-1")
        platform = ModalSamplingTaskPlatform(stores, spawn)
        assert await platform.submit(task()) == "fc-1"
        assert await platform.submit(task()) == "fc-1"
        spawn.assert_awaited_once_with(task())
        assert (
            await stores.for_session("session-a").get(call_key("sample-a:0")) == "fc-1"
        )

    asyncio.run(run())


def test_submit_isolated_by_parent_session() -> None:
    async def run() -> None:
        stores = InMemorySessionKeyValueStores()
        spawn = AsyncMock(side_effect=("fc-1", "fc-2"))
        platform = ModalSamplingTaskPlatform(stores, spawn)
        first = task()
        second = replace(first, session_id="session-b")

        assert await platform.submit(first) == "fc-1"
        assert await platform.submit(second) == "fc-2"

    asyncio.run(run())


async def retrieve_with(outcome):
    platform = ModalSamplingTaskPlatform(
        InMemorySessionKeyValueStores(),
        AsyncMock(),
    )
    call = AsyncMock()
    call.get.aio.side_effect = outcome if isinstance(outcome, BaseException) else None
    call.get.aio.return_value = None if isinstance(outcome, BaseException) else outcome
    with patch.object(modal.FunctionCall, "from_id", return_value=call):
        return await platform.retrieve("fc-1")


def test_retrieve_maps_modal_call_outcomes() -> None:
    async def run() -> None:
        complete = await retrieve_with({"type": "sample"})
        assert complete.status == SamplingTaskStatus.COMPLETE
        assert complete.result == {"type": "sample"}

        pending = await retrieve_with(TimeoutError())
        assert pending.status == SamplingTaskStatus.PENDING

        transient = await retrieve_with(modal.exception.ConnectionError("blip"))
        assert transient.status == SamplingTaskStatus.PENDING

        assert await retrieve_with(modal.exception.NotFoundError("gone")) is None

        failed = await retrieve_with(RuntimeError("bad request"))
        assert failed.status == SamplingTaskStatus.FAILED
        assert failed.error == "bad request"

    asyncio.run(run())
