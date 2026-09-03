import asyncio

import pytest

from lilo.control_plane import ControlPlane
from lilo.errors import RecordNotFound, RecordUnavailable
from tests.support import EchoExecutor
from lilo.providers.local import (
    LocalEnginePlatform,
    InMemoryKeyValueStore,
)

DEFINITION = "qwen3_4b_lora32_16k"


def control_plane(**kwargs) -> ControlPlane:
    return ControlPlane(
        InMemoryKeyValueStore(),
        LocalEnginePlatform(DEFINITION, EchoExecutor),
        **kwargs,
    )


def test_session_lifecycle() -> None:
    async def run() -> None:
        times = iter((1.0, 2.0, 3.0))
        plane = control_plane(
            session_id_factory=lambda: "session-1",
            clock=lambda: next(times),
        )
        session = await plane.create_session()
        assert session.created_at == 1.0
        assert (await plane.heartbeat("session-1")).seen_at == 2.0

        closed = await plane.close_session("session-1", "client requested")
        assert closed.closed_at == 3.0
        with pytest.raises(RecordUnavailable):
            await plane.heartbeat("session-1")

    asyncio.run(run())


def test_close_is_idempotent() -> None:
    async def run() -> None:
        plane = control_plane(session_id_factory=lambda: "session-1")
        await plane.create_session()
        first = await plane.close_session("session-1", "first")
        second = await plane.close_session("session-1", "second")
        assert second == first

    asyncio.run(run())


def test_sweep_closes_idle_sessions_only() -> None:
    async def run() -> None:
        now = 0.0
        sessions = iter(("session-idle", "session-live"))
        plane = control_plane(
            session_id_factory=lambda: next(sessions),
            clock=lambda: now,
        )
        await plane.create_session()
        now = 100.0
        await plane.create_session()

        now = 150.0
        assert await plane.sweep_idle_sessions(idle_timeout=60.0) == (
            "session-idle",
        )
        with pytest.raises(RecordNotFound):
            await plane.heartbeat("session-idle")
        assert await plane.kv.list_keys("session:") == ("session:session-live",)
        await plane.heartbeat("session-live")
        assert await plane.sweep_idle_sessions(idle_timeout=60.0) == ()

    asyncio.run(run())


def test_unknown_session_is_rejected() -> None:
    async def run() -> None:
        plane = control_plane()
        with pytest.raises(RecordNotFound):
            await plane.heartbeat("missing")
        with pytest.raises(RecordNotFound):
            await plane.close_session("missing", "reason")

    asyncio.run(run())
