from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import replace

from lilo.engine.api import EngineApi, Executor
from lilo.engine.server import EngineServer
from lilo.errors import RecordNotFound

from ..contracts import EngineInstance, EngineState


class LocalEnginePlatform:
    def __init__(
        self,
        definition_id: str,
        executor_factory: Callable[[], Executor],
        *,
        max_models: int = 8,
        revision: str | None = None,
    ) -> None:
        self.definition_id = definition_id
        self.executor_factory = executor_factory
        self.max_models = max_models
        self.revision = revision
        self._instances: dict[str, EngineInstance] = {}
        self._servers: dict[str, EngineServer] = {}
        self._current: str | None = None
        self._next_instance = 1
        self._lock = asyncio.Lock()

    async def ensure_instance(
        self,
        definition_id: str,
    ) -> EngineInstance:
        if definition_id != self.definition_id:
            raise RecordNotFound("engine definition", definition_id)
        async with self._lock:
            current = (
                self._instances.get(self._current) if self._current else None
            )
            if (
                current is not None
                and current.state in {"starting", "running"}
                and current.revision == self.revision
            ):
                return current
            instance = self._spawn_instance()
            self._current = instance.instance_id
            return instance

    async def spawn_instance(self, definition_id: str) -> EngineInstance:
        if definition_id != self.definition_id:
            raise RecordNotFound("engine definition", definition_id)
        async with self._lock:
            return self._spawn_instance()

    async def active_instances(
        self,
        definition_id: str,
    ) -> tuple[EngineInstance, ...]:
        if definition_id != self.definition_id:
            raise RecordNotFound("engine definition", definition_id)
        async with self._lock:
            return tuple(
                instance
                for instance in self._instances.values()
                if instance.state in {"starting", "running"}
                and instance.revision == self.revision
            )

    async def get_instance(self, instance_id: str) -> EngineInstance | None:
        async with self._lock:
            return self._instances.get(instance_id)

    async def list_instances(self) -> tuple[EngineInstance, ...]:
        async with self._lock:
            return tuple(self._instances.values())

    async def stop_instance(self, instance_id: str) -> None:
        async with self._lock:
            instance = self._instances.get(instance_id)
            if instance is None:
                return
            self._instances[instance_id] = replace(instance, state="stopped")
            if self._current == instance_id:
                self._current = None

    async def set_instance_state(
        self,
        instance_id: str,
        state: EngineState,
    ) -> EngineInstance:
        return await self._set_state(instance_id, state)

    def client(self, instance_id: str) -> EngineApi:
        server = self._servers.get(instance_id)
        if server is None:
            raise RecordNotFound("engine instance", instance_id)
        return server

    async def mark_dead(self, instance_id: str) -> EngineInstance:
        return await self._set_state(instance_id, "dead")

    async def _set_state(
        self,
        instance_id: str,
        state: EngineState,
    ) -> EngineInstance:
        async with self._lock:
            updated = replace(self._instances[instance_id], state=state)
            self._instances[instance_id] = updated
            return updated

    def _spawn_instance(self) -> EngineInstance:
        instance = EngineInstance(
            definition_id=self.definition_id,
            instance_id=f"instance-{self._next_instance}",
            state="running",
            revision=self.revision,
        )
        self._next_instance += 1
        self._instances[instance.instance_id] = instance
        self._servers[instance.instance_id] = EngineServer(
            self.executor_factory(),
            max_models=self.max_models,
        )
        return instance
