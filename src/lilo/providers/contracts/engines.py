from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from lilo.engine.api import EngineApi

EngineState = Literal["starting", "running", "draining", "stopped", "dead"]
Parameterization = Literal["lora", "full"]

TERMINAL_STATES: frozenset[EngineState] = frozenset({"stopped", "dead"})


@dataclass(frozen=True)
class EngineInstance:
    definition_id: str
    instance_id: str
    state: EngineState
    revision: str | None = None
    boot_id: str = ""

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES


class EnginePlatform(Protocol):
    async def ensure_instance(
        self,
        definition_id: str,
    ) -> EngineInstance: ...

    async def spawn_instance(self, definition_id: str) -> EngineInstance: ...

    async def active_instances(
        self,
        definition_id: str,
    ) -> tuple[EngineInstance, ...]: ...

    async def get_instance(self, instance_id: str) -> EngineInstance | None: ...

    async def list_instances(self) -> tuple[EngineInstance, ...]: ...

    def client(self, instance_id: str) -> EngineApi: ...

    async def set_instance_state(
        self,
        instance_id: str,
        state: EngineState,
    ) -> EngineInstance: ...

    async def stop_instance(self, instance_id: str) -> None: ...
