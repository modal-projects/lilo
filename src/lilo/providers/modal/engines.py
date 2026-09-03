from __future__ import annotations

import logging
import uuid
from collections.abc import Awaitable, Callable

import modal
from pydantic import BaseModel

from lilo.engine.api import EngineApi
from lilo.engine.http import HttpEngineClient
from lilo.errors import RecordNotFound

from ..contracts import (
    TERMINAL_STATES,
    EngineInstance,
    EngineState,
    KeyValueStore,
)


class EngineInstanceRecord(BaseModel):
    instance_id: str
    definition_id: str
    revision: str
    state: EngineState
    call_id: str | None = None
    url: str | None = None
    token: str | None = None
    boot_id: str = ""

    def instance(self) -> EngineInstance:
        return EngineInstance(
            self.definition_id,
            self.instance_id,
            self.state,
            self.revision or None,
            self.boot_id,
        )


def instance_key(instance_id: str) -> str:
    return f"engine_instance:{instance_id}"


def call_key(instance_id: str) -> str:
    return f"engine_call:{instance_id}"


def current_instance_key(definition_id: str) -> str:
    return f"engine_current:{definition_id}"


SpawnEngine = Callable[[str, str], Awaitable[str]]


class ModalEnginePlatform:
    def __init__(self, kv: KeyValueStore, spawn: SpawnEngine) -> None:
        self.kv = kv
        self.spawn = spawn
        self._records: dict[str, EngineInstanceRecord] = {}
        self._clients: dict[
            str, tuple[tuple[str, str | None], HttpEngineClient]
        ] = {}
        self._call_ids: dict[str, str] = {}

    async def ensure_instance(
        self,
        definition_id: str,
    ) -> EngineInstance:
        key = current_instance_key(definition_id)
        current = await self.kv.get(key)
        if current is not None:
            instance_id = str(current)
            instance = await self.get_instance(instance_id)
            if instance is not None and instance.state in {"starting", "running"}:
                return instance
            if instance is None and not await self._call_finished(instance_id):
                return EngineInstance(definition_id, instance_id, "starting")
        proposed = await self.spawn_instance(definition_id)
        if current is not None:
            await self.kv.put(key, proposed.instance_id)
            return proposed
        inserted = await self.kv.put_if_absent(key, proposed.instance_id)
        if inserted.created:
            return proposed
        await self.stop_instance(proposed.instance_id)
        return await self.ensure_instance(definition_id)

    async def spawn_instance(self, definition_id: str) -> EngineInstance:
        instance_id = f"engine-{uuid.uuid4().hex[:12]}"
        call_id = await self._spawn(definition_id, instance_id)
        record = EngineInstanceRecord(
            instance_id=instance_id,
            definition_id=definition_id,
            revision="",
            state="starting",
            call_id=call_id,
        )
        await self.kv.put_if_absent(
            instance_key(instance_id),
            record.model_dump(mode="json"),
        )
        self._records[instance_id] = record
        return EngineInstance(definition_id, instance_id, "starting")

    async def active_instances(
        self,
        definition_id: str,
    ) -> tuple[EngineInstance, ...]:
        return tuple(
            instance
            for instance in await self.list_instances()
            if instance.definition_id == definition_id
            and instance.state in {"starting", "running"}
        )

    async def set_instance_state(
        self,
        instance_id: str,
        state: EngineState,
    ) -> EngineInstance:
        value = await self.kv.get(instance_key(instance_id))
        if value is None:
            raise RecordNotFound("engine instance", instance_id)
        record = EngineInstanceRecord.model_validate(value).model_copy(
            update={"state": state}
        )
        await self.kv.put(instance_key(instance_id), record.model_dump(mode="json"))
        self._records[instance_id] = record
        return record.instance()

    async def list_instances(self) -> tuple[EngineInstance, ...]:
        instances = []
        for key in await self.kv.list_keys("engine_instance:"):
            instance = await self.get_instance(key.removeprefix("engine_instance:"))
            if instance is not None:
                instances.append(instance)
        return tuple(instances)

    async def stop_instance(self, instance_id: str) -> None:
        value = await self.kv.get(instance_key(instance_id))
        record = (
            EngineInstanceRecord.model_validate(value) if value is not None else None
        )
        call_id = await self.kv.get(call_key(instance_id)) or (
            record.call_id if record is not None else None
        )
        if call_id is not None:
            try:
                await modal.FunctionCall.from_id(str(call_id)).cancel.aio(
                    terminate_containers=True
                )
            except Exception:
                logging.getLogger(__name__).exception("cancel %s", instance_id)
        if record is not None:
            pointer = current_instance_key(record.definition_id)
            if await self.kv.get(pointer) == instance_id:
                await self.kv.delete(pointer)
            await self.kv.delete(instance_key(instance_id))
        await self.kv.delete(call_key(instance_id))
        self._records.pop(instance_id, None)
        self._clients.pop(instance_id, None)
        self._call_ids.pop(instance_id, None)

    async def get_instance(self, instance_id: str) -> EngineInstance | None:
        value = await self.kv.get(instance_key(instance_id))
        if value is None:
            return None
        record = EngineInstanceRecord.model_validate(value)
        if record.state not in TERMINAL_STATES and await self._call_finished(
            instance_id, record.call_id
        ):
            record = record.model_copy(update={"state": "dead"})
            await self.kv.put(
                instance_key(instance_id),
                record.model_dump(mode="json"),
            )
        self._records[instance_id] = record
        return record.instance()

    async def _spawn(self, definition_id: str, instance_id: str) -> str:
        call_id = await self.spawn(definition_id, instance_id)
        await self.kv.put(call_key(instance_id), call_id)
        self._call_ids[instance_id] = call_id
        return call_id

    async def _call_finished(
        self,
        instance_id: str,
        record_call_id: str | None = None,
    ) -> bool:
        call_id = self._call_ids.get(instance_id) or record_call_id
        if call_id is None:
            value = await self.kv.get(call_key(instance_id))
            if value is None:
                return True
            call_id = str(value)
        self._call_ids[instance_id] = call_id
        try:
            await modal.FunctionCall.from_id(call_id).get.aio(timeout=0)
        except TimeoutError:
            return False
        except (
            modal.exception.AuthError,
            modal.exception.ConnectionError,
            modal.exception.InternalError,
            modal.exception.ResourceExhaustedError,
            modal.exception.ServiceError,
            OSError,
        ):
            logging.getLogger(__name__).exception("liveness poll %s", instance_id)
            return False
        except Exception:
            return True
        return True

    def client(self, instance_id: str) -> EngineApi:
        record = self._records.get(instance_id)
        if record is None or record.url is None:
            raise RecordNotFound("engine instance", instance_id)
        endpoint = (record.url, record.token)
        cached = self._clients.get(instance_id)
        if cached is None or cached[0] != endpoint:
            cached = (endpoint, HttpEngineClient(record.url, token=record.token))
            self._clients[instance_id] = cached
        return cached[1]

