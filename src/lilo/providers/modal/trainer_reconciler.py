from __future__ import annotations

import logging
import math
import time
import uuid
from collections.abc import Awaitable, Callable

import modal
from pydantic import BaseModel

from lilo.control_plane.records import (
    ModelRecord,
    PlacementRecord,
    SessionClosedRecord,
)
from lilo.providers.contracts import (
    EngineInstance,
    EnginePlatform,
    KeyValueStore,
)

from .kv import shared_kv

RECONCILE_CALL_KEY = "trainer_reconcile:call"
RECONCILE_START_KEY = "trainer_reconcile:start"
RECONCILE_STARTED_AT_KEY = "trainer_reconcile:started_at"
RECONCILE_REQUEST_PREFIX = "trainer_reconcile_request:"
RECONCILE_COMPLETE_PREFIX = "trainer_reconcile_complete:"
TRAINER_PLAN_PREFIX = "trainer_plan:"
RECONCILE_DEBOUNCE_SECONDS = 5.0

_log = logging.getLogger(__name__)


class TrainerPlanRecord(BaseModel):
    definition_id: str
    observed_at: float
    live_models: int
    desired_instances: int
    maximum_instances: int | None
    active_instances: int
    starting_instances: int
    draining_instances: int
    pending_models: int
    converged: bool
    updated_at: float


def trainer_plan_key(definition_id: str) -> str:
    return f"{TRAINER_PLAN_PREFIX}{definition_id}"


async def request_reconcile(
    spawn: Callable[[float], Awaitable[str]],
    definition_id: str,
) -> str | None:
    store = shared_kv()
    await store.put(
        f"{RECONCILE_REQUEST_PREFIX}{definition_id}",
        uuid.uuid4().hex,
    )
    call_id = await store.get(RECONCILE_CALL_KEY)
    if call_id is not None and not await _call_finished(str(call_id)):
        return None
    claim = {"token": uuid.uuid4().hex, "created_at": time.time()}
    inserted = await store.put_if_absent(RECONCILE_START_KEY, claim)
    if not inserted.created:
        value = inserted.value
        if (
            isinstance(value, dict)
            and time.time() - float(value.get("created_at", 0)) < 60
        ):
            return None
        await store.delete(RECONCILE_START_KEY)
        return await request_reconcile(spawn, definition_id)
    try:
        now = time.time()
        last_started = await store.get(RECONCILE_STARTED_AT_KEY)
        delay = max(
            0.0,
            RECONCILE_DEBOUNCE_SECONDS
            - (
                now - float(last_started)
                if isinstance(last_started, (int, float))
                else RECONCILE_DEBOUNCE_SECONDS
            ),
        )
        call_id = await spawn(delay)
        await store.put(RECONCILE_CALL_KEY, call_id)
        await store.put(RECONCILE_STARTED_AT_KEY, now + delay)
        return call_id
    finally:
        await store.delete(RECONCILE_START_KEY)


async def pending_reconciliations() -> dict[str, str]:
    items = await shared_kv().list_items(
        RECONCILE_REQUEST_PREFIX,
        RECONCILE_COMPLETE_PREFIX,
    )
    requested = {
        key.removeprefix(RECONCILE_REQUEST_PREFIX): str(value)
        for key, value in items
        if key.startswith(RECONCILE_REQUEST_PREFIX)
    }
    completed = {
        key.removeprefix(RECONCILE_COMPLETE_PREFIX): str(value)
        for key, value in items
        if key.startswith(RECONCILE_COMPLETE_PREFIX)
    }
    return {
        definition_id: token
        for definition_id, token in requested.items()
        if completed.get(definition_id) != token
    }


async def complete_reconcile(definition_id: str, token: str) -> None:
    await shared_kv().put(
        f"{RECONCILE_COMPLETE_PREFIX}{definition_id}",
        token,
    )


async def release_reconcile_call(call_id: str) -> None:
    store = shared_kv()
    if await store.get(RECONCILE_CALL_KEY) == call_id:
        await store.delete(RECONCILE_CALL_KEY)


async def list_trainer_plans() -> tuple[TrainerPlanRecord, ...]:
    return tuple(
        TrainerPlanRecord.model_validate(value)
        for _, value in await shared_kv().list_items(TRAINER_PLAN_PREFIX)
    )


async def reconcile_trainers(
    kv: KeyValueStore,
    engines: EnginePlatform,
    definition_id: str,
    *,
    revision: str | None,
    maximum_instances: int | None,
    models_per_instance: int = 1,
    scale_up: bool = True,
    clock: Callable[[], float] = time.time,
) -> TrainerPlanRecord:
    def current_revision(instance: EngineInstance) -> bool:
        return revision is None or instance.revision == revision

    instances = await engines.list_instances()
    items = await kv.list_items(
        "model:",
        "placement:",
        "session_closed:",
        "trainer_demand:",
    )
    closed = {
        SessionClosedRecord.model_validate(value).session_id
        for key, value in items
        if key.startswith("session_closed:")
    }
    models = [
        ModelRecord.model_validate(value)
        for key, value in items
        if key.startswith("model:")
        and value["engine_definition_id"] == definition_id
        and value["session_id"] not in closed
    ]
    placements = {
        record.model_id: record
        for key, value in items
        if key.startswith("placement:")
        for record in (PlacementRecord.model_validate(value),)
    }
    demand_ids = {
        str(value["model_id"])
        for key, value in items
        if key.startswith("trainer_demand:")
        and isinstance(value, dict)
        and value.get("definition_id") == definition_id
        and isinstance(value.get("model_id"), str)
    } | set(placements)
    records = [
        instance
        for instance in instances
        if instance.definition_id == definition_id and not instance.terminal
    ]
    current_ids = {record.instance_id for record in records if current_revision(record)}
    demanded_models = [model for model in models if model.model_id in demand_ids]
    live_models = [
        model
        for model in demanded_models
        if (
            model.model_id not in placements
            or placements[model.model_id].engine_instance_id in current_ids
        )
    ]
    desired = math.ceil(len(live_models) / models_per_instance)
    if maximum_instances is not None:
        desired = min(desired, maximum_instances)
    current = [record for record in records if current_revision(record)]
    usable = [record for record in current if record.state in {"starting", "running"}]
    if scale_up:
        missing = max(0, desired - len(usable))
        if (
            missing
            and maximum_instances is not None
            and len(records) >= maximum_instances
        ):
            stale = [
                record
                for record in records
                if not current_revision(record)
                and record.state in {"running", "draining"}
            ]
            await _stop_empty(
                engines,
                stale,
                len(records) - maximum_instances + missing,
            )
            records = [
                instance
                for instance in await engines.list_instances()
                if instance.definition_id == definition_id
                and not instance.terminal
            ]
        available = (
            missing
            if maximum_instances is None
            else max(0, maximum_instances - len(records))
        )
        for _ in range(min(max(0, desired - len(usable)), available)):
            instance = await engines.spawn_instance(definition_id)
            usable.append(instance)
    records = [
        instance
        for instance in await engines.list_instances()
        if instance.definition_id == definition_id and not instance.terminal
    ]
    current = [record for record in records if current_revision(record)]
    running = sorted(
        (record for record in current if record.state == "running"),
        key=lambda record: record.instance_id,
    )
    await _stop_empty(engines, running, max(0, len(current) - desired))
    for record in records:
        if not current_revision(record) and record.state == "running":
            await engines.set_instance_state(record.instance_id, "draining")
    draining = [
        instance
        for instance in await engines.list_instances()
        if instance.definition_id == definition_id and instance.state == "draining"
    ]
    await _stop_empty(engines, draining, len(draining))
    final = [
        instance
        for instance in await engines.list_instances()
        if instance.definition_id == definition_id and not instance.terminal
    ]
    current_final = [record for record in final if current_revision(record)]
    active = sum(record.state == "running" for record in current_final)
    starting = sum(record.state == "starting" for record in current_final)
    draining_count = sum(record.state == "draining" for record in final)
    stale_starting = any(
        record.state == "starting" and not current_revision(record) for record in final
    )
    plan = TrainerPlanRecord(
        definition_id=definition_id,
        observed_at=clock(),
        live_models=len(demanded_models),
        desired_instances=desired,
        maximum_instances=maximum_instances,
        active_instances=active,
        starting_instances=starting,
        draining_instances=draining_count,
        pending_models=(
            0
            if maximum_instances is None
            else max(
                0,
                len(live_models) - maximum_instances * models_per_instance,
            )
        ),
        converged=(
            active + starting == desired and draining_count == 0 and not stale_starting
        ),
        updated_at=clock(),
    )
    await kv.put(trainer_plan_key(definition_id), plan.model_dump(mode="json"))
    return plan


async def _model_ids(
    engines: EnginePlatform,
    records: list[EngineInstance],
) -> dict[str, tuple[str, ...]]:
    result = {}
    for record in records:
        try:
            result[record.instance_id] = await engines.client(
                record.instance_id
            ).model_ids()
        except Exception:
            _log.exception("list models on %s", record.instance_id)
    return result


async def _stop_empty(
    engines: EnginePlatform,
    records: list[EngineInstance],
    limit: int,
) -> None:
    infos = await _model_ids(engines, records)
    stopped = 0
    for record in records:
        if stopped >= limit or infos.get(record.instance_id) != ():
            continue
        try:
            acknowledged = await engines.client(record.instance_id).shutdown_if_idle()
        except Exception:
            _log.exception("shutdown %s", record.instance_id)
            continue
        if not acknowledged:
            continue
        await engines.stop_instance(record.instance_id)
        stopped += 1


async def _call_finished(call_id: str) -> bool:
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
        _log.exception("trainer reconcile liveness poll %s", call_id)
        return False
    except Exception:
        return True
    return True
