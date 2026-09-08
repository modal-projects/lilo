import asyncio
import json
from dataclasses import replace

import pytest

from lilo.control_plane import ControlPlane, FutureResolutionStatus
from lilo.control_plane.keys import placement_key, trainer_demand_key
from lilo.control_plane.records import PlacementRecord
from lilo.errors import ModelLost, RecordNotFound, RecordUnavailable
from lilo.providers.local import (
    InMemoryKeyValueStore,
    LocalEnginePlatform,
)
from tests.support import EchoExecutor

DEFINITION = "qwen3_4b_lora32_16k"


async def plane_with_model() -> tuple[ControlPlane, LocalEnginePlatform, str, str]:
    engines = LocalEnginePlatform(DEFINITION, EchoExecutor)
    plane = ControlPlane(InMemoryKeyValueStore(), engines)
    session = await plane.create_session()
    creation = await plane.create_model(
        session_id=session.session_id,
        model_seq_id=0,
        definition_id=DEFINITION,
        spec={"rank": 32},
    )
    await plane.retrieve(creation.request_id)
    return plane, engines, session.session_id, creation.model.model_id


def forward_backward_body(model_id: str, seq_id: int) -> bytes:
    return json.dumps(
        {
            "model_id": model_id,
            "seq_id": seq_id,
            "forward_backward_input": {
                "data": [
                    {
                        "model_input": {"chunks": [{"tokens": [seq_id]}]},
                        "loss_fn_inputs": {},
                    }
                ],
                "loss_fn": "cross_entropy",
            },
        }
    ).encode()


def test_operations_route_to_engine_and_resolve() -> None:
    async def run() -> None:
        plane, _, _, model_id = await plane_with_model()
        engine = await plane.engine_for(model_id)
        request_id = await engine.forward_backward(
            forward_backward_body(model_id, 1),
            "application/json",
        )
        resolution = await plane.retrieve(request_id, timeout=1.0)
        assert resolution.status == FutureResolutionStatus.COMPLETE
        assert resolution.result == {
            "model_id": model_id,
            "kind": "forward_backward",
            "payload": {
                "data": [
                    {
                        "loss_fn_inputs": {},
                        "model_input": {
                            "chunks": [{"tokens": [1]}]
                        },
                    }
                ],
                "loss_fn": "cross_entropy",
            },
        }

    asyncio.run(run())


def test_gap_pends_through_control_plane_retrieve() -> None:
    async def run() -> None:
        plane, _, _, model_id = await plane_with_model()
        engine = await plane.engine_for(model_id)
        later = await engine.optim_step(
            {"model_id": model_id, "seq_id": 2, "adam_params": {}}
        )
        assert (
            await plane.retrieve(later)
        ).status == FutureResolutionStatus.PENDING
        await engine.forward_backward(
            forward_backward_body(model_id, 1),
            "application/json",
        )
        assert (
            await plane.retrieve(later, timeout=1.0)
        ).status == FutureResolutionStatus.COMPLETE

    asyncio.run(run())


def test_engine_death_loses_futures_and_rejects_new_work() -> None:
    async def run() -> None:
        plane, engines, _, model_id = await plane_with_model()
        engine = await plane.engine_for(model_id)
        request_id = await engine.forward_backward(
            forward_backward_body(model_id, 1),
            "application/json",
        )
        instance = await engines.ensure_instance(DEFINITION)
        await engines.mark_dead(instance.instance_id)

        resolution = await plane.retrieve(request_id)
        assert resolution.status == FutureResolutionStatus.LOST
        assert await plane.kv.get(placement_key(model_id)) is None
        assert await plane.kv.get(trainer_demand_key(model_id)) is None
        with pytest.raises(ModelLost):
            await plane.engine_for(model_id)

    asyncio.run(run())


def test_engine_death_seen_by_creation_future_releases_demand() -> None:
    async def run() -> None:
        plane, engines, _, model_id = await plane_with_model()
        instance = await engines.ensure_instance(DEFINITION)
        await engines.mark_dead(instance.instance_id)

        creation = await plane.retrieve(f"{model_id}:0")
        assert creation.status == FutureResolutionStatus.LOST
        assert await plane.kv.get(placement_key(model_id)) is None
        assert await plane.kv.get(trainer_demand_key(model_id)) is None

    asyncio.run(run())


async def reboot(engines: LocalEnginePlatform, boot_id: str) -> None:
    instance = await engines.ensure_instance(DEFINITION)
    engines._instances[instance.instance_id] = replace(instance, boot_id=boot_id)


async def placement_of(plane: ControlPlane, model_id: str) -> PlacementRecord:
    return PlacementRecord.model_validate(await plane.kv.get(placement_key(model_id)))


def test_engine_restart_loses_model_and_releases_demand() -> None:
    async def run() -> None:
        engines = LocalEnginePlatform(DEFINITION, EchoExecutor)
        await reboot(engines, "boot-1")
        plane = ControlPlane(InMemoryKeyValueStore(), engines)
        session = await plane.create_session()
        creation = await plane.create_model(
            session_id=session.session_id,
            model_seq_id=0,
            definition_id=DEFINITION,
            spec={"rank": 32},
        )
        await plane.retrieve(creation.request_id)
        model_id = creation.model.model_id
        assert (await placement_of(plane, model_id)).engine_boot_id == "boot-1"
        engine = await plane.engine_for(model_id)
        request_id = await engine.forward_backward(
            forward_backward_body(model_id, 1),
            "application/json",
        )

        await reboot(engines, "boot-2")
        with pytest.raises(ModelLost):
            await plane.engine_for(model_id)
        assert await plane.kv.get(placement_key(model_id)) is None
        assert await plane.kv.get(trainer_demand_key(model_id)) is None
        resolution = await plane.retrieve(request_id)
        assert resolution.status == FutureResolutionStatus.LOST
        with pytest.raises(ModelLost):
            await plane.engine_for(model_id)
        creation_again = await plane.retrieve(creation.request_id)
        assert creation_again.status == FutureResolutionStatus.LOST
        assert await plane.kv.get(placement_key(model_id)) is None

    asyncio.run(run())


def test_placement_without_boot_id_survives_engine_boot_ids() -> None:
    async def run() -> None:
        plane, engines, _, model_id = await plane_with_model()
        assert (await placement_of(plane, model_id)).engine_boot_id == ""
        await reboot(engines, "boot-1")
        assert await plane.engine_for(model_id) is engines.client("instance-1")
        assert await plane.kv.get(placement_key(model_id)) is not None

    asyncio.run(run())


def test_unknown_and_closed_identities_are_rejected() -> None:
    async def run() -> None:
        plane, _, session_id, model_id = await plane_with_model()
        with pytest.raises(RecordNotFound):
            await plane.engine_for("unknown-model")
        with pytest.raises(RecordNotFound):
            await plane.retrieve("unknown-model:1")
        with pytest.raises(RecordNotFound):
            await plane.retrieve("not-a-request-id")

        await plane.close_session(session_id, "client requested")
        with pytest.raises(RecordUnavailable):
            await plane.engine_for(model_id)

    asyncio.run(run())


def test_session_close_unloads_models() -> None:
    async def run() -> None:
        plane, engines, session_id, model_id = await plane_with_model()
        engine = await plane.engine_for(model_id)
        request_id = await engine.optim_step(
            {"model_id": model_id, "seq_id": 2, "adam_params": {}}
        )
        await plane.close_session(session_id, "client requested")
        instance = await engines.ensure_instance(DEFINITION)
        assert await engines.client(instance.instance_id).accept_model(
            model_id,
            {},
        )
        resolution = await plane.retrieve(request_id)
        assert resolution.status == FutureResolutionStatus.LOST

    asyncio.run(run())


def test_placement_failure_leaves_creation_pending() -> None:
    class FlakyPlatform(LocalEnginePlatform):
        fail = True

        async def ensure_instance(self, definition):
            if self.fail:
                raise RuntimeError("platform outage")
            return await super().ensure_instance(definition)

    async def run() -> None:
        engines = FlakyPlatform(DEFINITION, EchoExecutor)
        plane = ControlPlane(InMemoryKeyValueStore(), engines)
        session = await plane.create_session()
        creation = await plane.create_model(
            session_id=session.session_id,
            model_seq_id=0,
            definition_id=DEFINITION,
            spec={"rank": 32},
        )
        resolution = await plane.retrieve(creation.request_id)
        assert resolution.status == FutureResolutionStatus.PENDING
        engines.fail = False
        resolution = await plane.retrieve(creation.request_id)
        assert resolution.status == FutureResolutionStatus.COMPLETE

    asyncio.run(run())


def test_sweep_reconciles_failed_session_unloads() -> None:
    class FlakyUnloadExecutor(EchoExecutor):
        async def unload_model(self, model_id: str) -> None:
            raise RuntimeError("engine busy")

    async def run() -> None:
        engines = LocalEnginePlatform(DEFINITION, FlakyUnloadExecutor)
        plane = ControlPlane(InMemoryKeyValueStore(), engines)
        session = await plane.create_session()
        creation = await plane.create_model(
            session_id=session.session_id,
            model_seq_id=0,
            definition_id=DEFINITION,
            spec={"rank": 32},
        )
        await plane.retrieve(creation.request_id)
        await plane.close_session(session.session_id, "client requested")
        assert await plane.kv.list_keys("placement:")
        await plane.sweep_idle_sessions(3600.0)
        assert not await plane.kv.list_keys("placement:")
        assert not await plane.kv.list_keys("session")
        assert not await plane.kv.list_keys("model")

    asyncio.run(run())


@pytest.mark.parametrize("state", ("running", "draining"))
def test_sweep_stops_empty_engines_immediately(state: str) -> None:
    async def run() -> None:
        engines = LocalEnginePlatform(DEFINITION, EchoExecutor)
        plane = ControlPlane(InMemoryKeyValueStore(), engines)
        session = await plane.create_session()
        creation = await plane.create_model(
            session_id=session.session_id,
            model_seq_id=0,
            definition_id=DEFINITION,
            spec={"rank": 32},
        )
        await plane.retrieve(creation.request_id)
        assert await plane.sweep_idle_engines() == ()
        await plane.close_session(session.session_id, "client requested")
        await engines.set_instance_state("instance-1", state)
        assert await plane.sweep_idle_engines() == ("instance-1",)
        assert (await engines.get_instance("instance-1")).terminal

    asyncio.run(run())


def test_sweep_never_stops_engine_with_registered_model() -> None:
    async def run() -> None:
        engines = LocalEnginePlatform(DEFINITION, EchoExecutor)
        plane = ControlPlane(InMemoryKeyValueStore(), engines)
        session = await plane.create_session()
        creation = await plane.create_model(
            session_id=session.session_id,
            model_seq_id=0,
            definition_id=DEFINITION,
            spec={"rank": 32},
        )
        await plane.retrieve(creation.request_id)
        await plane.kv.delete(placement_key(creation.model.model_id))
        await plane.kv.delete(trainer_demand_key(creation.model.model_id))
        assert await plane.sweep_idle_engines() == ()
        assert not (await engines.get_instance("instance-1")).terminal

    asyncio.run(run())


def test_sweep_keeps_empty_engine_with_unplaced_demand() -> None:
    async def run() -> None:
        engines = LocalEnginePlatform(DEFINITION, EchoExecutor)
        plane = ControlPlane(InMemoryKeyValueStore(), engines)
        await engines.spawn_instance(DEFINITION)
        session = await plane.create_session()
        await plane.create_model(
            session_id=session.session_id,
            model_seq_id=0,
            definition_id=DEFINITION,
            spec={"rank": 32},
        )
        assert await plane.sweep_idle_engines() == ()
        assert not (await engines.get_instance("instance-1")).terminal

    asyncio.run(run())
