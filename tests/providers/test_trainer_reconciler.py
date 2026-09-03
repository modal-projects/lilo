import asyncio

from lilo.control_plane import ControlPlane, FutureResolutionStatus
from lilo.providers.local import (
    InMemoryKeyValueStore,
    LocalEnginePlatform,
)
from lilo.providers.modal.trainer_reconciler import reconcile_trainers
from tests.support import EchoExecutor

DEFINITION = "qwen_full"


def test_trainer_reconciler_scales_and_places_up_to_cap() -> None:
    async def run() -> None:
        kv = InMemoryKeyValueStore()
        engines = LocalEnginePlatform(DEFINITION, EchoExecutor, max_models=1)
        plane = ControlPlane(kv, engines)
        session = await plane.create_session()
        creations = [
            await plane.create_model(
                session_id=session.session_id,
                model_seq_id=index,
                definition_id=DEFINITION,
                spec={},
            )
            for index in range(3)
        ]

        plan = await reconcile_trainers(
            kv,
            engines,
            DEFINITION,
            revision=None,
            maximum_instances=2,
        )

        assert plan.desired_instances == 2
        assert plan.active_instances == 2
        assert plan.pending_models == 1
        first = await plane.retrieve(creations[0].request_id)
        second = await plane.retrieve(creations[1].request_id)
        overflow = await plane.retrieve(creations[2].request_id)
        assert first.status == FutureResolutionStatus.COMPLETE
        assert second.status == FutureResolutionStatus.COMPLETE
        assert overflow.status == FutureResolutionStatus.PENDING
        placements = await kv.list_items("placement:")
        assert len({value["engine_instance_id"] for _, value in placements}) == 2

    asyncio.run(run())


def test_trainer_reconciler_stops_surplus_immediately() -> None:
    async def run() -> None:
        kv = InMemoryKeyValueStore()
        engines = LocalEnginePlatform(DEFINITION, EchoExecutor, max_models=1)
        plane = ControlPlane(kv, engines)
        sessions = [await plane.create_session() for _ in range(2)]
        creations = [
            await plane.create_model(
                session_id=session.session_id,
                model_seq_id=0,
                definition_id=DEFINITION,
                spec={},
            )
            for session in sessions
        ]
        await reconcile_trainers(
            kv,
            engines,
            DEFINITION,
            revision=None,
            maximum_instances=2,
        )
        for creation in creations:
            await plane.retrieve(creation.request_id)

        await plane.close_session(sessions[1].session_id, "done")
        plan = await reconcile_trainers(
            kv,
            engines,
            DEFINITION,
            revision=None,
            maximum_instances=2,
        )
        assert plan.active_instances == 1
        assert plan.draining_instances == 0
        assert (
            len(
                [
                    instance
                    for instance in await engines.list_instances()
                    if not instance.terminal
                ]
            )
            == 1
        )

    asyncio.run(run())


def test_trainer_reconciler_keeps_empty_engine_for_unplaced_demand() -> None:
    async def run() -> None:
        kv = InMemoryKeyValueStore()
        engines = LocalEnginePlatform(DEFINITION, EchoExecutor, max_models=1)
        plane = ControlPlane(kv, engines)
        session = await plane.create_session()
        await plane.create_model(
            session_id=session.session_id,
            model_seq_id=0,
            definition_id=DEFINITION,
            spec={},
        )
        await reconcile_trainers(
            kv,
            engines,
            DEFINITION,
            revision=None,
            maximum_instances=2,
        )
        plan = await reconcile_trainers(
            kv,
            engines,
            DEFINITION,
            revision=None,
            maximum_instances=2,
        )

        assert plan.active_instances == 1
        assert plan.draining_instances == 0
        assert not (await engines.get_instance("instance-1")).terminal

    asyncio.run(run())


def test_trainer_reconciler_respawns_when_demand_returns() -> None:
    async def run() -> None:
        kv = InMemoryKeyValueStore()
        engines = LocalEnginePlatform(DEFINITION, EchoExecutor, max_models=1)
        plane = ControlPlane(kv, engines)
        sessions = [await plane.create_session() for _ in range(2)]
        creations = [
            await plane.create_model(
                session_id=session.session_id,
                model_seq_id=0,
                definition_id=DEFINITION,
                spec={},
            )
            for session in sessions
        ]
        await reconcile_trainers(
            kv,
            engines,
            DEFINITION,
            revision=None,
            maximum_instances=2,
        )
        for creation in creations:
            await plane.retrieve(creation.request_id)
        await plane.close_session(sessions[1].session_id, "done")
        await reconcile_trainers(
            kv,
            engines,
            DEFINITION,
            revision=None,
            maximum_instances=2,
        )

        replacement_session = await plane.create_session()
        replacement = await plane.create_model(
            session_id=replacement_session.session_id,
            model_seq_id=0,
            definition_id=DEFINITION,
            spec={},
        )
        plan = await reconcile_trainers(
            kv,
            engines,
            DEFINITION,
            revision=None,
            maximum_instances=2,
        )

        assert plan.active_instances == 2
        assert plan.draining_instances == 0
        assert (
            await plane.retrieve(replacement.request_id)
        ).status == FutureResolutionStatus.COMPLETE

    asyncio.run(run())


def test_trainer_reconciler_replaces_empty_stale_revision_at_cap() -> None:
    async def run() -> None:
        kv = InMemoryKeyValueStore()
        engines = LocalEnginePlatform(
            DEFINITION,
            EchoExecutor,
            max_models=1,
            revision="old",
        )
        await engines.spawn_instance(DEFINITION)
        plane = ControlPlane(kv, engines)
        session = await plane.create_session()
        await plane.create_model(
            session_id=session.session_id,
            model_seq_id=0,
            definition_id=DEFINITION,
            spec={},
        )
        engines.revision = "new"

        plan = await reconcile_trainers(
            kv,
            engines,
            DEFINITION,
            revision="new",
            maximum_instances=1,
        )

        live = [
            instance
            for instance in await engines.list_instances()
            if not instance.terminal
        ]
        assert plan.active_instances == 1
        assert len(live) == 1
        assert live[0].revision == "new"

    asyncio.run(run())


def test_unspecified_revision_keeps_running_engine_current() -> None:
    async def run() -> None:
        kv = InMemoryKeyValueStore()
        engines = LocalEnginePlatform(
            DEFINITION,
            EchoExecutor,
            max_models=8,
            revision="image-1",
        )
        await engines.spawn_instance(DEFINITION)
        plane = ControlPlane(kv, engines)
        session = await plane.create_session()
        creation = await plane.create_model(
            session_id=session.session_id,
            model_seq_id=0,
            definition_id=DEFINITION,
            spec={},
        )
        assert (
            await plane.retrieve(creation.request_id)
        ).status == FutureResolutionStatus.COMPLETE

        plan = await reconcile_trainers(
            kv,
            engines,
            DEFINITION,
            revision=None,
            maximum_instances=1,
            models_per_instance=8,
            scale_up=False,
        )

        assert plan.active_instances == 1
        assert plan.draining_instances == 0
        instances = await engines.list_instances()
        assert len(instances) == 1
        assert instances[0].state == "running"
        assert instances[0].revision == "image-1"

    asyncio.run(run())


def test_concurrent_stateless_placement_uses_one_engine() -> None:
    async def run() -> None:
        kv = InMemoryKeyValueStore()
        engines = LocalEnginePlatform(DEFINITION, EchoExecutor, max_models=1)
        first = ControlPlane(kv, engines)
        second = ControlPlane(kv, engines)
        session = await first.create_session()
        creation = await first.create_model(
            session_id=session.session_id,
            model_seq_id=0,
            definition_id=DEFINITION,
            spec={},
        )
        await reconcile_trainers(
            kv,
            engines,
            DEFINITION,
            revision=None,
            maximum_instances=2,
        )

        resolutions = await asyncio.gather(
            first.retrieve(creation.request_id),
            second.retrieve(creation.request_id),
        )

        assert {resolution.status for resolution in resolutions} <= {
            FutureResolutionStatus.COMPLETE,
            FutureResolutionStatus.PENDING,
        }
        assert FutureResolutionStatus.COMPLETE in {
            resolution.status for resolution in resolutions
        }
        instances = [
            instance
            for instance in await engines.list_instances()
            if not instance.terminal
        ]
        hosted = await asyncio.gather(
            *(
                engines.client(instance.instance_id).model_ids()
                for instance in instances
            )
        )
        assert sum(len(model_ids) for model_ids in hosted) == 1

    asyncio.run(run())


def test_trainer_reconciler_scales_without_a_cap() -> None:
    async def run() -> None:
        kv = InMemoryKeyValueStore()
        engines = LocalEnginePlatform(DEFINITION, EchoExecutor, max_models=1)
        plane = ControlPlane(kv, engines)
        session = await plane.create_session()
        creations = [
            await plane.create_model(
                session_id=session.session_id,
                model_seq_id=index,
                definition_id=DEFINITION,
                spec={},
            )
            for index in range(5)
        ]

        plan = await reconcile_trainers(
            kv,
            engines,
            DEFINITION,
            revision=None,
            maximum_instances=None,
        )

        assert plan.desired_instances == 5
        assert plan.active_instances == 5
        assert plan.maximum_instances is None
        assert plan.pending_models == 0
        resolutions = [
            await plane.retrieve(creation.request_id) for creation in creations
        ]
        assert all(
            resolution.status == FutureResolutionStatus.COMPLETE
            for resolution in resolutions
        )

    asyncio.run(run())
