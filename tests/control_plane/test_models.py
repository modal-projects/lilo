import asyncio

import pytest

from tests.support import EchoExecutor
from lilo.control_plane import ControlPlane, FutureResolutionStatus
from lilo.control_plane.keys import model_key, placement_key, trainer_demand_key
from lilo.engine import OperationKind
from lilo.errors import (
    RecordNotFound,
    RecordUnavailable,
    SequenceConflict,
)
from lilo.providers.local import (
    InMemoryKeyValueStore,
    LocalEnginePlatform,
)

DEFINITION = "qwen3_4b_lora32_16k"


async def checkpoint_metadata(path: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "base_model": "Qwen/Qwen3-4B",
        "engine_definition_id": DEFINITION,
        "parameterization": {"type": "lora"},
        "lora_config": {"rank": 32},
    }


async def session_and_plane(**platform_kwargs) -> tuple[ControlPlane, str]:
    plane = ControlPlane(
        InMemoryKeyValueStore(),
        LocalEnginePlatform(DEFINITION, EchoExecutor, **platform_kwargs),
    )
    session = await plane.create_session()
    return plane, session.session_id


def test_creation_places_and_resolves() -> None:
    async def run() -> None:
        plane, session_id = await session_and_plane()
        creation = await plane.create_model(
            session_id=session_id,
            model_seq_id=0,
            definition_id=DEFINITION,
            spec={"rank": 32},
        )
        assert creation.created
        resolution = await plane.retrieve(creation.request_id)
        assert resolution.status == FutureResolutionStatus.COMPLETE
        assert resolution.result == {
            "type": "create_model",
            "model_id": creation.model.model_id,
        }

    asyncio.run(run())


def test_checkpoint_load_failure_resolves_creation_as_failed() -> None:
    async def run() -> None:
        class FailingExecutor(EchoExecutor):
            async def execute(self, model_id, kind, payload):
                if kind == OperationKind.LOAD_WEIGHTS:
                    raise FileNotFoundError(payload.uri)
                return await super().execute(model_id, kind, payload)

        plane = ControlPlane(
            InMemoryKeyValueStore(),
            LocalEnginePlatform(DEFINITION, FailingExecutor),
            read_checkpoint_metadata=checkpoint_metadata,
        )
        session = await plane.create_session()
        creation = await plane.create_model_from_checkpoint(
            session_id=session.session_id,
            model_seq_id=0,
            path="tinker://run-x/weights/missing",
            base_model=None,
            user_metadata=None,
            optimizer=False,
        )

        resolution = await plane.retrieve(creation.request_id)
        assert resolution.status == FutureResolutionStatus.FAILED
        assert resolution.error == "accept model: /checkpoints/run-x/weights/missing"
        assert resolution.category == "user"

    asyncio.run(run())


CHECKPOINTS = [
    {
        "model_id": "run-a",
        "name": "older",
        "path": "/checkpoints/run-a/weights/older",
        "time": 100.0,
        "size_bytes": 10,
        "metadata": None,
    },
    {
        "model_id": "run-a",
        "name": "newer",
        "path": "/checkpoints/run-a/weights/newer",
        "time": 200.0,
        "size_bytes": 20,
        "metadata": {
            "base_model": "Qwen/Qwen3-4B",
            "parameterization": {"type": "lora"},
            "lora_config": {"rank": 32},
        },
    },
    {
        "model_id": "run-b",
        "name": "latest",
        "path": "/checkpoints/run-b/weights/latest",
        "time": 50.0,
        "size_bytes": 5,
        "metadata": {
            "base_model": "Qwen/Qwen3-4B",
            "parameterization": {"type": "full"},
        },
    },
]


def checkpoint_plane(**kwargs) -> tuple[ControlPlane, list[str]]:
    calls: list[str] = []

    async def list_checkpoints(model_id):
        return [entry for entry in CHECKPOINTS if model_id in (None, entry["model_id"])]

    async def record(path: str) -> None:
        calls.append(path)

    plane = ControlPlane(
        InMemoryKeyValueStore(),
        LocalEnginePlatform(DEFINITION, EchoExecutor),
        list_checkpoints=list_checkpoints,
        delete_checkpoint=record,
        **kwargs,
    )
    return plane, calls


def test_checkpoint_listing_derives_sdk_records() -> None:
    async def run() -> None:
        plane, calls = checkpoint_plane()
        entries = await plane.checkpoints("run-a")
        assert [entry["name"] for entry in entries] == ["newer", "older"]
        assert plane.checkpoint_record(entries[0]) == {
            "checkpoint_id": "weights/newer",
            "checkpoint_type": "training",
            "time": "1970-01-01T00:03:20+00:00",
            "tinker_path": "tinker://run-a/weights/newer",
            "size_bytes": 20,
        }
        run_a = await plane.training_run("run-a")
        assert run_a["base_model"] == "Qwen/Qwen3-4B"
        assert run_a["is_lora"] is True
        assert run_a["lora_rank"] == 32
        assert run_a["last_checkpoint"]["tinker_path"] == "tinker://run-a/weights/newer"
        runs = await plane.training_runs()
        assert [run["training_run_id"] for run in runs] == ["run-a", "run-b"]
        assert runs[1]["is_lora"] is False
        with pytest.raises(RecordNotFound):
            await plane.training_run("run-c")
        with pytest.raises(RecordNotFound):
            await plane.checkpoint("run-a", "weights/missing")

        await plane.remove_checkpoint("run-a", "older")
        assert calls == ["/checkpoints/run-a/weights/older"]

        assert (
            plane.resolve_checkpoint_path("tinker://run-a/weights/newer")
            == "/checkpoints/run-a/weights/newer"
        )
        assert plane.tinker_path("/checkpoints/run-a/weights/newer") == (
            "tinker://run-a/weights/newer"
        )
        for bad in (
            "/checkpoints/run-a/weights/newer",
            "tinker://run-a/sampler_weights/x",
            "tinker://../weights/x",
            "tinker://run-a/weights/..",
        ):
            with pytest.raises(ValueError):
                plane.resolve_checkpoint_path(bad)

    asyncio.run(run())


def test_checkpoint_paths_must_be_single_components() -> None:
    async def run() -> None:
        plane, _ = checkpoint_plane()
        for training_run_id, checkpoint_id in (
            ("../run-a", "weights/newer"),
            ("run-a", "weights/../newer"),
            ("run-a", "sampler_weights/newer"),
            ("run-a", " newer"),
            ("run-a", ".."),
        ):
            with pytest.raises(ValueError):
                await plane.checkpoint(training_run_id, checkpoint_id)
        bare = ControlPlane(
            InMemoryKeyValueStore(),
            LocalEnginePlatform(DEFINITION, EchoExecutor),
        )
        with pytest.raises(RecordUnavailable):
            await bare.checkpoints("run-a")
        with pytest.raises(RecordUnavailable):
            await ControlPlane(
                bare.kv,
                bare.engines,
                list_checkpoints=checkpoint_plane()[0].list_checkpoints,
            ).remove_checkpoint("run-a", "newer")

    asyncio.run(run())


def test_checkpoint_for_undeployed_definition_is_rejected() -> None:
    async def run() -> None:
        plane = ControlPlane(
            InMemoryKeyValueStore(),
            LocalEnginePlatform(DEFINITION, EchoExecutor),
            read_checkpoint_metadata=checkpoint_metadata,
        )
        session = await plane.create_session()
        with pytest.raises(ValueError, match="not deployed"):
            await plane.create_model_from_checkpoint(
                session_id=session.session_id,
                model_seq_id=0,
                path="tinker://run-x/weights/ckpt",
                base_model=None,
                user_metadata=None,
                optimizer=False,
                definition_ids={"other"},
            )

    asyncio.run(run())


def test_creation_is_idempotent() -> None:
    async def run() -> None:
        plane, session_id = await session_and_plane()
        first = await plane.create_model(
            session_id=session_id,
            model_seq_id=0,
            definition_id=DEFINITION,
            spec={"rank": 32},
        )
        retry = await plane.create_model(
            session_id=session_id,
            model_seq_id=0,
            definition_id=DEFINITION,
            spec={"rank": 32},
        )
        assert not retry.created
        assert retry.model == first.model
        assert retry.request_id == first.request_id

        with pytest.raises(SequenceConflict):
            await plane.create_model(
                session_id=session_id,
                model_seq_id=0,
                definition_id=DEFINITION,
                spec={"rank": 64},
            )

    asyncio.run(run())


def test_creation_pends_when_engine_is_full() -> None:
    async def run() -> None:
        plane, session_id = await session_and_plane(max_models=1)
        first = await plane.create_model(
            session_id=session_id,
            model_seq_id=0,
            definition_id=DEFINITION,
            spec={},
        )
        await plane.retrieve(first.request_id)
        overflow = await plane.create_model(
            session_id=session_id,
            model_seq_id=1,
            definition_id=DEFINITION,
            spec={},
        )
        resolution = await plane.retrieve(overflow.request_id)
        assert resolution.status == FutureResolutionStatus.PENDING

    asyncio.run(run())


def test_creation_reclaims_idle_session_when_engine_is_full() -> None:
    async def run() -> None:
        now = 0.0
        sessions = iter(("session-idle", "session-live"))
        plane = ControlPlane(
            InMemoryKeyValueStore(),
            LocalEnginePlatform(DEFINITION, EchoExecutor, max_models=1),
            session_idle_timeout=60.0,
            session_id_factory=lambda: next(sessions),
            clock=lambda: now,
        )
        idle = await plane.create_session()
        first = await plane.create_model(
            session_id=idle.session_id,
            model_seq_id=0,
            definition_id=DEFINITION,
            spec={},
        )
        assert (
            await plane.retrieve(first.request_id)
        ).status == FutureResolutionStatus.COMPLETE

        now = 100.0
        live = await plane.create_session()
        replacement = await plane.create_model(
            session_id=live.session_id,
            model_seq_id=0,
            definition_id=DEFINITION,
            spec={},
        )
        assert (
            await plane.retrieve(replacement.request_id)
        ).status == FutureResolutionStatus.COMPLETE
        with pytest.raises(RecordUnavailable):
            await plane.heartbeat(idle.session_id)

    asyncio.run(run())


def test_sweep_reclaims_idle_models() -> None:
    async def run() -> None:
        now = 0.0
        plane = ControlPlane(
            InMemoryKeyValueStore(),
            LocalEnginePlatform(DEFINITION, EchoExecutor, max_models=1),
            clock=lambda: now,
        )
        session = await plane.create_session()
        creation = await plane.create_model(
            session_id=session.session_id,
            model_seq_id=0,
            definition_id=DEFINITION,
            spec={},
        )
        await plane.retrieve(creation.request_id)

        now = 61.0
        assert await plane.sweep_idle_models(60.0) == (creation.model.model_id,)
        with pytest.raises(RecordUnavailable):
            await plane.heartbeat(session.session_id)

    asyncio.run(run())


def test_sweep_reclaims_orphaned_engine_models() -> None:
    async def run() -> None:
        kv = InMemoryKeyValueStore()
        engines = LocalEnginePlatform(DEFINITION, EchoExecutor, max_models=1)
        plane = ControlPlane(kv, engines, clock=lambda: 0.0)
        session = await plane.create_session()
        creation = await plane.create_model(
            session_id=session.session_id,
            model_seq_id=0,
            definition_id=DEFINITION,
            spec={},
        )
        await plane.retrieve(creation.request_id)
        model_id = creation.model.model_id
        for key in (model_key, placement_key, trainer_demand_key):
            await kv.delete(key(model_id))
        (instance,) = await engines.list_instances()
        client = engines.client(instance.instance_id)
        assert list(await client.model_ids()) == [model_id]

        assert await plane.sweep_idle_models(60.0) == (model_id,)
        assert list(await client.model_ids()) == []
        assert await plane.sweep_idle_engines() == (instance.instance_id,)

    asyncio.run(run())


def test_creation_future_reports_lost_after_engine_death() -> None:
    async def run() -> None:
        kv = InMemoryKeyValueStore()
        engines = LocalEnginePlatform(DEFINITION, EchoExecutor)
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
        instance = await engines.ensure_instance(DEFINITION)
        await engines.mark_dead(instance.instance_id)
        resolution = await plane.retrieve(creation.request_id)
        assert resolution.status == FutureResolutionStatus.LOST

    asyncio.run(run())


def test_prepare_model_runs_once_per_created_model() -> None:
    prepared = []

    async def prepare(model) -> None:
        prepared.append(model.model_id)

    async def run() -> None:
        plane = ControlPlane(
            InMemoryKeyValueStore(),
            LocalEnginePlatform(DEFINITION, EchoExecutor),
            prepare_model=prepare,
        )
        session = await plane.create_session()
        for _ in range(2):
            creation = await plane.create_model(
                session_id=session.session_id,
                model_seq_id=0,
                definition_id=DEFINITION,
                spec={"rank": 32},
            )
        assert prepared == [creation.model.model_id]

    asyncio.run(run())
