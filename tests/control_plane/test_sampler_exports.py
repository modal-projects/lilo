import asyncio

import pytest

from tests.support import TinkerStubExecutor
from lilo.control_plane import ControlPlane, FutureResolutionStatus
from lilo.control_plane.keys import sampler_export_result_key
from lilo.control_plane.records import SamplerExportResultRecord
from lilo.engine.api import OperationKind
from lilo.errors import RecordNotFound, RecordUnavailable, SequenceConflict
from lilo.providers.local import (
    InMemoryKeyValueStore,
    LocalEnginePlatform,
)

BASE_MODEL = "Qwen/Qwen3-8B"
DEFINITION = "qwen3_8b"


class VersionedExecutor(TinkerStubExecutor):
    async def execute(
        self,
        model_id: str,
        kind: OperationKind,
        payload: object,
    ) -> object:
        if kind == OperationKind.SAVE_WEIGHTS_FOR_SAMPLER:
            return {"publish_version": 7}
        return await super().execute(model_id, kind, payload)


async def plane_with_model(clock=lambda: 100.0):
    plane = ControlPlane(
        InMemoryKeyValueStore(),
        LocalEnginePlatform(DEFINITION, VersionedExecutor),
        clock=clock,
    )
    session = await plane.create_session()
    creation = await plane.create_model(
        session_id=session.session_id,
        model_seq_id=0,
        definition_id=DEFINITION,
        spec={"base_model": BASE_MODEL},
    )
    await plane.retrieve(creation.request_id)
    return plane, session.session_id, creation.model.model_id


def export_request(model_id: str, seq_id: int = 1, **updates) -> dict:
    request = {
        "type": "save_weights_for_sampler",
        "model_id": model_id,
        "seq_id": seq_id,
        "path": "checkpoint",
        "sampling_session_seq_id": None,
        "ttl_seconds": None,
    }
    request.update(updates)
    return request


def test_named_export_is_idempotent_and_resolves_exact_artifact() -> None:
    async def run() -> None:
        plane, session_id, model_id = await plane_with_model()
        request = export_request(model_id, ttl_seconds=60)
        request_id = await plane.submit_sampler_export(request)
        assert await plane.submit_sampler_export(request) == request_id

        resolution = await plane.retrieve(request_id, timeout=1.0)
        assert resolution.status == FutureResolutionStatus.COMPLETE
        model_path = f"tinker://{model_id}:train:0/sampler_weights/checkpoint"
        assert resolution.result == {
            "type": "save_weights_for_sampler",
            "path": model_path,
        }
        artifact = await plane.get_sampler_artifact(model_path)
        assert artifact.publish_version == 7
        assert artifact.model_path == model_path
        assert artifact.engine_definition_id == DEFINITION

        sampling = await plane.create_sampling_session(
            session_id=session_id,
            sampling_session_seq_id=0,
            model_path=model_path,
        )
        assert sampling.base_model == BASE_MODEL
        assert sampling.engine_definition_id == DEFINITION
        assert sampling.model_path == model_path
        assert sampling.publish_version == 7
        latest_path = f"tinker://{model_id}:train:0/sampler_weights/latest"
        latest = await plane.create_sampling_session(
            session_id=session_id,
            sampling_session_seq_id=1,
            model_path=latest_path,
        )
        assert latest.model_path == f"{latest_path}/000007"
        assert latest.latest
        assert latest.publish_version == 7
        legacy = sampling.model_dump(mode="json")
        legacy.pop("engine_definition_id")
        assert type(sampling).model_validate(legacy).engine_definition_id is None
        assert await plane.submit_sampler_export(request) == request_id
        assert (await plane.retrieve(request_id)).result == resolution.result

    asyncio.run(run())


def test_ephemeral_export_finalizes_deterministic_sampling_session() -> None:
    async def run() -> None:
        plane, session_id, model_id = await plane_with_model()
        request = export_request(
            model_id,
            path=None,
            sampling_session_seq_id=4,
        )
        request_id = await plane.submit_sampler_export(request)
        first = await plane.retrieve(request_id, timeout=1.0)
        second = await plane.retrieve(request_id)
        assert first == second
        assert first.status == FutureResolutionStatus.COMPLETE
        sampling_session_id = first.result["sampling_session_id"]
        session = await plane.get_sampling_session(sampling_session_id)
        assert session.session_id == session_id
        assert session.sampling_session_seq_id == 4
        assert session.model_id == model_id
        assert session.engine_definition_id == DEFINITION
        assert session.model_path == (
            f"tinker://{model_id}:train:0/sampler_weights/latest/000007"
        )
        assert session.latest
        assert session.publish_version == 7

    asyncio.run(run())


def test_ephemeral_export_retry_keeps_first_sampling_session_sequence() -> None:
    async def run() -> None:
        plane, session_id, model_id = await plane_with_model()
        first = export_request(
            model_id,
            path=None,
            sampling_session_seq_id=4,
        )
        retry = {**first, "sampling_session_seq_id": 5}

        request_id = await plane.submit_sampler_export(first)
        assert await plane.submit_sampler_export(retry) == request_id
        result = await plane.retrieve(request_id, timeout=1.0)
        session = await plane.get_sampling_session(result.result["sampling_session_id"])

        assert session.session_id == session_id
        assert session.sampling_session_seq_id == 4

    asyncio.run(run())


def test_export_validation_expiry_and_conflicts() -> None:
    async def run() -> None:
        now = 100.0
        plane, session_id, model_id = await plane_with_model(lambda: now)
        for request in (
            export_request(model_id, path=""),
            export_request(model_id, path="bad/name"),
            export_request(model_id, ttl_seconds=0),
            export_request(model_id, ttl_seconds=1.5),
            export_request(model_id, path=None),
            export_request(model_id, sampling_session_seq_id=0),
        ):
            with pytest.raises(ValueError):
                await plane.submit_sampler_export(request)

        request = export_request(model_id, ttl_seconds=10)
        request_id = await plane.submit_sampler_export(request)
        result = await plane.retrieve(request_id, timeout=1.0)
        model_path = result.result["path"]
        with pytest.raises(SequenceConflict):
            await plane.submit_sampler_export(export_request(model_id, ttl_seconds=11))
        with pytest.raises(RecordNotFound):
            await plane.get_sampler_artifact(model_path + "-other")

        now = 110.0
        with pytest.raises(RecordUnavailable):
            await plane.get_sampler_artifact(model_path)
        with pytest.raises(RecordUnavailable):
            await plane.create_sampling_session(
                session_id=session_id,
                sampling_session_seq_id=0,
                model_path=model_path,
            )

    asyncio.run(run())


def test_named_artifact_cannot_be_replaced_by_another_sequence() -> None:
    async def run() -> None:
        plane, _, model_id = await plane_with_model()
        first = await plane.submit_sampler_export(export_request(model_id))
        await plane.retrieve(first, timeout=1.0)
        second = await plane.submit_sampler_export(export_request(model_id, seq_id=2))
        with pytest.raises(SequenceConflict):
            await plane.retrieve(second, timeout=1.0)

    asyncio.run(run())


def test_ttl_starts_at_completion_and_propagates_to_session() -> None:
    async def run() -> None:
        now = 100.0
        plane, session_id, model_id = await plane_with_model(lambda: now)
        request_id = await plane.submit_sampler_export(
            export_request(model_id, ttl_seconds=10)
        )
        now = 200.0
        result = await plane.retrieve(request_id, timeout=1.0)
        session = await plane.create_sampling_session(
            session_id=session_id,
            sampling_session_seq_id=0,
            model_path=result.result["path"],
        )
        assert session.expires_at == 210.0

        now = 210.0
        with pytest.raises(RecordUnavailable):
            await plane.get_sampling_session(session.sampling_session_id)
        assert (
            await plane.create_sampling_session(
                session_id=session_id,
                sampling_session_seq_id=0,
                model_path=result.result["path"],
            )
            == session
        )
        expired = await plane.sweep_expired_sampling()
        assert len(expired) == 2
        with pytest.raises(RecordNotFound):
            await plane.get_sampler_artifact(result.result["path"])
        with pytest.raises(RecordNotFound):
            await plane.get_sampling_session(session.sampling_session_id)

    asyncio.run(run())


def test_ephemeral_session_id_cannot_resolve_another_export() -> None:
    async def run() -> None:
        plane, _, model_id = await plane_with_model()
        first = export_request(
            model_id,
            path=None,
            sampling_session_seq_id=4,
        )
        await plane.retrieve(
            await plane.submit_sampler_export(first),
            timeout=1.0,
        )
        second = export_request(
            model_id,
            seq_id=2,
            path=None,
            sampling_session_seq_id=4,
        )
        request_id = await plane.submit_sampler_export(second)
        with pytest.raises(SequenceConflict):
            await plane.retrieve(request_id, timeout=1.0)

    asyncio.run(run())


def test_durable_export_result_finalizes_without_engine_future() -> None:
    async def run() -> None:
        plane, _, model_id = await plane_with_model()
        request = export_request(model_id)
        request_id = await plane.submit_sampler_export(request)
        receipt = SamplerExportResultRecord(
            model_id=model_id,
            seq_id=1,
            result={"publish_version": 7},
            completed_at=123.0,
        )
        await plane.kv.put(
            sampler_export_result_key(model_id, 1),
            receipt.model_dump(mode="json"),
        )

        result = await plane.retrieve(request_id)
        assert result.status == FutureResolutionStatus.COMPLETE
        artifact = await plane.get_sampler_artifact(result.result["path"])
        assert artifact.created_at == 123.0

    asyncio.run(run())
