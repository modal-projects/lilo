import asyncio

import pytest

from lilo.control_plane import ControlPlane, FutureResolutionStatus
from lilo.control_plane.keys import checkpoint_key
from lilo.control_plane.records import CheckpointRecord
from lilo.engine import OperationKind
from lilo.errors import RecordUnavailable
from lilo.providers.local import InMemoryKeyValueStore, LocalEnginePlatform
from tests.support import EchoExecutor

DEFINITION = "qwen3_4b_lora32_16k"


class CheckpointExecutor(EchoExecutor):
    async def persist_checkpoint(self, model_id, payload, snapshot):
        assert payload.destination
        return {
            "path": f"/checkpoints/{model_id}/weights/{payload.destination}",
            "type": OperationKind.SAVE_WEIGHTS.value,
        }


async def checkpoint_plane(
    now: list[float],
    delete_checkpoint,
) -> tuple[ControlPlane, str]:
    plane = ControlPlane(
        InMemoryKeyValueStore(),
        LocalEnginePlatform(DEFINITION, CheckpointExecutor),
        clock=lambda: now[0],
        delete_checkpoint=delete_checkpoint,
    )
    session = await plane.create_session()
    creation = await plane.create_model(
        session_id=session.session_id,
        model_seq_id=0,
        definition_id=DEFINITION,
        spec={"rank": 32},
    )
    await plane.retrieve(creation.request_id)
    return plane, creation.model.model_id


def test_checkpoint_ttl_starts_after_persistence_and_none_is_persistent() -> None:
    async def run() -> None:
        now = [100.0]
        deleted: list[str] = []

        async def delete_checkpoint(path: str) -> None:
            deleted.append(path)

        plane, model_id = await checkpoint_plane(now, delete_checkpoint)
        request_id = await plane.submit_checkpoint_save(
            {
                "type": "save_weights",
                "model_id": model_id,
                "seq_id": 1,
                "path": "ephemeral",
                "ttl_seconds": 60,
            }
        )
        resolution = await plane.retrieve(request_id, timeout=1.0)
        path = f"/checkpoints/{model_id}/weights/ephemeral"
        assert resolution.status == FutureResolutionStatus.COMPLETE
        assert resolution.result == {
            "type": "save_weights",
            "path": f"tinker://{model_id}/weights/ephemeral",
        }
        record = CheckpointRecord.model_validate(
            await plane.kv.get(checkpoint_key(path))
        )
        assert record.created_at == 100.0
        assert record.expires_at == 160.0
        assert record.state == "ready"

        now[0] = 159.0
        assert await plane.sweep_expired_checkpoints() == ()
        now[0] = 160.0
        with pytest.raises(RecordUnavailable, match="expired"):
            await plane.checkpoint_path_for_load(
                f"tinker://{model_id}/weights/ephemeral"
            )
        assert await plane.sweep_expired_checkpoints() == (path,)
        assert deleted == [path]
        record = CheckpointRecord.model_validate(
            await plane.kv.get(checkpoint_key(path))
        )
        assert record.state == "deleted"

        request_id = await plane.submit_checkpoint_save(
            {
                "type": "save_weights",
                "model_id": model_id,
                "seq_id": 2,
                "path": "persistent",
                "ttl_seconds": None,
            }
        )
        await plane.retrieve(request_id, timeout=1.0)
        persistent_path = f"/checkpoints/{model_id}/weights/persistent"
        persistent = CheckpointRecord.model_validate(
            await plane.kv.get(checkpoint_key(persistent_path))
        )
        assert persistent.expires_at is None
        assert (
            await plane.checkpoint_path_for_load(
                f"tinker://{model_id}/weights/persistent"
            )
            == persistent_path
        )
        now[0] = 1_000_000.0
        assert await plane.sweep_expired_checkpoints() == ()
        assert deleted == [path]

    asyncio.run(run())


def test_checkpoint_cleanup_retries_after_storage_error() -> None:
    async def run() -> None:
        now = [10.0]
        attempts: list[str] = []

        async def flaky_delete(path: str) -> None:
            attempts.append(path)
            if len(attempts) == 1:
                raise RuntimeError("volume unavailable")

        plane, model_id = await checkpoint_plane(now, flaky_delete)
        request_id = await plane.submit_checkpoint_save(
            {
                "type": "save_weights",
                "model_id": model_id,
                "seq_id": 1,
                "path": "retry",
                "ttl_seconds": 1,
            }
        )
        await plane.retrieve(request_id, timeout=1.0)
        path = f"/checkpoints/{model_id}/weights/retry"
        now[0] = 11.0

        assert await plane.sweep_expired_checkpoints() == ()
        record = CheckpointRecord.model_validate(
            await plane.kv.get(checkpoint_key(path))
        )
        assert record.state == "ready"
        assert await plane.sweep_expired_checkpoints() == (path,)
        assert attempts == [path, path]

    asyncio.run(run())


def test_checkpoint_sweep_reconciles_an_unretrieved_save() -> None:
    async def run() -> None:
        now = [10.0]
        deleted: list[str] = []

        async def delete_checkpoint(path: str) -> None:
            deleted.append(path)

        plane, model_id = await checkpoint_plane(now, delete_checkpoint)
        await plane.submit_checkpoint_save(
            {
                "type": "save_weights",
                "model_id": model_id,
                "seq_id": 1,
                "path": "unretrieved",
                "ttl_seconds": 1,
            }
        )
        await asyncio.sleep(0.01)
        now[0] = 20.0
        assert await plane.sweep_expired_checkpoints() == ()

        path = f"/checkpoints/{model_id}/weights/unretrieved"
        record = CheckpointRecord.model_validate(
            await plane.kv.get(checkpoint_key(path))
        )
        assert record.expires_at == 21.0
        now[0] = 21.0
        assert await plane.sweep_expired_checkpoints() == (path,)
        assert deleted == [path]

    asyncio.run(run())


def test_metadata_expiry_survives_missing_dict_records() -> None:
    async def run() -> None:
        path = "/checkpoints/run-a/weights/long-lived"
        metadata = {
            "expires_at": 100.0,
            "checkpoint_save_seq_id": 7,
        }
        deleted: list[str] = []

        async def read_metadata(uri: str):
            assert uri == path
            return metadata

        async def list_checkpoints(model_id):
            assert model_id is None
            return [
                {
                    "model_id": "run-a",
                    "name": "long-lived",
                    "path": path,
                    "time": 10.0,
                    "metadata": metadata,
                }
            ]

        async def delete_checkpoint(uri: str) -> None:
            deleted.append(uri)

        plane = ControlPlane(
            InMemoryKeyValueStore(),
            LocalEnginePlatform(DEFINITION, CheckpointExecutor),
            clock=lambda: 100.0,
            read_checkpoint_metadata=read_metadata,
            list_checkpoints=list_checkpoints,
            delete_checkpoint=delete_checkpoint,
        )
        with pytest.raises(RecordUnavailable, match="expired"):
            await plane.checkpoint_path_for_load(
                "tinker://run-a/weights/long-lived"
            )
        assert await plane.sweep_expired_checkpoints() == (path,)
        assert deleted == [path]

    asyncio.run(run())
