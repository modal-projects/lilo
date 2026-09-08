from __future__ import annotations

import hashlib
import logging
import time
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import PurePosixPath

from lilo.encoding import fingerprint
from lilo.engine.api import EngineApi, FutureStatus
from lilo.errors import (
    ModelLost,
    RecordNotFound,
    RecordUnavailable,
    SequenceConflict,
)
from lilo.providers.contracts import (
    EngineInstance,
    EnginePlatform,
    KeyValueStore,
    SamplingTask,
    SamplingTaskPlatform,
    SamplingTaskStatus,
    SessionKeyValueStores,
)

from .keys import (
    model_creation_key,
    model_key,
    placement_claim_key,
    placement_key,
    trainer_demand_key,
    sample_task_key,
    sampler_artifact_key,
    sampler_export_result_key,
    sampler_export_submission_key,
    sampling_session_creation_key,
    sampling_session_key,
    session_closed_key,
    session_key,
    session_last_seen_key,
)
from .records import (
    ModelCreationRecord,
    ModelRecord,
    PlacementRecord,
    SamplerArtifactRecord,
    SamplerExportResultRecord,
    SamplerExportSubmissionRecord,
    SampleTaskRecord,
    SamplingSessionCreationRecord,
    SamplingSessionRecord,
    SessionClosedRecord,
    SessionLastSeenRecord,
    SessionRecord,
)


def request_id_for(model_id: str, seq_id: int) -> str:
    return f"{model_id}:{seq_id}"


def parse_request_id(request_id: str) -> tuple[str, int]:
    model_id, _, seq = request_id.rpartition(":")
    if not model_id or not seq.isdigit():
        raise RecordNotFound("future", request_id)
    return model_id, int(seq)


def path_component(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or value in {".", ".."}
        or any(char in value for char in "/\\\0")
    ):
        raise ValueError(f"{label} must be a single path component")
    return value


def checkpoint_tinker_path(model_id: str, name: str) -> str:
    return f"tinker://{model_id}/weights/{name}"


CheckpointListing = Callable[[str | None], Awaitable[Sequence[Mapping[str, object]]]]


class FutureResolutionStatus(StrEnum):
    PENDING = "pending"
    COMPLETE = "complete"
    FAILED = "failed"
    LOST = "lost"
    RETRYABLE = "retryable"


@dataclass(frozen=True)
class FutureResolution:
    request_id: str
    status: FutureResolutionStatus
    result: object | None = None
    error: str | None = None
    category: str = "server"


@dataclass(frozen=True)
class ModelCreation:
    model: ModelRecord
    request_id: str
    created: bool


class ControlPlane:
    def __init__(
        self,
        kv: KeyValueStore,
        engines: EnginePlatform,
        *,
        sampling_tasks: SamplingTaskPlatform | None = None,
        session_idle_timeout: float | None = None,
        session_id_factory: Callable[[], str] = lambda: str(uuid.uuid4()),
        clock: Callable[[], float] = time.time,
        ensure_sampling_pool: Callable[
            [SamplingSessionRecord], Awaitable[None]
        ]
        | None = None,
        sampling_task_stores: SessionKeyValueStores | None = None,
        read_checkpoint_metadata: Callable[
            [str], Awaitable[Mapping[str, object]]
        ]
        | None = None,
        reconcile_trainers: Callable[[str], Awaitable[bool | None]] | None = None,
        prepare_model: Callable[[ModelRecord], Awaitable[None]] | None = None,
        trainer_autoscaling: Callable[[str], bool] = lambda _: False,
        list_checkpoints: CheckpointListing | None = None,
        delete_checkpoint: Callable[[str], Awaitable[None]] | None = None,
        checkpoint_root: str = "/checkpoints",
    ) -> None:
        self.kv = kv
        self.engines = engines
        self.sampling_tasks = sampling_tasks
        self.session_idle_timeout = session_idle_timeout
        self.session_id_factory = session_id_factory
        self.clock = clock
        self.ensure_sampling_pool = ensure_sampling_pool
        self.sampling_task_stores = sampling_task_stores
        self.read_checkpoint_metadata = read_checkpoint_metadata
        self.reconcile_trainers = reconcile_trainers
        self.prepare_model = prepare_model
        self.trainer_autoscaling = trainer_autoscaling
        self.list_checkpoints = list_checkpoints
        self.delete_checkpoint = delete_checkpoint
        self.checkpoint_root = checkpoint_root

    async def create_session(
        self,
        *,
        tags: tuple[str, ...] = (),
        user_metadata: dict[str, str] | None = None,
        sdk_version: str | None = None,
    ) -> SessionRecord:
        now = self.clock()
        session = SessionRecord(
            session_id=self.session_id_factory(),
            created_at=now,
            tags=tags,
            user_metadata=user_metadata,
            sdk_version=sdk_version,
        )
        await self.kv.put_if_absent(
            session_key(session.session_id),
            session.model_dump(mode="json"),
        )
        await self.kv.put(
            session_last_seen_key(session.session_id),
            SessionLastSeenRecord(
                session_id=session.session_id,
                seen_at=now,
            ).model_dump(mode="json"),
        )
        return session

    async def heartbeat(self, session_id: str) -> SessionLastSeenRecord:
        return await self._touch_session(session_id)

    async def close_session(
        self,
        session_id: str,
        reason: str,
    ) -> SessionClosedRecord:
        if await self.kv.get(session_key(session_id)) is None:
            raise RecordNotFound("session", session_id)
        closed = SessionClosedRecord(
            session_id=session_id,
            reason=reason,
            closed_at=self.clock(),
        )
        inserted = await self.kv.put_if_absent(
            session_closed_key(session_id),
            closed.model_dump(mode="json"),
        )
        await self._unload_session_models(session_id)
        return SessionClosedRecord.model_validate(inserted.value)

    async def create_model(
        self,
        *,
        session_id: str,
        model_seq_id: int,
        definition_id: str,
        spec: dict[str, object],
        model_id: str | None = None,
    ) -> ModelCreation:
        await self._touch_session(session_id)
        mark = fingerprint(
            "create_model",
            {
                "definition_id": definition_id,
                "spec": spec,
            },
        )
        anchor = ModelCreationRecord(
            session_id=session_id,
            model_seq_id=model_seq_id,
            model_id=model_id or self._model_id(session_id, model_seq_id),
            fingerprint=mark,
            created_at=self.clock(),
        )
        inserted = await self.kv.put_if_absent(
            model_creation_key(session_id, model_seq_id),
            anchor.model_dump(mode="json"),
        )
        stored = ModelCreationRecord.model_validate(inserted.value)
        if stored.fingerprint != mark:
            raise SequenceConflict(session_id, model_seq_id)

        model = ModelRecord(
            model_id=stored.model_id,
            session_id=session_id,
            model_seq_id=model_seq_id,
            engine_definition_id=definition_id,
            spec=spec,
            created_at=stored.created_at,
        )
        model_insert = await self.kv.put_if_absent(
            model_key(model.model_id),
            model.model_dump(mode="json"),
        )
        model = ModelRecord.model_validate(model_insert.value)
        if model_insert.created:
            await self.kv.put(
                trainer_demand_key(model.model_id),
                {
                    "model_id": model.model_id,
                    "definition_id": definition_id,
                    "created_at": model.created_at,
                },
            )
            if self.reconcile_trainers is not None:
                await self.reconcile_trainers(definition_id)
            if self.prepare_model is not None:
                await self.prepare_model(model)
        return ModelCreation(
            model,
            request_id_for(model.model_id, 0),
            inserted.created,
        )

    async def checkpoints(
        self,
        training_run_id: str | None,
    ) -> list[dict[str, object]]:
        if self.list_checkpoints is None:
            raise RecordUnavailable("checkpoint", "listing", "unconfigured")
        if training_run_id is not None:
            path_component(training_run_id, "training_run_id")
        entries = [
            dict(entry) for entry in await self.list_checkpoints(training_run_id)
        ]
        entries.sort(key=lambda entry: float(entry["time"]), reverse=True)
        return entries

    async def checkpoint(
        self, training_run_id: str, checkpoint_id: str
    ) -> dict[str, object]:
        name = path_component(checkpoint_id.removeprefix("weights/"), "checkpoint_id")
        for entry in await self.checkpoints(training_run_id):
            if entry["name"] == name:
                return entry
        raise RecordNotFound(
            "checkpoint", checkpoint_tinker_path(training_run_id, name)
        )

    def checkpoint_record(self, entry: Mapping[str, object]) -> dict[str, object]:
        model_id, name = str(entry["model_id"]), str(entry["name"])
        return {
            "checkpoint_id": f"weights/{name}",
            "checkpoint_type": "training",
            "time": datetime.fromtimestamp(float(entry["time"]), UTC).isoformat(),
            "tinker_path": checkpoint_tinker_path(model_id, name),
            "size_bytes": entry.get("size_bytes"),
        }

    async def training_runs(self) -> list[dict[str, object]]:
        runs: dict[str, list[dict[str, object]]] = defaultdict(list)
        for entry in await self.checkpoints(None):
            runs[str(entry["model_id"])].append(entry)
        return [self._training_run(entries) for entries in runs.values()]

    async def training_run(self, training_run_id: str) -> dict[str, object]:
        entries = await self.checkpoints(training_run_id)
        if not entries:
            raise RecordNotFound("training run", training_run_id)
        return self._training_run(entries)

    def _training_run(self, entries: list[dict[str, object]]) -> dict[str, object]:
        latest = self.checkpoint_record(entries[0])
        metadata = next(
            (
                entry["metadata"]
                for entry in entries
                if isinstance(entry.get("metadata"), Mapping)
            ),
            {},
        )
        lora_config = metadata.get("lora_config") or {}
        parameterization = metadata.get("parameterization") or {}
        return {
            "training_run_id": str(entries[0]["model_id"]),
            "base_model": metadata.get("base_model") or "",
            "model_owner": "lilo",
            "is_lora": parameterization.get("type") == "lora",
            "lora_rank": lora_config.get("rank"),
            "last_request_time": latest["time"],
            "last_checkpoint": latest,
        }

    async def remove_checkpoint(self, training_run_id: str, checkpoint_id: str) -> None:
        if self.delete_checkpoint is None:
            raise RecordUnavailable(
                "checkpoint", checkpoint_id, "deletion unconfigured"
            )
        entry = await self.checkpoint(training_run_id, checkpoint_id)
        await self.delete_checkpoint(str(entry["path"]))

    def resolve_checkpoint_path(self, path: str) -> str:
        parts = path.removeprefix("tinker://").split("/")
        if not path.startswith("tinker://") or len(parts) != 3 or parts[1] != "weights":
            raise ValueError(f"invalid checkpoint path: {path}")
        path_component(parts[0], "training_run_id")
        path_component(parts[2], "checkpoint name")
        return f"{self.checkpoint_root}/{'/'.join(parts)}"

    def tinker_path(self, uri: str) -> str:
        return f"tinker://{PurePosixPath(uri).relative_to(self.checkpoint_root)}"

    async def checkpoint_metadata(self, path: str) -> dict[str, object]:
        if self.read_checkpoint_metadata is None:
            raise RecordUnavailable("checkpoint", path, "metadata unavailable")
        path = self.resolve_checkpoint_path(path)
        try:
            metadata = dict(await self.read_checkpoint_metadata(path))
        except FileNotFoundError:
            raise RecordNotFound("checkpoint metadata", path) from None
        schema_version = metadata.get("schema_version")
        if isinstance(schema_version, bool) or schema_version != 1:
            raise ValueError("unsupported checkpoint metadata")
        saved_base_model = metadata.get("base_model")
        definition_id = metadata.get("engine_definition_id")
        parameterization = metadata.get("parameterization")
        lora_config = metadata.get("lora_config")
        if not isinstance(saved_base_model, str) or not saved_base_model:
            raise ValueError("checkpoint metadata has no base_model")
        if not isinstance(definition_id, str) or not definition_id:
            raise ValueError("checkpoint metadata has no engine_definition_id")
        if (
            not isinstance(parameterization, Mapping)
            or parameterization.get("type") not in {"lora", "full"}
        ):
            raise ValueError("checkpoint metadata has invalid parameterization")
        if parameterization["type"] == "lora" and not isinstance(
            lora_config, Mapping
        ):
            raise ValueError("checkpoint metadata has no lora_config")
        if parameterization["type"] == "full" and lora_config is not None:
            raise ValueError("full checkpoint metadata has lora_config")
        return metadata

    async def create_model_from_checkpoint(
        self,
        *,
        session_id: str,
        model_seq_id: int,
        path: str,
        base_model: str | None,
        user_metadata: dict[str, object] | None,
        optimizer: bool,
        definition_ids: Collection[str] | None = None,
    ) -> ModelCreation:
        metadata = await self.checkpoint_metadata(path)
        path = self.resolve_checkpoint_path(path)
        saved_base_model = metadata["base_model"]
        definition_id = metadata["engine_definition_id"]
        parameterization = metadata["parameterization"]
        lora_config = metadata["lora_config"]
        if base_model is not None and base_model != saved_base_model:
            raise ValueError("base_model does not match checkpoint")
        if definition_ids is not None and definition_id not in definition_ids:
            raise ValueError(
                f"checkpoint definition {definition_id} is not deployed"
            )
        return await self.create_model(
            session_id=session_id,
            model_seq_id=model_seq_id,
            definition_id=definition_id,
            spec={
                "base_model": saved_base_model,
                "lora_config": dict(lora_config) if lora_config is not None else None,
                "parameterization": dict(parameterization),
                "user_metadata": user_metadata,
                "checkpoint": {
                    "uri": path,
                    "restore_optimizer": optimizer,
                },
            },
            model_id=f"{session_id}:train:{model_seq_id}",
        )

    async def submit_sampler_export(self, request: dict) -> str:
        model_id = str(request["model_id"])
        seq_id = request["seq_id"]
        name = request.get("path")
        sampling_session_seq_id = request.get("sampling_session_seq_id")
        ttl_seconds = request.get("ttl_seconds")
        if isinstance(seq_id, bool) or not isinstance(seq_id, int) or seq_id <= 0:
            raise ValueError("seq_id must be a positive integer")
        if name is not None and (
            not isinstance(name, str)
            or not name
            or name != name.strip()
            or name in {".", "..", "latest"}
            or "/" in name
            or "\\" in name
            or "\0" in name
        ):
            raise ValueError("path must be a non-empty checkpoint name")
        if sampling_session_seq_id is not None and (
            isinstance(sampling_session_seq_id, bool)
            or not isinstance(sampling_session_seq_id, int)
            or sampling_session_seq_id < 0
        ):
            raise ValueError("sampling_session_seq_id must be a non-negative integer")
        if (name is None) == (sampling_session_seq_id is None):
            raise ValueError(
                "exactly one of path and sampling_session_seq_id is required"
            )
        if ttl_seconds is not None and (
            isinstance(ttl_seconds, bool)
            or not isinstance(ttl_seconds, int)
            or ttl_seconds <= 0
        ):
            raise ValueError("ttl_seconds must be a positive integer")
        model = await self.get_model(model_id)
        await self._open_session(model.session_id)
        payload = {
            "path": name,
            "sampling_session_seq_id": sampling_session_seq_id,
            "ttl_seconds": ttl_seconds,
        }
        mark = fingerprint("save_weights_for_sampler", payload)
        anchor = SamplerExportSubmissionRecord(
            model_id=model_id,
            seq_id=seq_id,
            name=name,
            sampling_session_seq_id=sampling_session_seq_id,
            ttl_seconds=ttl_seconds,
            fingerprint=mark,
            created_at=self.clock(),
        )
        inserted = await self.kv.put_if_absent(
            sampler_export_submission_key(model_id, seq_id),
            anchor.model_dump(mode="json"),
        )
        stored = SamplerExportSubmissionRecord.model_validate(inserted.value)
        if stored.fingerprint != mark:
            sdk_retry = (
                stored.name is None
                and name is None
                and stored.ttl_seconds == ttl_seconds
            )
            if not sdk_retry:
                raise SequenceConflict(model_id, seq_id)
        if not inserted.created and await self._completed_sampler_export(stored):
            return request_id_for(model_id, seq_id)
        payload = {
            "path": stored.name,
            "sampling_session_seq_id": stored.sampling_session_seq_id,
            "ttl_seconds": stored.ttl_seconds,
        }
        engine = await self.engine_for(model_id)
        return await engine.save_weights_for_sampler(
            {
                "type": "save_weights_for_sampler",
                "model_id": model_id,
                "seq_id": seq_id,
                "publish_version": seq_id,
                **payload,
            }
        )

    async def create_sampling_session(
        self,
        *,
        session_id: str,
        sampling_session_seq_id: int,
        base_model: str | None = None,
        model_path: str | None = None,
        engine_definition_id: str | None = None,
    ) -> SamplingSessionRecord:
        await self._touch_session(session_id)
        mark = fingerprint(
            "create_sampling_session",
            {"base_model": base_model, "model_path": model_path},
        )
        key = sampling_session_creation_key(session_id, sampling_session_seq_id)
        value = await self.kv.get(key)
        if value is not None:
            stored = SamplingSessionCreationRecord.model_validate(value)
            if stored.session is not None and stored.fingerprint != mark:
                raise SequenceConflict(session_id, sampling_session_seq_id)
            persisted = await self.kv.get(
                sampling_session_key(stored.sampling_session_id)
            )
            if persisted is not None:
                session = SamplingSessionRecord.model_validate(persisted)
                if stored.session is None and (
                    model_path != session.model_path
                    or (base_model is not None and base_model != session.base_model)
                ):
                    raise SequenceConflict(session_id, sampling_session_seq_id)
                await self._ensure_sampling_pool(session)
                return session
            if stored.session is not None:
                session = SamplingSessionRecord.model_validate(stored.session)
                await self._validate_sampling_session(session)
                await self.kv.put(
                    sampling_session_key(session.sampling_session_id),
                    session.model_dump(mode="json"),
                )
                await self._ensure_sampling_pool(session)
                return session
        model_id = None
        publish_version = None
        export_seq_id = None
        expires_at = None
        latest = False
        if model_path is not None:
            artifact = await self.get_sampler_artifact(model_path)
            if base_model is not None and base_model != artifact.base_model:
                raise ValueError("base_model does not match model_path")
            if (
                engine_definition_id is not None
                and artifact.engine_definition_id is not None
                and engine_definition_id != artifact.engine_definition_id
            ):
                raise ValueError("engine_definition_id does not match model_path")
            base_model = artifact.base_model
            engine_definition_id = artifact.engine_definition_id or engine_definition_id
            model_id = artifact.model_id
            publish_version = artifact.publish_version
            export_seq_id = artifact.export_seq_id
            expires_at = artifact.expires_at
            latest = self._is_latest_sampler_path(model_path)
            if model_path.endswith("/sampler_weights/latest"):
                model_path = self._latest_sampler_model_path(
                    artifact.model_id,
                    artifact.publish_version,
                )
        if not base_model:
            raise ValueError("base_model or model_path is required")
        sampling_session_id = self._sampling_session_id(
            session_id,
            sampling_session_seq_id,
        )
        session = SamplingSessionRecord(
            sampling_session_id=sampling_session_id,
            session_id=session_id,
            sampling_session_seq_id=sampling_session_seq_id,
            base_model=base_model,
            engine_definition_id=engine_definition_id,
            model_path=model_path,
            model_id=model_id,
            publish_version=publish_version,
            latest=latest,
            export_seq_id=export_seq_id,
            expires_at=expires_at,
            created_at=self.clock(),
        )
        anchor = SamplingSessionCreationRecord(
            session_id=session_id,
            sampling_session_seq_id=sampling_session_seq_id,
            sampling_session_id=sampling_session_id,
            fingerprint=mark,
            created_at=session.created_at,
            session=session.model_dump(mode="json"),
        )
        inserted = await self.kv.put_if_absent(
            key,
            anchor.model_dump(mode="json"),
        )
        stored = SamplingSessionCreationRecord.model_validate(inserted.value)
        legacy_mark = fingerprint(
            "create_sampling_session",
            {
                "base_model": base_model,
                "model_path": model_path,
                "model_id": model_id,
                "publish_version": publish_version,
            },
        )
        if stored.fingerprint not in {mark, legacy_mark}:
            raise SequenceConflict(session_id, sampling_session_seq_id)
        if stored.session is not None:
            session = SamplingSessionRecord.model_validate(stored.session)
        await self._validate_sampling_session(session)
        result = await self.kv.put_if_absent(
            sampling_session_key(session.sampling_session_id),
            session.model_dump(mode="json"),
        )
        persisted = SamplingSessionRecord.model_validate(result.value)
        if persisted != session:
            raise SequenceConflict(session_id, sampling_session_seq_id)
        await self._ensure_sampling_pool(persisted)
        return persisted

    async def get_sampler_artifact(
        self,
        model_path: str,
    ) -> SamplerArtifactRecord:
        value = await self.kv.get(sampler_artifact_key(model_path))
        if value is None:
            raise RecordNotFound("sampler artifact", model_path)
        artifact = SamplerArtifactRecord.model_validate(value)
        if artifact.model_path != model_path:
            raise RecordNotFound("sampler artifact", model_path)
        self._check_expiry("sampler artifact", model_path, artifact.expires_at)
        return artifact

    def _check_expiry(
        self, kind: str, key: str, expires_at: float | None
    ) -> None:
        if expires_at is not None and expires_at <= self.clock():
            raise RecordUnavailable(kind, key, "expired")

    async def _validate_sampling_session(self, session: SamplingSessionRecord) -> None:
        self._check_expiry(
            "sampling session", session.sampling_session_id, session.expires_at
        )
        await self._open_session(session.session_id)

    async def get_sampling_session(
        self,
        sampling_session_id: str,
    ) -> SamplingSessionRecord:
        value = await self.kv.get(sampling_session_key(sampling_session_id))
        if value is None:
            raise RecordNotFound("sampling session", sampling_session_id)
        session = SamplingSessionRecord.model_validate(value)
        await self._validate_sampling_session(session)
        return session

    async def _ensure_sampling_pool(self, session: SamplingSessionRecord) -> None:
        await self._validate_sampling_session(session)
        if self.ensure_sampling_pool is not None:
            await self.ensure_sampling_pool(session)

    async def submit_sample(self, request: dict) -> str:
        if self.sampling_tasks is None:
            raise RecordUnavailable("sampling", "tasks", "unconfigured")
        sampling_session_id = str(request["sampling_session_id"])
        seq_id = int(request["seq_id"])
        session = await self.get_sampling_session(sampling_session_id)
        await self._touch_session(session.session_id)
        mark = fingerprint("sample", request)
        request_id = request_id_for(sampling_session_id, seq_id)
        key = sample_task_key(sampling_session_id, seq_id)
        task_store = self._sampling_task_store(session.session_id)
        candidate = SampleTaskRecord(
            sampling_session_id=sampling_session_id,
            seq_id=seq_id,
            request_id=request_id,
            fingerprint=mark,
            created_at=self.clock(),
        )
        inserted = await task_store.put_if_absent(
            key,
            candidate.model_dump(mode="json"),
        )
        stored = SampleTaskRecord.model_validate(inserted.value)
        if stored.fingerprint != mark:
            raise SequenceConflict(sampling_session_id, seq_id)
        if stored.task_id is not None:
            return request_id
        await self._ensure_sampling_pool(session)
        task_id = await self.sampling_tasks.submit(
            SamplingTask(
                request_id=request_id,
                session_id=session.session_id,
                sampling_session_id=sampling_session_id,
                base_model=session.base_model,
                engine_definition_id=session.engine_definition_id,
                model_path=session.model_path,
                model_id=session.model_id,
                publish_version=session.publish_version,
                payload=request,
                latest=session.latest,
            )
        )
        await task_store.put(
            key,
            stored.model_copy(update={"task_id": task_id}).model_dump(mode="json"),
        )
        return request_id

    async def engine_for(self, model_id: str) -> EngineApi:
        model = await self.get_model(model_id)
        await self._touch_session(model.session_id)
        placement = await self._placement(model_id)
        if placement is None:
            if await self._lost(model_id):
                raise ModelLost(model_id)
            raise RecordUnavailable("model", model_id, "unplaced")
        instance = await self._live_instance(placement)
        return self.engines.client(instance.instance_id)

    async def unload_model(self, model_id: str) -> str:
        model = await self._unload_model(model_id)
        if self.reconcile_trainers is not None:
            await self.reconcile_trainers(model.engine_definition_id)
        return f"{model.model_id}:unload"

    async def _unload_model(self, model_id: str) -> ModelRecord:
        model = await self.get_model(model_id)
        placement = await self._placement(model.model_id)
        if placement is not None:
            instance = await self.engines.get_instance(placement.engine_instance_id)
            if instance is not None and not instance.terminal:
                await self.engines.client(instance.instance_id).unload_model(
                    model.model_id
                )
            await self.kv.delete(placement_key(model.model_id))
        await self.kv.delete(trainer_demand_key(model.model_id))
        return model

    async def retrieve(
        self,
        request_id: str,
        timeout: float = 0.0,
    ) -> FutureResolution:
        if request_id.endswith(":unload"):
            model = await self.get_model(request_id.rsplit(":", 1)[0])
            return FutureResolution(
                request_id,
                FutureResolutionStatus.COMPLETE,
                result={"type": "unload_model", "model_id": model.model_id},
            )
        model_id, seq_id = parse_request_id(request_id)
        if model_id.startswith("sample-"):
            return await self._retrieve_sample(request_id, model_id, seq_id, timeout)
        model = await self.get_model(model_id)
        if seq_id == 0:
            return await self._retrieve_creation(request_id, model)
        export = await self._sampler_export_submission(model_id, seq_id)
        if export is not None:
            completed = await self._completed_sampler_export(export)
            if completed is not None:
                return FutureResolution(
                    request_id,
                    FutureResolutionStatus.COMPLETE,
                    result=completed,
                )
        placement = await self._placement(model_id)
        if placement is None:
            if await self.kv.get(session_closed_key(model.session_id)) is not None:
                return FutureResolution(
                    request_id,
                    FutureResolutionStatus.LOST,
                    error="session closed before producing a result",
                )
            if await self.kv.get(trainer_demand_key(model_id)) is None:
                return FutureResolution(
                    request_id,
                    FutureResolutionStatus.LOST,
                    error="model lost before producing a result",
                )
            return FutureResolution(request_id, FutureResolutionStatus.PENDING)
        instance = await self.engines.get_instance(placement.engine_instance_id)
        if instance is None or instance.terminal:
            return FutureResolution(
                request_id,
                FutureResolutionStatus.LOST,
                error="engine terminated before producing a result",
            )
        state = await self.engines.client(instance.instance_id).retrieve_future(
            request_id,
            timeout,
        )
        if state is None:
            return FutureResolution(
                request_id,
                FutureResolutionStatus.LOST,
                error="engine does not know this request",
            )
        if state.status == FutureStatus.PENDING:
            return FutureResolution(request_id, FutureResolutionStatus.PENDING)
        if state.status == FutureStatus.COMPLETE:
            if export is not None:
                try:
                    receipt = await self._persist_sampler_export_result(
                        export,
                        state.result,
                    )
                    result = await self._finalize_sampler_export(
                        model,
                        export,
                        receipt.result,
                        receipt.completed_at,
                    )
                except (TypeError, ValueError) as exc:
                    return FutureResolution(
                        request_id,
                        FutureResolutionStatus.FAILED,
                        error=str(exc),
                    )
                return FutureResolution(
                    request_id,
                    FutureResolutionStatus.COMPLETE,
                    result=result,
                )
            result = state.result
            if isinstance(result, Mapping) and result.get("type") in {
                "save_weights",
                "load_weights",
            }:
                result = {**result, "path": self.tinker_path(str(result["path"]))}
            return FutureResolution(
                request_id,
                FutureResolutionStatus.COMPLETE,
                result=result,
            )
        return FutureResolution(
            request_id,
            FutureResolutionStatus.FAILED,
            error=state.error,
        )

    async def _sampler_export_submission(
        self,
        model_id: str,
        seq_id: int,
    ) -> SamplerExportSubmissionRecord | None:
        value = await self.kv.get(sampler_export_submission_key(model_id, seq_id))
        if value is None:
            return None
        return SamplerExportSubmissionRecord.model_validate(value)

    async def _completed_sampler_export(
        self,
        export: SamplerExportSubmissionRecord,
    ) -> dict[str, str] | None:
        if export.name is not None:
            path = self._sampler_model_path(export.model_id, export.name)
            value = await self.kv.get(sampler_artifact_key(path))
            if value is not None:
                artifact = SamplerArtifactRecord.model_validate(value)
                if (
                    artifact.model_path == path
                    and artifact.model_id == export.model_id
                    and artifact.export_seq_id == export.seq_id
                ):
                    self._check_expiry("sampler artifact", path, artifact.expires_at)
                    return {"type": "save_weights_for_sampler", "path": path}
        else:
            assert export.sampling_session_seq_id is not None
            model = await self.get_model(export.model_id)
            sampling_session_id = self._sampling_session_id(
                model.session_id,
                export.sampling_session_seq_id,
            )
            value = await self.kv.get(sampling_session_key(sampling_session_id))
            if value is not None:
                session = SamplingSessionRecord.model_validate(value)
                if (
                    session.model_id == export.model_id
                    and session.export_seq_id == export.seq_id
                ):
                    await self._ensure_sampling_pool(session)
                    return {
                        "type": "save_weights_for_sampler",
                        "sampling_session_id": sampling_session_id,
                    }
        value = await self.kv.get(
            sampler_export_result_key(export.model_id, export.seq_id)
        )
        if value is None:
            return None
        receipt = SamplerExportResultRecord.model_validate(value)
        if receipt.model_id != export.model_id or receipt.seq_id != export.seq_id:
            return None
        model = await self.get_model(export.model_id)
        return await self._finalize_sampler_export(
            model,
            export,
            receipt.result,
            receipt.completed_at,
        )

    async def _persist_sampler_export_result(
        self,
        export: SamplerExportSubmissionRecord,
        result: object,
    ) -> SamplerExportResultRecord:
        if not isinstance(result, Mapping):
            raise TypeError("save_weights_for_sampler returned no publish version")
        receipt = SamplerExportResultRecord(
            model_id=export.model_id,
            seq_id=export.seq_id,
            result=dict(result),
            completed_at=self.clock(),
        )
        inserted = await self.kv.put_if_absent(
            sampler_export_result_key(export.model_id, export.seq_id),
            receipt.model_dump(mode="json"),
        )
        stored = SamplerExportResultRecord.model_validate(inserted.value)
        if (
            stored.model_id != export.model_id
            or stored.seq_id != export.seq_id
            or stored.result != receipt.result
        ):
            raise SequenceConflict(export.model_id, export.seq_id)
        return stored

    async def _finalize_sampler_export(
        self,
        model: ModelRecord,
        export: SamplerExportSubmissionRecord,
        result: object,
        completed_at: float,
    ) -> dict[str, str]:
        if not isinstance(result, Mapping):
            raise TypeError("save_weights_for_sampler returned no publish version")
        publish_version = result.get("publish_version")
        if (
            isinstance(publish_version, bool)
            or not isinstance(publish_version, int)
            or publish_version < 0
        ):
            raise ValueError(
                "save_weights_for_sampler returned an invalid publish version"
            )
        base_model = self._model_base_model(model)
        expires_at = (
            completed_at + export.ttl_seconds
            if export.ttl_seconds is not None
            else None
        )
        if export.name is not None:
            self._check_expiry(
                "sampler artifact",
                self._sampler_model_path(model.model_id, export.name),
                expires_at,
            )
        else:
            assert export.sampling_session_seq_id is not None
            self._check_expiry(
                "sampling session",
                self._sampling_session_id(model.session_id, export.sampling_session_seq_id),
                expires_at,
            )
            await self._open_session(model.session_id)
        latest_path = self._sampler_model_path(model.model_id, "latest")
        latest_version_path = self._latest_sampler_model_path(
            model.model_id,
            publish_version,
        )
        latest = SamplerArtifactRecord(
            model_path=latest_path,
            model_id=model.model_id,
            export_seq_id=export.seq_id,
            base_model=base_model,
            engine_definition_id=model.engine_definition_id,
            publish_version=publish_version,
            created_at=completed_at,
        )
        await self.kv.put(
            sampler_artifact_key(latest_path),
            latest.model_dump(mode="json"),
        )
        versioned_latest = latest.model_copy(
            update={"model_path": latest_version_path}
        )
        inserted = await self.kv.put_if_absent(
            sampler_artifact_key(latest_version_path),
            versioned_latest.model_dump(mode="json"),
        )
        if SamplerArtifactRecord.model_validate(inserted.value) != versioned_latest:
            raise SequenceConflict(model.model_id, export.seq_id)
        if export.name is not None:
            model_path = self._sampler_model_path(model.model_id, export.name)
            artifact = SamplerArtifactRecord(
                model_path=model_path,
                model_id=model.model_id,
                export_seq_id=export.seq_id,
                base_model=base_model,
                engine_definition_id=model.engine_definition_id,
                publish_version=publish_version,
                created_at=completed_at,
                expires_at=expires_at,
            )
            inserted = await self.kv.put_if_absent(
                sampler_artifact_key(model_path),
                artifact.model_dump(mode="json"),
            )
            if SamplerArtifactRecord.model_validate(inserted.value) != artifact:
                raise SequenceConflict(model.model_id, export.seq_id)
            return {
                "type": "save_weights_for_sampler",
                "path": model_path,
            }
        assert export.sampling_session_seq_id is not None
        sampling_session_id = self._sampling_session_id(
            model.session_id,
            export.sampling_session_seq_id,
        )
        session = SamplingSessionRecord(
            sampling_session_id=sampling_session_id,
            session_id=model.session_id,
            sampling_session_seq_id=export.sampling_session_seq_id,
            base_model=base_model,
            engine_definition_id=model.engine_definition_id,
            model_path=latest_version_path,
            model_id=model.model_id,
            publish_version=publish_version,
            latest=True,
            export_seq_id=export.seq_id,
            expires_at=expires_at,
            created_at=completed_at,
        )
        inserted = await self.kv.put_if_absent(
            sampling_session_key(sampling_session_id),
            session.model_dump(mode="json"),
        )
        if SamplingSessionRecord.model_validate(inserted.value) != session:
            raise SequenceConflict(
                model.session_id,
                export.sampling_session_seq_id,
            )
        await self._ensure_sampling_pool(session)
        return {
            "type": "save_weights_for_sampler",
            "sampling_session_id": sampling_session_id,
        }

    async def _retrieve_sample(
        self,
        request_id: str,
        sampling_session_id: str,
        seq_id: int,
        timeout: float,
    ) -> FutureResolution:
        session = await self.get_sampling_session(sampling_session_id)
        value = await self._sampling_task_store(session.session_id).get(
            sample_task_key(sampling_session_id, seq_id)
        )
        if value is None:
            raise RecordNotFound("future", request_id)
        record = SampleTaskRecord.model_validate(value)
        if record.request_id != request_id:
            raise RecordNotFound("future", request_id)
        if record.task_id is None:
            return FutureResolution(request_id, FutureResolutionStatus.PENDING)
        if self.sampling_tasks is None:
            return FutureResolution(
                request_id,
                FutureResolutionStatus.RETRYABLE,
                error="sampling task platform is unavailable",
            )
        state = await self.sampling_tasks.retrieve(record.task_id, timeout)
        if state is None:
            return FutureResolution(
                request_id,
                FutureResolutionStatus.RETRYABLE,
                error="sampling task was lost",
            )
        if state.status == SamplingTaskStatus.PENDING:
            return FutureResolution(request_id, FutureResolutionStatus.PENDING)
        if state.status == SamplingTaskStatus.COMPLETE:
            return FutureResolution(
                request_id,
                FutureResolutionStatus.COMPLETE,
                result=state.result,
            )
        return FutureResolution(
            request_id,
            FutureResolutionStatus.FAILED,
            error=state.error,
        )

    def _sampling_task_store(self, session_id: str) -> KeyValueStore:
        if self.sampling_task_stores is None:
            return self.kv
        return self.sampling_task_stores.for_session(session_id)

    async def _retrieve_creation(
        self,
        request_id: str,
        model: ModelRecord,
    ) -> FutureResolution:
        try:
            placement = await self._place(model)
        except ModelLost:
            return FutureResolution(
                request_id,
                FutureResolutionStatus.LOST,
                error="model lost before producing a result",
            )
        except ValueError as exc:
            return FutureResolution(
                request_id,
                FutureResolutionStatus.FAILED,
                error=str(exc),
                category="user",
            )
        if placement is None:
            return FutureResolution(request_id, FutureResolutionStatus.PENDING)
        instance = await self.engines.get_instance(placement.engine_instance_id)
        if instance is None or instance.terminal:
            return FutureResolution(
                request_id,
                FutureResolutionStatus.LOST,
                error="engine terminated before producing a result",
            )
        return FutureResolution(
            request_id,
            FutureResolutionStatus.COMPLETE,
            result={
                "type": (
                    "load_weights" if "checkpoint" in model.spec else "create_model"
                ),
                "model_id": model.model_id,
            },
        )

    async def _lost(self, model_id: str) -> bool:
        return await self.kv.get(trainer_demand_key(model_id)) is None

    async def _place(self, model: ModelRecord) -> PlacementRecord | None:
        value = await self.kv.get(placement_key(model.model_id))
        if value is not None:
            return PlacementRecord.model_validate(value)
        if await self._lost(model.model_id):
            raise ModelLost(model.model_id)
        claim_key = placement_claim_key(model.model_id)
        claim = {"token": uuid.uuid4().hex, "created_at": self.clock()}
        inserted_claim = await self.kv.put_if_absent(claim_key, claim)
        if not inserted_claim.created:
            value = inserted_claim.value
            if (
                isinstance(value, dict)
                and self.clock() - float(value.get("created_at", 0)) >= 60
            ):
                await self.kv.delete(claim_key)
            return None
        try:
            return await self._place_claimed(model)
        finally:
            if await self.kv.get(claim_key) == claim:
                await self.kv.delete(claim_key)

    async def _place_claimed(self, model: ModelRecord) -> PlacementRecord | None:
        definition_id = model.engine_definition_id
        try:
            active = await self.engines.active_instances(definition_id)
            instances = [
                instance for instance in active if instance.state == "running"
            ]
            if not active:
                if self.trainer_autoscaling(definition_id):
                    if self.reconcile_trainers is not None:
                        await self.reconcile_trainers(definition_id)
                    return None
                instance = await self.engines.ensure_instance(definition_id)
                instances = [instance] if instance.state == "running" else []
            instances.sort(
                key=lambda instance: hashlib.sha256(
                    f"{model.model_id}\0{instance.instance_id}".encode()
                ).digest(),
                reverse=True,
            )
        except Exception:
            logging.getLogger(__name__).exception(
                "list placement candidates for %s",
                model.model_id,
            )
            return None
        accepted_instance = None
        for instance in instances:
            try:
                accepted = await self.engines.client(
                    instance.instance_id
                ).accept_model(model.model_id, model.spec)
            except ValueError:
                raise
            except Exception:
                logging.getLogger(__name__).exception(
                    "place %s on %s",
                    model.model_id,
                    instance.instance_id,
                )
                continue
            if accepted:
                accepted_instance = instance
                break
        if accepted_instance is None and self.reconcile_trainers is not None:
            has_capacity = await self.reconcile_trainers(definition_id)
            if self.trainer_autoscaling(definition_id) and has_capacity:
                return None
        if accepted_instance is None and self.session_idle_timeout is not None:
            for instance in instances:
                reclaimed = await self._reclaim_idle_models(
                    instance.instance_id,
                    self.session_idle_timeout,
                )
                if not reclaimed:
                    continue
                try:
                    accepted = await self.engines.client(
                        instance.instance_id
                    ).accept_model(model.model_id, model.spec)
                except ValueError:
                    raise
                except Exception:
                    logging.getLogger(__name__).exception(
                        "place %s after reclaim",
                        model.model_id,
                    )
                    continue
                if accepted:
                    accepted_instance = instance
                    break
        if accepted_instance is None:
            return None
        record = PlacementRecord(
            model_id=model.model_id,
            engine_definition_id=definition_id,
            engine_instance_id=accepted_instance.instance_id,
            placed_at=self.clock(),
            engine_boot_id=accepted_instance.boot_id,
        )
        inserted = await self.kv.put_if_absent(
            placement_key(model.model_id),
            record.model_dump(mode="json"),
        )
        placement = PlacementRecord.model_validate(inserted.value)
        if placement.engine_instance_id != accepted_instance.instance_id:
            try:
                await self.engines.client(
                    accepted_instance.instance_id
                ).unload_model(model.model_id)
            except Exception:
                logging.getLogger(__name__).exception(
                    "release duplicate placement %s from %s",
                    model.model_id,
                    accepted_instance.instance_id,
                )
        return placement

    async def _reclaim_idle_models(
        self,
        instance_id: str,
        idle_timeout: float,
    ) -> tuple[str, ...]:
        client = self.engines.client(instance_id)
        try:
            model_ids = await client.model_ids()
        except Exception:
            logging.getLogger(__name__).exception("list models on %s", instance_id)
            return ()
        cutoff = self.clock() - idle_timeout
        reclaimed = []
        for model_id in model_ids:
            value = await self.kv.get(model_key(model_id))
            if value is not None:
                incumbent = ModelRecord.model_validate(value)
                closed = await self.kv.get(session_closed_key(incumbent.session_id))
                last_seen = await self.kv.get(
                    session_last_seen_key(incumbent.session_id)
                )
                seen_at = (
                    SessionLastSeenRecord.model_validate(last_seen).seen_at
                    if last_seen is not None
                    else 0.0
                )
                if closed is None and seen_at > cutoff:
                    continue
                if closed is None:
                    await self.kv.put_if_absent(
                        session_closed_key(incumbent.session_id),
                        SessionClosedRecord(
                            session_id=incumbent.session_id,
                            reason="idle model reclaimed for waiting session",
                            closed_at=self.clock(),
                        ).model_dump(mode="json"),
                    )
            try:
                await client.unload_model(model_id)
            except Exception:
                logging.getLogger(__name__).exception(
                    "reclaim model %s from %s",
                    model_id,
                    instance_id,
                )
                continue
            await self.kv.delete(placement_key(model_id))
            reclaimed.append(model_id)
        return tuple(reclaimed)

    async def sweep_idle_sessions(self, idle_timeout: float) -> tuple[str, ...]:
        cutoff = self.clock() - idle_timeout
        session_items = await self.kv.list_items(
            "session:",
            "session_last_seen:",
            "session_closed:",
        )
        model_items = await self.kv.list_items(
            "model:",
            "model_creation:",
            "placement:",
        )
        sessions: list[str] = []
        last_seen: dict[str, float] = {}
        closed_sessions: set[str] = set()
        for key, value in session_items:
            if key.startswith("session_last_seen:"):
                record = SessionLastSeenRecord.model_validate(value)
                last_seen[record.session_id] = record.seen_at
            elif key.startswith("session_closed:"):
                closed_sessions.add(
                    SessionClosedRecord.model_validate(value).session_id
                )
            elif key.startswith("session:"):
                sessions.append(SessionRecord.model_validate(value).session_id)

        models_by_session: dict[str, list[ModelRecord]] = defaultdict(list)
        creations_by_session: dict[str, list[str]] = defaultdict(list)
        placed: set[str] = set()
        for key, value in model_items:
            if key.startswith("model_creation:"):
                creation = ModelCreationRecord.model_validate(value)
                creations_by_session[creation.session_id].append(key)
            elif key.startswith("placement:"):
                placed.add(PlacementRecord.model_validate(value).model_id)
            elif key.startswith("model:"):
                model = ModelRecord.model_validate(value)
                models_by_session[model.session_id].append(model)

        closed: list[str] = []
        for session_id in sessions:
            if session_id in closed_sessions:
                continue
            if last_seen.get(session_id, 0.0) > cutoff:
                continue
            await self.kv.put_if_absent(
                session_closed_key(session_id),
                SessionClosedRecord(
                    session_id=session_id,
                    reason="idle timeout",
                    closed_at=self.clock(),
                ).model_dump(mode="json"),
            )
            closed_sessions.add(session_id)
            closed.append(session_id)

        for session_id in closed_sessions:
            for model in models_by_session[session_id]:
                if model.model_id not in placed:
                    continue
                await self._unload_quietly(model.model_id)
                if await self.kv.get(placement_key(model.model_id)) is None:
                    placed.discard(model.model_id)

        for session_id in closed_sessions:
            models = models_by_session[session_id]
            if any(model.model_id in placed for model in models):
                continue
            if self.sampling_task_stores is not None:
                await self.sampling_task_stores.delete_session(session_id)
            for model in models:
                await self.kv.delete(model_key(model.model_id))
                await self.kv.delete(trainer_demand_key(model.model_id))
            for key in creations_by_session[session_id]:
                await self.kv.delete(key)
            await self.kv.delete(session_last_seen_key(session_id))
            await self.kv.delete(session_key(session_id))
            await self.kv.delete(session_closed_key(session_id))

        return tuple(closed)

    async def sweep_idle_models(self, idle_timeout: float) -> tuple[str, ...]:
        reclaimed = []
        for instance in await self.engines.list_instances():
            if instance.state != "running":
                continue
            reclaimed.extend(
                await self._reclaim_idle_models(
                    instance.instance_id,
                    idle_timeout,
                )
            )
        return tuple(reclaimed)

    async def sweep_idle_engines(self) -> tuple[str, ...]:
        items = await self.kv.list_items("trainer_demand:", "placement:")
        placed = {
            PlacementRecord.model_validate(value).model_id
            for key, value in items
            if key.startswith("placement:")
        }
        demanded = {
            str(value["definition_id"])
            for key, value in items
            if key.startswith("trainer_demand:") and value["model_id"] not in placed
        }
        stopped: list[str] = []
        for instance in await self.engines.list_instances():
            if instance.terminal:
                await self.engines.stop_instance(instance.instance_id)
                continue
            if (
                instance.state not in {"running", "draining"}
                or instance.definition_id in demanded
            ):
                continue
            try:
                acknowledged = await self.engines.client(
                    instance.instance_id
                ).shutdown_if_idle()
            except Exception:
                logging.getLogger(__name__).exception(
                    "shutdown %s", instance.instance_id
                )
                continue
            if not acknowledged:
                continue
            await self.engines.stop_instance(instance.instance_id)
            stopped.append(instance.instance_id)
        return tuple(stopped)

    async def _unload_session_models(self, session_id: str) -> None:
        definitions = set()
        for _, value in await self.kv.list_items("model:"):
            model = ModelRecord.model_validate(value)
            if model.session_id == session_id:
                definitions.add(model.engine_definition_id)
                await self._unload_quietly(model.model_id)
        for definition_id in definitions:
            if self.reconcile_trainers is not None:
                await self.reconcile_trainers(definition_id)

    async def _unload_quietly(self, model_id: str) -> None:
        try:
            await self._unload_model(model_id)
        except Exception:
            logging.getLogger(__name__).exception("unload %s", model_id)

    async def _touch_session(self, session_id: str) -> SessionLastSeenRecord:
        await self._open_session(session_id)
        last_seen = SessionLastSeenRecord(
            session_id=session_id,
            seen_at=self.clock(),
        )
        await self.kv.put(
            session_last_seen_key(session_id),
            last_seen.model_dump(mode="json"),
        )
        return last_seen

    async def _open_session(self, session_id: str) -> SessionRecord:
        if await self.kv.get(session_closed_key(session_id)) is not None:
            raise RecordUnavailable("session", session_id, "closed")
        value = await self.kv.get(session_key(session_id))
        if value is None:
            raise RecordNotFound("session", session_id)
        return SessionRecord.model_validate(value)

    async def get_model(self, model_id: str) -> ModelRecord:
        value = await self.kv.get(model_key(model_id))
        if value is None:
            raise RecordNotFound("model", model_id)
        return ModelRecord.model_validate(value)

    async def _placement(self, model_id: str) -> PlacementRecord | None:
        value = await self.kv.get(placement_key(model_id))
        if value is None:
            return None
        return PlacementRecord.model_validate(value)

    async def _live_instance(self, placement: PlacementRecord) -> EngineInstance:
        instance = await self.engines.get_instance(placement.engine_instance_id)
        if (
            instance is None
            or instance.terminal
            or placement.engine_boot_id not in ("", instance.boot_id)
        ):
            await self.kv.delete(placement_key(placement.model_id))
            await self.kv.delete(trainer_demand_key(placement.model_id))
            raise ModelLost(placement.model_id)
        return instance

    def _model_id(self, session_id: str, model_seq_id: int) -> str:
        digest = hashlib.sha256(f"{session_id}\0{model_seq_id}".encode())
        return digest.hexdigest()[:32]

    def _sampling_session_id(self, session_id: str, seq_id: int) -> str:
        digest = hashlib.sha256(f"sample\0{session_id}\0{seq_id}".encode())
        return f"sample-{digest.hexdigest()[:32]}"

    def _sampler_model_path(self, model_id: str, name: str) -> str:
        training_run_id = model_id if ":train:" in model_id else f"{model_id}:train:0"
        return f"tinker://{training_run_id}/sampler_weights/{name}"

    def _latest_sampler_model_path(self, model_id: str, version: int) -> str:
        return self._sampler_model_path(model_id, f"latest/{version:06d}")

    def _is_latest_sampler_path(self, model_path: str) -> bool:
        marker = "/sampler_weights/latest"
        _, found, suffix = model_path.partition(marker)
        return bool(found) and (
            suffix == ""
            or (
                suffix.startswith("/")
                and suffix[1:].isdigit()
                and "/" not in suffix[1:]
            )
        )

    def _model_base_model(self, model: ModelRecord) -> str:
        base_model = model.spec.get("base_model")
        if not isinstance(base_model, str) or not base_model:
            raise RecordUnavailable("model", model.model_id, "missing base model")
        return base_model
