from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Identifier = Annotated[str, Field(min_length=1, pattern=r"^[^:]+$")]
ModelIdentifier = Annotated[str, Field(min_length=1)]
NonEmptyString = Annotated[str, Field(min_length=1)]
Timestamp = Annotated[float, Field(ge=0)]
PublishVersion = Annotated[int, Field(strict=True, ge=0)]
PositiveInteger = Annotated[int, Field(strict=True, gt=0)]


class DurableRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1


class SessionRecord(DurableRecord):
    session_id: Identifier
    created_at: Timestamp
    tags: tuple[str, ...] = ()
    user_metadata: dict[str, str] | None = None
    sdk_version: str | None = None


class SessionLastSeenRecord(DurableRecord):
    session_id: Identifier
    seen_at: Timestamp


class SessionClosedRecord(DurableRecord):
    session_id: Identifier
    reason: NonEmptyString
    closed_at: Timestamp


class ModelCreationRecord(DurableRecord):
    session_id: Identifier
    model_seq_id: int = Field(ge=0)
    model_id: ModelIdentifier
    fingerprint: NonEmptyString
    created_at: Timestamp


class ModelRecord(DurableRecord):
    model_id: ModelIdentifier
    session_id: Identifier
    model_seq_id: int = Field(ge=0)
    engine_definition_id: Identifier
    spec: dict[str, object]
    created_at: Timestamp


class PlacementRecord(DurableRecord):
    model_id: ModelIdentifier
    engine_definition_id: Identifier
    engine_instance_id: Identifier
    placed_at: Timestamp
    engine_boot_id: str = ""


class SamplingSessionCreationRecord(DurableRecord):
    session_id: Identifier
    sampling_session_seq_id: int = Field(ge=0)
    sampling_session_id: Identifier
    fingerprint: NonEmptyString
    created_at: Timestamp
    session: dict | None = None


class SamplingSessionRecord(DurableRecord):
    sampling_session_id: Identifier
    session_id: Identifier
    sampling_session_seq_id: int = Field(ge=0)
    base_model: NonEmptyString
    engine_definition_id: Identifier | None = None
    model_path: str | None = None
    model_id: ModelIdentifier | None = None
    publish_version: PublishVersion | None = None
    latest: bool = False
    export_seq_id: int | None = Field(default=None, gt=0)
    expires_at: Timestamp | None = None
    created_at: Timestamp


class SamplerExportSubmissionRecord(DurableRecord):
    model_id: ModelIdentifier
    seq_id: int = Field(gt=0)
    name: NonEmptyString | None = None
    sampling_session_seq_id: int | None = Field(default=None, ge=0)
    ttl_seconds: PositiveInteger | None = None
    fingerprint: NonEmptyString
    created_at: Timestamp


class SamplerExportResultRecord(DurableRecord):
    model_id: ModelIdentifier
    seq_id: int = Field(gt=0)
    result: dict
    completed_at: Timestamp


class SamplerArtifactRecord(DurableRecord):
    model_path: NonEmptyString
    model_id: ModelIdentifier
    export_seq_id: int = Field(gt=0)
    base_model: NonEmptyString
    engine_definition_id: Identifier | None = None
    publish_version: PublishVersion
    created_at: Timestamp
    expires_at: Timestamp | None = None


class SampleTaskRecord(DurableRecord):
    sampling_session_id: Identifier
    seq_id: int = Field(ge=0)
    request_id: NonEmptyString
    fingerprint: NonEmptyString
    task_id: NonEmptyString | None = None
    created_at: Timestamp
