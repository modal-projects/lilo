from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterable, Mapping
from pathlib import PurePosixPath
from typing import Any

import httpx
import zstandard
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from lilo.engine.api import JSON_OPERATIONS, OperationKind, submit_json_operation
from lilo.errors import (
    EngineSaturated,
    ModelLost,
    RecordNotFound,
    RecordUnavailable,
    SequenceConflict,
)
from lilo.proto import tinker_public_pb2
from lilo.providers.contracts import Parameterization

from .service import ControlPlane, FutureResolutionStatus

ERROR_STATUSES: tuple[tuple[type[Exception], int, str], ...] = (
    (ModelLost, 410, "model_lost"),
    (EngineSaturated, 429, "saturated"),
    (SequenceConflict, 409, "conflict"),
    (RecordUnavailable, 409, "conflict"),
    (RecordNotFound, 404, "not_found"),
    (RequestValidationError, 400, "invalid_request"),
    (ValueError, 400, "invalid_request"),
    (httpx.TransportError, 503, "engine_unreachable"),
)


class CreateSessionBody(BaseModel):
    tags: tuple[str, ...] = ()
    user_metadata: dict[str, str] | None = None
    sdk_version: str | None = None


class SessionHeartbeatBody(BaseModel):
    session_id: str


class RolloutPoolBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_containers: int | None = Field(default=None, ge=0)
    max_containers: int | None = Field(default=None, ge=1)
    scaledown_window: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _ordered(self) -> RolloutPoolBody:
        if None not in (self.min_containers, self.max_containers) and (
            self.min_containers > self.max_containers
        ):
            raise ValueError("min_containers cannot exceed max_containers")
        return self


class CreateModelBody(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    session_id: str
    model_seq_id: int
    base_model: str
    lora_config: dict[str, Any] | None = None
    parameterization: dict[str, str] | None = None
    rollout: RolloutPoolBody | None = None
    user_metadata: dict[str, str] | None = None


class ModelIdBody(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str


class OperationEnvelope(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    model_id: str
    seq_id: int


class LoadWeightsBody(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    model_id: str | None = None
    seq_id: int | None = None
    session_id: str | None = None
    model_seq_id: int | None = None
    base_model: str | None = None
    user_metadata: dict[str, Any] | None = None
    path: str
    optimizer: bool = False
    weights_access_token: str | None = None


class WeightsInfoBody(BaseModel):
    tinker_path: str


class RetrieveFutureBody(BaseModel):
    request_id: str
    allow_metadata_only: bool = False


class CreateSamplingSessionBody(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    session_id: str
    sampling_session_seq_id: int
    base_model: str | None = None
    model_path: str | None = None


class SampleEnvelope(BaseModel):
    model_config = ConfigDict(extra="allow")

    sampling_session_id: str
    seq_id: int
    cache_affinity_key: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
    )

    @model_validator(mode="after")
    def _nonblank_cache_affinity_key(self) -> SampleEnvelope:
        if self.cache_affinity_key is not None and not self.cache_affinity_key.strip():
            raise ValueError("cache_affinity_key must not be blank")
        return self


class SaveWeightsForSamplerBody(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    model_id: str
    seq_id: int
    path: str | None = None
    sampling_session_seq_id: int | None = None
    ttl_seconds: int | None = None


def create_control_plane_app(
    control_plane: ControlPlane,
    definitions: Iterable[Any],
    *,
    api_key: str | None = None,
    retrieve_window: float = 30.0,
    checkpoint_volume: str = "lilo-checkpoints",
) -> FastAPI:
    definitions = tuple(
        definition
        for definition in definitions
        if definition.CATALOG_VISIBLE
    )

    def definition_for(
        model_name: str,
        parameterization: Parameterization,
    ) -> str | None:
        matches = [
            definition.DEFINITION_ID
            for definition in definitions
            if definition.MODEL_NAME == model_name
            and definition.PARAMETERIZATION == parameterization
        ]
        if len(matches) > 1:
            raise ValueError(
                f"duplicate {parameterization} definition for {model_name}"
            )
        return matches[0] if matches else None

    def supports_model(model_name: str) -> bool:
        return any(
            definition.MODEL_NAME == model_name for definition in definitions
        )

    async def authorize(request: Request) -> None:
        if api_key is not None and request.headers.get("x-api-key") != api_key:
            raise HTTPException(status_code=401, detail="unauthorized")

    app = FastAPI(dependencies=[Depends(authorize)])

    async def skip_rejected_operation(request: Request, exc: Exception) -> None:
        path = request.url.path
        if api_key is not None and request.headers.get("x-api-key") != api_key:
            return
        if path == "/api/v1/forward_backward":
            if request.headers.get("content-type", "").startswith(
                "application/x-protobuf"
            ):
                message = tinker_public_pb2.ForwardBackwardRequest()
                message.ParseFromString(await request.body())
                body = {"model_id": message.model_id, "seq_id": message.seq_id}
            else:
                body = await request.json()
        elif path in {f"/api/v1/{kind.value}" for kind in JSON_OPERATIONS}:
            body = (
                exc.body
                if isinstance(exc, RequestValidationError)
                else await request.json()
            )
        else:
            return
        if not isinstance(body, Mapping):
            return
        model_id = body.get("model_id")
        seq_id = body.get("seq_id")
        if (
            not isinstance(model_id, str)
            or isinstance(seq_id, bool)
            or not isinstance(seq_id, int)
            or seq_id <= 0
        ):
            return
        engine = await control_plane.engine_for(model_id)
        await engine.skip_sequence(model_id, seq_id, str(exc))

    async def handle_error(request: Request, exc: Exception) -> JSONResponse:
        for kind, status_code, error in ERROR_STATUSES:
            if isinstance(exc, kind):
                if isinstance(exc, RequestValidationError | ValueError):
                    try:
                        await skip_rejected_operation(request, exc)
                    except Exception:
                        logging.getLogger(__name__).exception(
                            "skip rejected operation on %s", request.url.path
                        )
                return JSONResponse(
                    status_code=status_code,
                    content={"error": error, "message": str(exc)},
                )
        error_id = uuid.uuid4().hex
        logging.getLogger(__name__).exception(
            "unhandled error %s on %s", error_id, request.url.path
        )
        return JSONResponse(
            status_code=500,
            content={"error": "internal", "message": str(exc), "error_id": error_id},
        )

    for kind in (*(entry[0] for entry in ERROR_STATUSES), Exception):
        app.add_exception_handler(kind, handle_error)

    @app.exception_handler(HTTPException)
    async def handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
        error = {401: "unauthorized", 405: "unsupported"}.get(
            exc.status_code, "invalid_request"
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": error, "message": str(exc.detail)},
        )

    @app.get("/api/v1/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/get_server_capabilities")
    async def get_server_capabilities() -> dict[str, object]:
        return {
            "supported_models": [
                {
                    "model_name": name,
                    "max_context_length": min(
                        definition.MAX_CONTEXT_LENGTH
                        for definition in definitions
                        if definition.MODEL_NAME == name
                    ),
                }
                for name in dict.fromkeys(
                    definition.MODEL_NAME for definition in definitions
                )
            ],
        }

    @app.post("/api/v1/client/config")
    async def client_config() -> dict[str, object]:
        return {
            "pjwt_auth_enabled": False,
            "credential_default_source": "api_key",
            "parallel_fwdbwd_chunks": True,
            "proto_write_fwdbwd": False,
            "use_pyqwest_transport": False,
            "create_model_via_load_weights": True,
        }

    @app.post("/api/v1/telemetry")
    async def telemetry() -> dict[str, str]:
        return {"status": "accepted"}

    @app.post("/api/v1/create_session")
    async def create_session(body: CreateSessionBody) -> dict[str, object]:
        session = await control_plane.create_session(
            tags=body.tags,
            user_metadata=body.user_metadata,
            sdk_version=body.sdk_version,
        )
        return {"type": "create_session", "session_id": session.session_id}

    @app.post("/api/v1/session_heartbeat")
    async def session_heartbeat(body: SessionHeartbeatBody) -> dict[str, str]:
        await control_plane.heartbeat(body.session_id)
        return {"type": "session_heartbeat"}

    @app.post("/api/v1/create_model")
    async def create_model(body: CreateModelBody) -> dict[str, object]:
        if not supports_model(body.base_model):
            raise HTTPException(
                status_code=400,
                detail=f"unsupported base_model: {body.base_model}",
            )
        if body.parameterization is None:
            parameterization: Parameterization | None = (
                "lora" if body.lora_config is not None else None
            )
        else:
            if body.lora_config is not None:
                raise ValueError(
                    "parameterization and lora_config cannot both be provided"
                )
            if set(body.parameterization) != {"type"}:
                raise ValueError("parameterization must contain only type")
            value = body.parameterization["type"]
            if value not in ("lora", "full"):
                raise ValueError(f"unsupported parameterization type: {value}")
            parameterization = value
        if parameterization is None:
            raise ValueError("parameterization is required when lora_config is absent")
        if parameterization == "lora" and body.lora_config is None:
            raise ValueError("lora parameterization is configured through lora_config")
        if body.rollout is not None and parameterization != "full":
            raise ValueError("rollout is only configurable for full parameterization")
        definition_id = definition_for(body.base_model, parameterization)
        if definition_id is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{parameterization} parameterization is not available for "
                    f"{body.base_model}"
                ),
            )
        creation = await control_plane.create_model(
            session_id=body.session_id,
            model_seq_id=body.model_seq_id,
            definition_id=definition_id,
            spec={
                "base_model": body.base_model,
                "lora_config": body.lora_config,
                "parameterization": {"type": parameterization},
                "rollout": body.rollout and body.rollout.model_dump(exclude_none=True),
                "user_metadata": body.user_metadata,
            },
        )
        return {
            "request_id": creation.request_id,
            "model_id": creation.model.model_id,
        }

    @app.post("/api/v1/create_sampling_session")
    async def create_sampling_session(
        body: CreateSamplingSessionBody,
    ) -> dict[str, str]:
        if body.base_model is None and body.model_path is None:
            raise HTTPException(
                status_code=400,
                detail="base_model or model_path is required",
            )
        definition_id = (
            definition_for(body.base_model, "full")
            or definition_for(body.base_model, "lora")
            if body.base_model
            else None
        )
        if body.model_path is None and definition_id is None:
            raise HTTPException(
                status_code=400,
                detail=f"unsupported base_model: {body.base_model}",
            )
        session = await control_plane.create_sampling_session(
            session_id=body.session_id,
            sampling_session_seq_id=body.sampling_session_seq_id,
            base_model=body.base_model,
            model_path=body.model_path,
            engine_definition_id=definition_id,
        )
        if not supports_model(session.base_model):
            raise HTTPException(
                status_code=400,
                detail=f"unsupported base_model: {session.base_model}",
            )
        return {
            "type": "create_sampling_session",
            "sampling_session_id": session.sampling_session_id,
        }

    @app.post("/api/v1/asample")
    async def asample(body: SampleEnvelope) -> dict[str, str]:
        request = body.model_dump(mode="json")
        if body.cache_affinity_key is None:
            request.pop("cache_affinity_key")
        request_id = await control_plane.submit_sample(request)
        return {"request_id": request_id}

    @app.get("/api/v1/samplers/{sampling_session_id}")
    async def get_sampler(sampling_session_id: str) -> dict[str, object]:
        session = await control_plane.get_sampling_session(sampling_session_id)
        return {
            "sampler_id": session.sampling_session_id,
            "base_model": session.base_model,
            "model_path": session.model_path,
        }

    @app.post("/api/v1/unload_model")
    async def unload_model(body: ModelIdBody) -> dict[str, str]:
        request_id = await control_plane.unload_model(body.model_id)
        return {"request_id": request_id, "model_id": body.model_id}

    @app.post("/api/v1/get_info")
    async def get_info(body: ModelIdBody) -> dict[str, object]:
        model = await control_plane.get_model(body.model_id)
        spec = model.spec if isinstance(model.spec, dict) else {}
        lora_config = spec.get("lora_config") or {}
        parameterization = spec.get("parameterization") or {"type": "lora"}
        return {
            "type": "get_info",
            "model_id": model.model_id,
            "model_data": {"model_name": spec.get("base_model")},
            "model_name": spec.get("base_model"),
            "is_lora": parameterization["type"] == "lora",
            "lora_rank": lora_config.get("rank"),
            "parameterization": parameterization,
        }

    def operation_route(kind: OperationKind) -> None:
        @app.post(f"/api/v1/{kind.value}")
        async def submit_operation(body: OperationEnvelope) -> dict[str, str]:
            engine = await control_plane.engine_for(body.model_id)
            request_id = await submit_json_operation(
                engine,
                kind,
                body.model_dump(mode="json"),
            )
            return {"request_id": request_id, "model_id": body.model_id}

    @app.post("/api/v1/load_weights")
    async def load_weights(body: LoadWeightsBody) -> dict[str, str]:
        if body.weights_access_token is not None:
            raise ValueError("weights_access_token is not supported")
        create_via_load = body.session_id is not None or body.model_seq_id is not None
        if create_via_load:
            if (
                body.session_id is None
                or body.model_seq_id is None
                or body.model_id is not None
                or body.seq_id is not None
            ):
                raise ValueError(
                    "session_id and model_seq_id must be provided without model_id and seq_id"
                )
            creation = await control_plane.create_model_from_checkpoint(
                session_id=body.session_id,
                model_seq_id=body.model_seq_id,
                path=body.path,
                base_model=body.base_model,
                user_metadata=body.user_metadata,
                optimizer=body.optimizer,
                definition_ids={d.DEFINITION_ID for d in definitions},
            )
            return {
                "request_id": creation.request_id,
                "model_id": creation.model.model_id,
            }
        if body.model_id is None or body.seq_id is None:
            raise ValueError("model_id and seq_id are required")
        engine = await control_plane.engine_for(body.model_id)
        body.path = control_plane.resolve_checkpoint_path(body.path)
        request_id = await engine.load_weights(
            body.model_dump(mode="json", exclude_none=True)
        )
        return {"request_id": request_id, "model_id": body.model_id}

    @app.post("/api/v1/weights_info")
    async def weights_info(body: WeightsInfoBody) -> dict[str, object]:
        metadata = await control_plane.checkpoint_metadata(body.tinker_path)
        parameterization = metadata["parameterization"]
        lora_config = metadata["lora_config"]
        assert isinstance(parameterization, dict)
        assert lora_config is None or isinstance(lora_config, dict)
        return {
            "base_model": metadata["base_model"],
            "is_lora": parameterization["type"] == "lora",
            "lora_rank": lora_config.get("rank") if lora_config else None,
            "train_unembed": (
                lora_config.get("train_unembed") if lora_config else None
            ),
            "train_mlp": lora_config.get("train_mlp") if lora_config else None,
            "train_attn": lora_config.get("train_attn") if lora_config else None,
        }

    @app.post("/api/v1/save_weights_for_sampler")
    async def save_weights_for_sampler(
        body: SaveWeightsForSamplerBody,
    ) -> dict[str, str]:
        request_id = await control_plane.submit_sampler_export(
            body.model_dump(mode="json")
        )
        return {"request_id": request_id, "model_id": body.model_id}

    @app.post("/api/v1/forward_backward")
    async def forward_backward(request: Request) -> dict[str, str]:
        body = await request.body()
        if request.headers.get("content-encoding") == "zstd":
            body = zstandard.ZstdDecompressor().decompress(body)
        content_type = request.headers.get("content-type", "application/json")
        if content_type.startswith("application/x-protobuf"):
            message = tinker_public_pb2.ForwardBackwardRequest()
            message.ParseFromString(body)
            model_id = message.model_id
        else:
            model_id = json.loads(body)["model_id"]
        engine = await control_plane.engine_for(model_id)
        request_id = await engine.forward_backward(body, content_type)
        return {"request_id": request_id, "model_id": model_id}

    for kind in JSON_OPERATIONS:
        if kind not in {
            OperationKind.LOAD_WEIGHTS,
            OperationKind.SAVE_WEIGHTS_FOR_SAMPLER,
        }:
            operation_route(kind)

    @app.post("/api/v1/retrieve_future")
    async def retrieve_future(body: RetrieveFutureBody) -> JSONResponse:
        resolution = await control_plane.retrieve(
            body.request_id,
            timeout=retrieve_window,
        )
        if resolution.status == FutureResolutionStatus.PENDING:
            return JSONResponse(
                status_code=408,
                content={
                    "type": "try_again",
                    "request_id": body.request_id,
                    "queue_state": "active",
                },
            )
        if resolution.status == FutureResolutionStatus.COMPLETE:
            return JSONResponse(resolution.result)
        if resolution.status == FutureResolutionStatus.RETRYABLE:
            return JSONResponse(
                status_code=410,
                content={
                    "error": "request_lost",
                    "message": resolution.error or "request lost",
                },
            )
        return JSONResponse(
            {
                "error": resolution.error or "request failed",
                "category": resolution.category,
            }
        )

    @app.get("/api/v1/training_runs")
    async def list_training_runs(limit: int = 20, offset: int = 0) -> dict[str, object]:
        runs = await control_plane.training_runs()
        return {
            "training_runs": runs[offset : offset + limit],
            "cursor": {"offset": offset, "limit": limit, "total_count": len(runs)},
        }

    @app.get("/api/v1/training_runs/{training_run_id}")
    async def get_training_run(training_run_id: str) -> dict[str, object]:
        return await control_plane.training_run(training_run_id)

    @app.get("/api/v1/training_runs/{training_run_id}/checkpoints")
    async def list_checkpoints(training_run_id: str) -> dict[str, object]:
        return {
            "checkpoints": [
                control_plane.checkpoint_record(entry)
                for entry in await control_plane.checkpoints(training_run_id)
            ]
        }

    @app.delete(
        "/api/v1/training_runs/{training_run_id}/checkpoints/{checkpoint_id:path}"
    )
    async def delete_checkpoint(training_run_id: str, checkpoint_id: str) -> None:
        await control_plane.remove_checkpoint(training_run_id, checkpoint_id)

    @app.get(
        "/api/v1/training_runs/{training_run_id}/checkpoints/{checkpoint_id:path}/archive"
    )
    async def get_checkpoint_archive_url(
        training_run_id: str, checkpoint_id: str
    ) -> None:
        entry = await control_plane.checkpoint(training_run_id, checkpoint_id)
        relative = PurePosixPath(str(entry["path"])).relative_to(
            control_plane.checkpoint_root
        )
        raise HTTPException(
            status_code=405,
            detail=(
                "archive downloads are not served; copy the checkpoint directly with "
                f"`modal volume get {checkpoint_volume} /{relative} <dest>`"
            ),
        )

    return app
