from __future__ import annotations

from typing import Any

import httpx
import zstandard
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from lilo.errors import EngineSaturated, RecordNotFound, SequenceConflict

from .api import (
    JSON_OPERATIONS,
    EngineApi,
    FutureState,
    FutureStatus,
    OperationKind,
    submit_json_operation,
)


class AcceptModelBody(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    spec: Any


class RetrieveFutureBody(BaseModel):
    request_id: str
    timeout: float = 0.0


class UnloadModelBody(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str


class SkipSequenceBody(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    seq_id: int
    error: str


def create_engine_app(server: EngineApi, *, token: str | None = None) -> FastAPI:
    async def authorize(request: Request) -> None:
        if (
            token is not None
            and request.headers.get("authorization") != f"Bearer {token}"
        ):
            raise HTTPException(status_code=401, detail="unauthorized")

    app = FastAPI(dependencies=[Depends(authorize)])

    @app.exception_handler(RecordNotFound)
    async def not_found(request: Request, exc: RecordNotFound) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={
                "detail": {
                    "error": "not_found",
                    "kind": exc.kind,
                    "object_id": exc.object_id,
                }
            },
        )

    @app.exception_handler(SequenceConflict)
    async def conflict(request: Request, exc: SequenceConflict) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={
                "detail": {
                    "error": "sequence_conflict",
                    "object_id": exc.object_id,
                    "seq_id": exc.seq_id,
                }
            },
        )

    @app.exception_handler(EngineSaturated)
    async def saturated(request: Request, exc: EngineSaturated) -> JSONResponse:
        return JSONResponse(
            status_code=429,
            content={"detail": {"error": "saturated", "reason": exc.reason}},
        )

    @app.exception_handler(ValueError)
    async def invalid(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"detail": {"error": "invalid", "reason": str(exc)}},
        )

    @app.post("/api/v1/accept_model")
    async def accept_model(body: AcceptModelBody) -> dict[str, bool]:
        return {"accepted": await server.accept_model(body.model_id, body.spec)}

    @app.get("/api/v1/models")
    async def models() -> dict[str, tuple[str, ...]]:
        return {"model_ids": await server.model_ids()}

    @app.post("/api/v1/forward_backward")
    async def forward_backward(request: Request) -> dict[str, str]:
        body = await request.body()
        if request.headers.get("content-encoding") == "zstd":
            body = zstandard.ZstdDecompressor().decompress(body)
        request_id = await server.forward_backward(
            body,
            request.headers.get("content-type", "application/json"),
        )
        return {"request_id": request_id}

    def json_operation_route(kind: OperationKind) -> None:
        @app.post(f"/api/v1/{kind.value}")
        async def submit_operation(request: dict[str, Any]) -> dict[str, str]:
            request_id = await submit_json_operation(
                server,
                kind,
                request,
            )
            return {"request_id": request_id}

    for kind in JSON_OPERATIONS:
        json_operation_route(kind)

    @app.post("/api/v1/skip_sequence")
    async def skip_sequence(body: SkipSequenceBody) -> dict[str, str]:
        request_id = await server.skip_sequence(body.model_id, body.seq_id, body.error)
        return {"request_id": request_id}

    @app.post("/api/v1/retrieve_future")
    async def retrieve_future(body: RetrieveFutureBody) -> dict[str, object]:
        state = await server.retrieve_future(body.request_id, body.timeout)
        if state is None:
            raise HTTPException(
                status_code=404,
                detail={"error": "unknown_future"},
            )
        return {
            "status": state.status.value,
            "result": state.result,
            "error": state.error,
        }

    @app.post("/api/v1/unload_model")
    async def unload_model(body: UnloadModelBody) -> dict[str, bool]:
        await server.unload_model(body.model_id)
        return {"unloaded": True}

    @app.post("/api/v1/shutdown_if_idle")
    async def shutdown_if_idle() -> dict[str, bool]:
        return {"shutdown": await server.shutdown_if_idle()}

    return app


class HttpEngineClient:
    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {token}"} if token is not None else {}
        self.http = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            transport=transport,
            timeout=httpx.Timeout(30.0),
        )

    async def accept_model(self, model_id: str, spec: object) -> bool:
        response = await self.http.post(
            "/api/v1/accept_model",
            json={"model_id": model_id, "spec": spec},
            timeout=45.0,
        )
        _raise_mapped(response)
        return response.json()["accepted"]

    async def model_ids(self) -> tuple[str, ...]:
        response = await self.http.get("/api/v1/models")
        _raise_mapped(response)
        return tuple(response.json()["model_ids"])

    async def forward_backward(self, body: bytes, content_type: str) -> str:
        response = await self.http.post(
            "/api/v1/forward_backward",
            content=body,
            headers={"Content-Type": content_type},
        )
        _raise_mapped(response)
        return response.json()["request_id"]

    async def forward(self, request: dict) -> str:
        return await self._submit("forward", request)

    async def optim_step(self, request: dict) -> str:
        return await self._submit("optim_step", request)

    async def save_weights(self, request: dict) -> str:
        return await self._submit("save_weights", request)

    async def load_weights(self, request: dict) -> str:
        return await self._submit("load_weights", request)

    async def save_weights_for_sampler(self, request: dict) -> str:
        return await self._submit("save_weights_for_sampler", request)

    async def skip_sequence(self, model_id: str, seq_id: int, error: str) -> str:
        response = await self.http.post(
            "/api/v1/skip_sequence",
            json={"model_id": model_id, "seq_id": seq_id, "error": error},
        )
        _raise_mapped(response)
        return response.json()["request_id"]

    async def retrieve_future(
        self,
        request_id: str,
        timeout: float = 0.0,
    ) -> FutureState | None:
        response = await self.http.post(
            "/api/v1/retrieve_future",
            json={"request_id": request_id, "timeout": timeout},
            timeout=timeout + 5.0,
        )
        if _detail(response).get("error") == "unknown_future":
            return None
        _raise_mapped(response)
        body = response.json()
        return FutureState(
            status=FutureStatus(body["status"]),
            result=body["result"],
            error=body["error"],
        )

    async def unload_model(self, model_id: str) -> None:
        response = await self.http.post(
            "/api/v1/unload_model",
            json={"model_id": model_id},
        )
        _raise_mapped(response)

    async def shutdown_if_idle(self) -> bool:
        response = await self.http.post("/api/v1/shutdown_if_idle")
        _raise_mapped(response)
        return response.json()["shutdown"]

    async def close(self) -> None:
        await self.http.aclose()

    async def _submit(self, path: str, request: dict) -> str:
        response = await self.http.post(f"/api/v1/{path}", json=request)
        _raise_mapped(response)
        return response.json()["request_id"]


def _detail(response: httpx.Response) -> dict:
    if response.is_success:
        return {}
    try:
        detail = response.json().get("detail")
    except ValueError:
        return {}
    return detail if isinstance(detail, dict) else {}


def _raise_mapped(response: httpx.Response) -> None:
    if response.is_success:
        return
    detail = _detail(response)
    error = detail.get("error")
    if error == "not_found":
        raise RecordNotFound(detail["kind"], detail["object_id"])
    if error == "sequence_conflict":
        raise SequenceConflict(detail["object_id"], detail["seq_id"])
    if error == "saturated":
        raise EngineSaturated(detail["reason"])
    if error == "invalid":
        raise ValueError(detail["reason"])
    response.raise_for_status()
