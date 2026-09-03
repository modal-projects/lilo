from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from .api import Execution, Executor, OperationKind
from .operations import (
    OperationPayload,
    parse_model_spec,
    parse_operation_payload,
    serialize_operation_payload,
)


class ModelBody(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    spec: Any = None


class ExecuteBody(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    kind: OperationKind
    payload: Any


class ExecuteBatchBody(BaseModel):
    executions: tuple[ExecuteBody, ...]


class PersistedOperationBody(ExecuteBody):
    capture: Any = None


def create_backend_app(executor: Executor) -> FastAPI:
    app = FastAPI()
    app.state.executor_closed = False

    async def run(action: Awaitable[object]) -> JSONResponse:
        try:
            return JSONResponse({"result": await action})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/accept_model")
    async def accept_model(body: ModelBody) -> JSONResponse:
        return await run(
            executor.accept_model(body.model_id, parse_model_spec(body.spec))
        )

    @app.post("/execute")
    async def execute(body: ExecuteBody) -> JSONResponse:
        return await run(
            executor.execute(
                body.model_id,
                body.kind,
                parse_operation_payload(body.kind, body.payload),
            )
        )

    @app.post("/execute_batch")
    async def execute_batch(body: ExecuteBatchBody) -> JSONResponse:
        return await run(
            executor.execute_batch(
                tuple(
                    Execution(
                        item.model_id,
                        item.kind,
                        parse_operation_payload(item.kind, item.payload),
                    )
                    for item in body.executions
                )
            )
        )

    @app.post("/capture_operation")
    async def capture_operation(body: PersistedOperationBody) -> JSONResponse:
        return await run(
            executor.capture_operation(
                body.model_id,
                body.kind,
                parse_operation_payload(body.kind, body.payload),
            )
        )

    @app.post("/persist_operation")
    async def persist_operation(body: PersistedOperationBody) -> JSONResponse:
        return await run(
            executor.persist_operation(
                body.model_id,
                body.kind,
                parse_operation_payload(body.kind, body.payload),
                body.capture,
            )
        )

    @app.post("/unload_model")
    async def unload_model(body: ModelBody) -> JSONResponse:
        return await run(executor.unload_model(body.model_id))

    @app.post("/close")
    async def close() -> JSONResponse:
        if app.state.executor_closed:
            return JSONResponse({"result": None})
        response = await run(executor.close())
        if response.status_code < 400:
            app.state.executor_closed = True
        return response

    return app


class HttpExecutor:
    def __init__(
        self,
        base_url: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        read_timeout: float | None = None,
        on_read_timeout: Callable[[], None] | None = None,
    ) -> None:
        self.read_timeout = read_timeout
        self.on_read_timeout = on_read_timeout
        self.http = httpx.AsyncClient(
            base_url=base_url,
            transport=transport,
            timeout=httpx.Timeout(30.0, read=read_timeout),
        )

    async def accept_model(self, model_id: str, spec: object) -> None:
        await self._post("/accept_model", {"model_id": model_id, "spec": spec})

    async def execute(
        self,
        model_id: str,
        kind: OperationKind,
        payload: OperationPayload,
    ) -> object:
        return await self._post(
            "/execute",
            {
                "model_id": model_id,
                "kind": kind.value,
                "payload": serialize_operation_payload(payload),
            },
        )

    async def execute_batch(
        self,
        executions: tuple[Execution, ...],
    ) -> tuple[object, ...]:
        result = await self._post(
            "/execute_batch",
            {
                "executions": [
                    {
                        "model_id": item.model_id,
                        "kind": item.kind.value,
                        "payload": serialize_operation_payload(item.payload),
                    }
                    for item in executions
                ]
            },
        )
        return tuple(result)

    async def capture_operation(
        self,
        model_id: str,
        kind: OperationKind,
        payload: OperationPayload,
    ) -> object:
        return await self._post(
            "/capture_operation",
            {
                "model_id": model_id,
                "kind": kind.value,
                "payload": serialize_operation_payload(payload),
            },
        )

    async def persist_operation(
        self,
        model_id: str,
        kind: OperationKind,
        payload: OperationPayload,
        capture: object,
    ) -> object:
        return await self._post(
            "/persist_operation",
            {
                "model_id": model_id,
                "kind": kind.value,
                "payload": serialize_operation_payload(payload),
                "capture": capture,
            },
        )

    async def unload_model(self, model_id: str) -> None:
        await self._post("/unload_model", {"model_id": model_id})

    async def shutdown_backend(self) -> None:
        await self._post("/close", {})

    async def _post(self, path: str, body: dict) -> object:
        try:
            response = await self.http.post(path, json=body)
        except httpx.ReadTimeout as exc:
            if self.on_read_timeout is not None:
                self.on_read_timeout()
            raise TimeoutError(
                f"backend {path} exceeded {self.read_timeout:g}s"
            ) from exc
        if not response.is_success:
            try:
                message = response.json()["error"]
            except (ValueError, KeyError):
                message = response.text
            raise RuntimeError(message)
        return response.json()["result"]

    async def close(self) -> None:
        await self.http.aclose()


def main() -> None:
    import asyncio
    import importlib
    import os
    import sys

    reference, port = sys.argv[1], int(sys.argv[2])
    module_name, _, attr = reference.partition(":")
    executor = getattr(importlib.import_module(module_name), attr)()
    if int(os.environ.get("RANK", "0")) > 0:
        executor.follow()
        return
    import uvicorn

    app = create_backend_app(executor)
    try:
        uvicorn.run(app, host="127.0.0.1", port=port)
    finally:
        if not app.state.executor_closed:
            asyncio.run(executor.close())


if __name__ == "__main__":
    main()
