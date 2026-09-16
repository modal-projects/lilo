from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from lilo.telemetry import backend as telemetry
from lilo.telemetry.otlp import provider

from .api import Command, Executor, OperationKind
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


class ForwardBackwardBatchBody(BaseModel):
    executions: tuple[ExecuteBody, ...]


class SnapshotBody(ExecuteBody):
    capture: Any = None


def create_backend_app(executor: Executor) -> FastAPI:
    app = FastAPI()
    app.state.executor_closed = False

    @app.middleware("http")
    async def collect_measurements(request, call_next):
        with telemetry.recording(request.headers.get("x-lilo-telemetry") == "1"):
            return await call_next(request)

    async def run(action: Awaitable[object]) -> JSONResponse:
        def respond(content, status=200):
            measurements = telemetry.active.get()
            if measurements is not None:
                content["telemetry"] = measurements.as_dict()
            return JSONResponse(status_code=status, content=content)

        try:
            return respond({"result": await action})
        except Exception as exc:  # noqa: BLE001 - executor errors cross the HTTP boundary
            return respond({"error": str(exc)}, status=500)

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

    @app.post("/execute_forward_backward_batch")
    async def execute_forward_backward_batch(
        body: ForwardBackwardBatchBody,
    ) -> JSONResponse:
        return await run(
            executor.execute_forward_backward_batch(
                tuple(
                    Command(
                        item.model_id,
                        item.kind,
                        parse_operation_payload(item.kind, item.payload),
                    )
                    for item in body.executions
                )
            )
        )

    @app.post("/capture_snapshot")
    async def capture_snapshot(body: SnapshotBody) -> JSONResponse:
        return await run(
            executor.capture_snapshot(
                body.model_id,
                body.kind,
                parse_operation_payload(body.kind, body.payload),
            )
        )

    @app.post("/persist_snapshot")
    async def persist_snapshot(body: SnapshotBody) -> JSONResponse:
        return await run(
            executor.persist_snapshot(
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


class HttpBackendClient:
    """Proxy the executor interface to the training subprocess over HTTP."""

    def __init__(
        self,
        base_url: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        read_timeout: float | None = None,
        on_read_timeout: Callable[[], None] | None = None,
        on_transport_error: Callable[[], None] | None = None,
    ) -> None:
        self.read_timeout = read_timeout
        self.on_read_timeout = on_read_timeout
        self.on_transport_error = on_transport_error
        self._transport_failed = False
        self.http = httpx.AsyncClient(
            base_url=base_url,
            transport=transport,
            timeout=httpx.Timeout(30.0, read=read_timeout),
            # Commands mutate training state. Avoid racing the local server's
            # idle connection timeout, and never retry an ambiguous command.
            limits=httpx.Limits(max_keepalive_connections=0),
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

    async def execute_forward_backward_batch(
        self,
        executions: tuple[Command, ...],
    ) -> tuple[object, ...]:
        result = await self._post(
            "/execute_forward_backward_batch",
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

    async def capture_snapshot(
        self,
        model_id: str,
        kind: OperationKind,
        payload: OperationPayload,
    ) -> object:
        return await self._post(
            "/capture_snapshot",
            {
                "model_id": model_id,
                "kind": kind.value,
                "payload": serialize_operation_payload(payload),
            },
        )

    async def persist_snapshot(
        self,
        model_id: str,
        kind: OperationKind,
        payload: OperationPayload,
        capture: object,
    ) -> object:
        return await self._post(
            "/persist_snapshot",
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
        if self._transport_failed:
            raise RuntimeError("backend transport failed; checkpoint recovery required")
        telemetry.received.set(None)
        enabled = provider() is not None
        try:
            response = await self.http.post(
                path,
                json=body,
                headers={"x-lilo-telemetry": "1"} if enabled else {},
            )
        except httpx.ReadTimeout as exc:
            self._fence_transport_failure(read_timeout=True)
            raise TimeoutError(
                f"backend {path} exceeded {self.read_timeout:g}s"
            ) from exc
        except httpx.TransportError as exc:
            self._fence_transport_failure()
            raise RuntimeError(
                f"backend {path} transport failed ({type(exc).__name__}); "
                "execution outcome unknown; checkpoint recovery required"
            ) from exc
        if enabled:
            try:
                evidence = response.json().get("telemetry")
                if isinstance(evidence, dict):
                    telemetry.received.set(evidence)
            except (ValueError, AttributeError):
                pass
        if not response.is_success:
            try:
                message = response.json()["error"]
            except (ValueError, KeyError):
                message = response.text
            raise RuntimeError(message)
        return response.json()["result"]

    def _fence_transport_failure(self, *, read_timeout: bool = False) -> None:
        if self._transport_failed:
            return
        self._transport_failed = True
        callback = self.on_transport_error
        if callback is None and read_timeout:
            callback = self.on_read_timeout
        if callback is not None:
            try:
                callback()
            except Exception:
                logging.getLogger(__name__).exception("fence backend transport failure")

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
        executor.run_follower_loop()
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
