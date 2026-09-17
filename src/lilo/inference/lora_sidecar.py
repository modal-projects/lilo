from __future__ import annotations

import argparse
import asyncio
import contextlib
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import modal
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from opentelemetry.propagate import extract
from opentelemetry.trace import StatusCode
from stitch.types import VersionConstraint, VersionRef

from lilo.telemetry.performance import stages
from lilo.telemetry.serving_metrics import ServingMetrics

from .bulletin import SnapshotBulletin, SnapshotNotFound


def create_app(
    bulletin: SnapshotBulletin,
    upstream_url: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = httpx.AsyncClient(
            base_url=upstream_url.rstrip("/"),
            timeout=3000,
            trust_env=False,
            transport=transport,
        )
        app.state.adapter_lock = asyncio.Lock()
        app.state.registered_adapters = {}
        app.state.adapter_resolutions = {}
        metrics = ServingMetrics("inference", upstream_url)
        metrics.start()
        try:
            yield
        finally:
            pending = list(app.state.adapter_resolutions.values())
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            await app.state.client.aclose()
            await asyncio.to_thread(metrics.close)

    app = FastAPI(lifespan=lifespan)

    @app.middleware("http")
    async def trace_request(request, call_next):
        if request.url.path != "/generate":
            return await call_next(request)
        with stages("inference").track(
            "request", context=extract(dict(request.headers))
        ) as span:
            response = await call_next(request)
            if span is not None:
                span.set_attribute("http.response.status_code", response.status_code)
                if response.status_code >= 400:
                    span.set_status(StatusCode.ERROR)
            return response

    @app.get("/health")
    async def health() -> Response:
        response = await app.state.client.get("/health")
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type"),
        )

    @app.post("/generate")
    async def generate(request: Request) -> Response:
        payload = await request.json()
        run_id = payload.pop("weight_run_id", None)
        constraint = VersionConstraint.from_payload(payload)
        payload.pop("weight_version", None)
        served_version = None
        if run_id is not None:
            try:
                ref, registration_error = await _prepare_adapter(
                    app, bulletin, str(run_id), constraint
                )
                if registration_error is not None:
                    return Response(
                        content=registration_error.content,
                        status_code=registration_error.status_code,
                        media_type=registration_error.headers.get("content-type"),
                    )
                payload["lora_path"] = ref.identity
                served_version = ref.version
            except (SnapshotNotFound, ValueError) as exc:
                return JSONResponse(
                    {"error": f"adapter version is not ready: {exc}"},
                    status_code=409,
                )

        with stages("inference").track("sglang_request") as span:
            response = await _post_generate(app.state.client, request, payload)
            if span is not None:
                status = response.status_code if response is not None else 499
                span.set_attribute("http.response.status_code", status)
                if status >= 400:
                    span.set_status(StatusCode.ERROR)
        if response is None:
            return Response(status_code=499)
        try:
            body = response.json()
        except ValueError:
            return Response(
                content=response.content,
                status_code=response.status_code,
                media_type=response.headers.get("content-type"),
            )
        if served_version is not None and isinstance(body, dict):
            metadata = body.setdefault("meta_info", {})
            metadata["weight_version_start"] = served_version
            metadata["weight_version_end"] = served_version
        return JSONResponse(body, status_code=response.status_code)

    return app


async def _prepare_adapter(
    app: FastAPI,
    bulletin: SnapshotBulletin,
    run_id: str,
    constraint: VersionConstraint,
) -> tuple[VersionRef, httpx.Response | None]:
    ref = (
        VersionRef(run_id, constraint.exact_version)
        if constraint.exact_version is not None
        else None
    )
    if ref is not None and ref.identity in app.state.registered_adapters:
        return ref, None
    key = (run_id, constraint.exact_version, constraint.min_version)
    pending = app.state.adapter_resolutions
    task = pending.get(key)
    if task is None:

        async def prepare():
            # Reload must not hide paths while SGLang reads a new adapter.
            telemetry = stages("inference")
            with telemetry.track(
                "adapter_lock_wait", attributes={"lilo.model_id": run_id}
            ):
                await app.state.adapter_lock.acquire()
            try:
                if ref is not None and ref.identity in app.state.registered_adapters:
                    return ref, None
                if ref is not None:
                    resolved = ref
                    try:
                        with telemetry.track("adapter_validate"):
                            path = await asyncio.to_thread(bulletin.resolve, resolved)
                    except SnapshotNotFound:
                        # Exact versions are immutable: reload only if this mount
                        # does not yet contain the complete requested snapshot.
                        with telemetry.track("volume_refresh"):
                            await bulletin.refresh()
                        with telemetry.track("adapter_validate"):
                            path = await asyncio.to_thread(bulletin.resolve, resolved)
                else:
                    # Latest/minimum requests must see newly published policies.
                    with telemetry.track("volume_refresh"):
                        await bulletin.refresh()
                    resolved = bulletin.read_latest(run_id)
                    if resolved is None:
                        raise SnapshotNotFound(run_id)
                    if (
                        constraint.min_version is not None
                        and resolved.version < constraint.min_version
                    ):
                        raise SnapshotNotFound(
                            f"{run_id} latest={resolved.version} required={constraint.min_version}"
                        )
                    if resolved.identity in app.state.registered_adapters:
                        return resolved, None
                    with telemetry.track("adapter_validate"):
                        path = await asyncio.to_thread(bulletin.resolve, resolved)
                with telemetry.track(
                    "adapter_load",
                    attributes={
                        "lilo.model_id": run_id,
                        "lilo.version": resolved.version,
                    },
                ) as span:
                    response = await app.state.client.post(
                        "/load_lora_adapter",
                        json={
                            "lora_name": resolved.identity,
                            "lora_path": str(path),
                            "pinned": False,
                        },
                    )
                    if span is not None:
                        span.set_attribute(
                            "http.response.status_code", response.status_code
                        )
                        if response.is_error:
                            span.set_status(StatusCode.ERROR)
                if response.is_error:
                    return resolved, response
                app.state.registered_adapters[resolved.identity] = str(path)
                return resolved, None
            finally:
                app.state.adapter_lock.release()

        task = asyncio.create_task(prepare())
        pending[key] = task

        def finished(completed):
            if pending.get(key) is completed:
                del pending[key]
            if not completed.cancelled():
                # A disconnected caller may leave no waiter for this failure.
                completed.exception()

        task.add_done_callback(finished)
    # Share only in-flight work. Later requests still resolve the latest version.
    # One disconnected request must not cancel registration for other callers.
    with stages("inference").track(
        "adapter_resolution_wait", attributes={"lilo.model_id": run_id}
    ):
        return await asyncio.shield(task)


async def _post_generate(
    client: httpx.AsyncClient,
    request: Request,
    payload: dict,
) -> httpx.Response | None:
    rid = payload.setdefault("rid", f"lilo-{uuid.uuid4().hex}")
    task = asyncio.create_task(client.post("/generate", json=payload))
    try:
        while not task.done():
            done, _ = await asyncio.wait((task,), timeout=0.25)
            if done:
                break
            if await request.is_disconnected():
                await _abort(client, rid)
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                return None
        return await task
    except asyncio.CancelledError:
        await _abort(client, rid)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        raise


async def _abort(client: httpx.AsyncClient, rid) -> None:
    rids = rid if isinstance(rid, list) else [rid]
    await asyncio.gather(
        *(client.post("/abort_request", json={"rid": str(item)}) for item in rids),
        return_exceptions=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--upstream-url", required=True)
    parser.add_argument("--bulletin-root", required=True)
    parser.add_argument("--bulletin-volume", default="")
    args = parser.parse_args()

    refresh = None
    if args.bulletin_volume:
        refresh = modal.Volume.from_name(args.bulletin_volume, version=2).reload
    bulletin = SnapshotBulletin(Path(args.bulletin_root), refresh=refresh)
    uvicorn.run(
        create_app(bulletin, args.upstream_url),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
