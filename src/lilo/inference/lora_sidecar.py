from __future__ import annotations

import argparse
import asyncio
import contextlib
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from stitch.types import VersionConstraint, VersionRef

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
        try:
            yield
        finally:
            await app.state.client.aclose()

    app = FastAPI(lifespan=lifespan)

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
                ref = await _resolve(bulletin, str(run_id), constraint)
                registration_error = await _ensure_adapter(
                    app,
                    ref.identity,
                    str(bulletin.resolve(ref)),
                )
                if registration_error is not None:
                    return _passthrough(registration_error)
                payload["lora_path"] = ref.identity
                served_version = ref.version
            except (SnapshotNotFound, ValueError) as exc:
                return JSONResponse(
                    {"error": f"adapter version is not ready: {exc}"},
                    status_code=409,
                )

        response = await _post_generate(app.state.client, request, payload)
        if response is None:
            return Response(status_code=499)
        try:
            body = response.json()
        except ValueError:
            return _passthrough(response)
        if served_version is not None and isinstance(body, dict):
            metadata = body.setdefault("meta_info", {})
            metadata["weight_version_start"] = served_version
            metadata["weight_version_end"] = served_version
        return JSONResponse(body, status_code=response.status_code)

    return app


async def _ensure_adapter(
    app: FastAPI,
    name: str,
    path: str,
) -> httpx.Response | None:
    async with app.state.adapter_lock:
        registered = app.state.registered_adapters
        if registered.get(name) == path:
            return None
        if name in registered:
            raise RuntimeError(f"adapter {name!r} changed its immutable path")
        response = await app.state.client.post(
            "/load_lora_adapter",
            json={"lora_name": name, "lora_path": path, "pinned": False},
        )
        if response.is_error:
            return response
        registered[name] = path
        return None


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


def _passthrough(response: httpx.Response) -> Response:
    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=response.headers.get("content-type"),
    )


async def _resolve(
    bulletin: SnapshotBulletin,
    run_id: str,
    constraint: VersionConstraint,
) -> VersionRef:
    await bulletin.refresh()
    if constraint.exact_version is not None:
        return VersionRef(run_id, constraint.exact_version)
    latest = bulletin.read_latest(run_id)
    if latest is None:
        raise SnapshotNotFound(run_id)
    if constraint.min_version is not None and latest.version < constraint.min_version:
        raise SnapshotNotFound(
            f"{run_id} latest={latest.version} required={constraint.min_version}"
        )
    return latest


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
        import modal

        refresh = modal.Volume.from_name(args.bulletin_volume, version=2).reload
    bulletin = SnapshotBulletin(Path(args.bulletin_root), refresh=refresh)
    import uvicorn

    uvicorn.run(
        create_app(bulletin, args.upstream_url),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
