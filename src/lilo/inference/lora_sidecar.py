from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import modal
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from stitch.types import VersionConstraint, VersionRef

from .bulletin import SnapshotBulletin, SnapshotNotFound
from .snapshot_access import SnapshotAccess


def create_app(
    bulletin: SnapshotBulletin,
    upstream_url: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    refresh_interval_s: float = 1.0,
) -> FastAPI:
    if refresh_interval_s <= 0:
        raise ValueError("refresh_interval_s must be positive")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = httpx.AsyncClient(
            base_url=upstream_url.rstrip("/"),
            timeout=3000,
            trust_env=False,
            transport=transport,
        )
        app.state.snapshot_access = SnapshotAccess(bulletin)
        app.state.adapter_registrations = {}
        app.state.registered_adapters = {}
        app.state.registered_latest = {}
        app.state.refresh_models = set()
        app.state.adapter_resolutions = {}
        stop_refresh = asyncio.Event()
        refresher = asyncio.create_task(
            _refresh_active_models(app, bulletin, stop_refresh, refresh_interval_s)
        )
        try:
            yield
        finally:
            stop_refresh.set()
            await refresher
            pending = list(app.state.adapter_resolutions.values())
            await asyncio.gather(*pending, return_exceptions=True)
            await asyncio.gather(
                *app.state.adapter_registrations.values(), return_exceptions=True
            )
            await app.state.snapshot_access.close()
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

        response = await _post_generate(app.state.client, request, payload)
        if response is None:
            return Response(status_code=499)
        admission_retries = int(response.headers.get("x-lilo-admission-retries", "0"))
        try:
            body = response.json()
        except ValueError:
            return Response(
                content=response.content,
                status_code=response.status_code,
                media_type=response.headers.get("content-type"),
                headers={"x-lilo-admission-retries": str(admission_retries)},
            )
        if admission_retries and isinstance(body, dict):
            body.setdefault("meta_info", {})["lilo_admission_retries"] = (
                admission_retries
            )
        if served_version is not None and isinstance(body, dict):
            metadata = body.setdefault("meta_info", {})
            metadata["weight_version_start"] = served_version
            metadata["weight_version_end"] = served_version
        return JSONResponse(body, status_code=response.status_code)

    @app.post("/internal/reload_lora_adapter")
    async def reload_adapter(request: Request) -> Response:
        # SGLang's implicit CPU-cache reload must participate in the same file
        # read barrier as explicit registration. This endpoint is loopback-only.
        if request.client is None or request.client.host not in {"127.0.0.1", "::1"}:
            return Response(status_code=404)
        name = (await request.json()).get("lora_name")
        if not isinstance(name, str) or name not in app.state.registered_adapters:
            return Response(status_code=404)
        _, error = await _register_adapter(
            app, bulletin, VersionRef.parse(name), reload=True
        )
        if error is not None:
            return Response(
                content=error.content,
                status_code=error.status_code,
                media_type=error.headers.get("content-type"),
            )
        return JSONResponse({"success": True})

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
    if ref is None and constraint.min_version is not None:
        cached = app.state.registered_latest.get(run_id)
        if cached is not None and cached.version >= constraint.min_version:
            # The caller's publication watermark is already satisfied. Poll for
            # newer publications off the request path; a higher watermark still
            # forces foreground discovery. An unconstrained latest read below
            # retains foreground refresh semantics.
            app.state.refresh_models.add(run_id)
            return cached, None
    key = (run_id, constraint.exact_version, constraint.min_version)
    pending = app.state.adapter_resolutions
    task = pending.get(key)
    if task is None:

        async def prepare():
            access = app.state.snapshot_access
            if ref is not None:
                resolved = ref
            else:
                # Preserve latest/minimum freshness, including across clients.
                await access.refresh()
                async with access.read():
                    resolved = await asyncio.to_thread(bulletin.read_latest, run_id)
                if resolved is None:
                    raise SnapshotNotFound(run_id)
                if (
                    constraint.min_version is not None
                    and resolved.version < constraint.min_version
                ):
                    raise SnapshotNotFound(
                        f"{run_id} latest={resolved.version} required={constraint.min_version}"
                    )
            return await _register_adapter(app, bulletin, resolved)

        task = asyncio.create_task(prepare())
        pending[key] = task

        def finished(completed):
            if pending.get(key) is completed:
                del pending[key]
            if not completed.cancelled():
                # A disconnected caller may leave no waiter for this failure.
                completed.exception()

        task.add_done_callback(finished)
    # Coalesce foreground discovery only while it is in flight. The fast paths
    # above separately decide whether a later request needs discovery at all.
    # One disconnected request must not cancel registration for other callers.
    return await asyncio.shield(task)


async def _register_adapter(app, bulletin, ref, *, reload=False):
    if not reload and ref.identity in app.state.registered_adapters:
        return ref, None
    pending = app.state.adapter_registrations
    task = pending.get(ref.identity)
    if task is None:

        async def register():
            access = app.state.snapshot_access
            for attempt in range(2):
                try:
                    # Independent adapters validate and load concurrently. A
                    # refresh waits for ALL reads, including SGLang's file reads.
                    # Hold through the load acknowledgement, not just resolve().
                    async with access.read():
                        if reload:
                            # This immutable publication was already validated.
                            path = Path(app.state.registered_adapters[ref.identity])
                        else:
                            path = await asyncio.to_thread(bulletin.resolve, ref)
                        response = await app.state.client.post(
                            "/load_lora_adapter",
                            json={
                                "lora_name": ref.identity,
                                "lora_path": str(path),
                                "pinned": False,
                            },
                        )
                    if response.is_error:
                        return ref, response
                    app.state.registered_adapters[ref.identity] = str(path)
                    latest = app.state.registered_latest.get(ref.run_id)
                    if latest is None or latest.version < ref.version:
                        app.state.registered_latest[ref.run_id] = ref
                    return ref, None
                except SnapshotNotFound:
                    if attempt:
                        raise
                    await access.refresh()

        task = asyncio.create_task(register())
        pending[ref.identity] = task

        def finished(completed):
            if pending.get(ref.identity) is completed:
                del pending[ref.identity]
            if not completed.cancelled():
                completed.exception()

        task.add_done_callback(finished)
    return await asyncio.shield(task)


async def _refresh_active_models(app, bulletin, stop, interval):
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), interval)
            return
        except TimeoutError:
            pass
        run_ids = list(app.state.refresh_models)
        app.state.refresh_models.clear()
        if not run_ids:
            continue
        try:
            access = app.state.snapshot_access
            await access.refresh()
            async with access.read():
                refs = await asyncio.to_thread(
                    lambda: [bulletin.read_latest(run_id) for run_id in run_ids]
                )
            results = await asyncio.gather(
                *(
                    _register_adapter(app, bulletin, ref)
                    for ref in refs
                    if ref is not None
                ),
                return_exceptions=True,
            )
            if any(
                isinstance(result, Exception) or result[1] is not None
                for result in results
            ):
                logging.getLogger(__name__).warning(
                    "Background adapter refresh failed; foreground resolution remains available"
                )
        except Exception:
            # Never break a request whose required version is already available.
            logging.getLogger(__name__).warning(
                "Background adapter discovery failed; foreground resolution remains available"
            )


async def _post_generate(
    client: httpx.AsyncClient,
    request: Request,
    payload: dict,
) -> httpx.Response | None:
    original_rid = payload.setdefault("rid", f"lilo-{uuid.uuid4().hex}")
    input_ids = payload.get("input_ids") or []
    params = payload.get("sampling_params") or {}
    # A batch-level HTTP error doesn't prove that every sequence was rejected
    # before execution. Leave such requests to their caller instead of replaying.
    batched = (
        isinstance(original_rid, list)
        or isinstance(payload.get("text"), list)
        or bool(input_ids and isinstance(input_ids[0], list))
        or not isinstance(params, dict)
        or params.get("n", 1) != 1
    )
    for attempt in range(1 if batched else 3):
        if attempt:
            # A rejection produced no tokens. Give the scheduler another turn
            # before paying a gateway round trip and second-scale retry delay.
            # Use fresh IDs so a late abort/release cannot affect the retry.
            payload["rid"] = f"{original_rid}:admit-{attempt}"
        response = await _post_generate_once(client, request, payload)
        if response is not None:
            response.headers["x-lilo-admission-retries"] = str(attempt)
        if response is None or not _queue_rejected(response):
            return response
        if await request.is_disconnected():
            return None
    # Sustained overload must still reach the caller's rerouting/backoff policy.
    return response


def _queue_rejected(response: httpx.Response) -> bool:
    if response.status_code == 503:
        try:
            body = response.json()
        except ValueError:
            return False
        return (
            isinstance(body, dict)
            and body.get("detail") == "The request queue is full."
        )
    if response.status_code == 200 and response.headers.get(
        "content-type", ""
    ).startswith("text/event-stream"):
        # SGLang streams admission failures as HTTP 200 with an abort event.
        # Inspect only a small, token-free response; never replay partial output
        # or a mixed-success batch. Successful long streams incur no JSON parse.
        if len(response.content) > 4096:
            return False
        events = []
        try:
            for line in response.text.splitlines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    events.append(json.loads(line[6:]))
        except ValueError:
            return False
        if len(events) != 1 or not isinstance(events[0], dict):
            return False
        meta = events[0].get("meta_info", {})
        reason = meta.get("finish_reason", {})
        return (
            meta.get("completion_tokens") == 0
            and isinstance(reason, dict)
            and reason.get("type") == "abort"
            and reason.get("status_code") == 503
            and reason.get("message") == "The request queue is full."
        )
    return False


async def _post_generate_once(
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
