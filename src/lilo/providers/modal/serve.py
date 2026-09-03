from __future__ import annotations

import asyncio
import logging
import os
import secrets
import signal
import socket
import subprocess
import sys
import uuid
from collections.abc import Awaitable, Callable

import httpx
import modal

from lilo.engine import EngineServer
from lilo.engine.backend import HttpExecutor
from lilo.engine.http import create_engine_app

from .engines import EngineInstanceRecord, instance_key
from .kv import ModalKeyValueStore

ENGINE_PORT = 8000
BACKEND_SHUTDOWN_TIMEOUT = 120.0
BACKEND_STARTUP_TIMEOUT = 60 * 60
BACKEND_OPERATION_TIMEOUT = 3 * 60 * 60
NCCL_HEARTBEAT_TIMEOUT = 30 * 60


async def _kick_trainer_reconciler(definition_id: str) -> None:
    from .trainer_reconciler import request_reconcile

    async def spawn(delay_seconds: float) -> str:
        function = modal.Function.from_name("lilo", "trainer_reconciler")
        call = await function.spawn.aio(delay_seconds)
        return call.object_id

    await request_reconcile(spawn, definition_id)


async def serve_engine(
    kv: ModalKeyValueStore,
    make_server: Callable[[], Awaitable[EngineServer]],
    *,
    definition_id: str,
    revision: str,
    instance_id: str,
) -> None:
    import uvicorn

    record = EngineInstanceRecord(
        instance_id=instance_id,
        definition_id=definition_id,
        revision=revision,
        state="starting",
        call_id=modal.current_function_call_id(),
        boot_id=uuid.uuid4().hex,
    )
    await kv.put(instance_key(instance_id), record.model_dump(mode="json"))
    engine = None
    try:
        engine = await make_server()
        token = secrets.token_urlsafe(16)
        engine_app = create_engine_app(engine, token=token)
        with modal.forward(ENGINE_PORT) as tunnel:
            record = record.model_copy(
                update={"state": "running", "url": tunnel.url, "token": token}
            )
            await kv.put(instance_key(instance_id), record.model_dump(mode="json"))
            try:
                await _kick_trainer_reconciler(definition_id)
            except Exception:
                logging.getLogger(__name__).exception(
                    "trainer reconcile %s",
                    definition_id,
                )
            server = uvicorn.Server(
                uvicorn.Config(engine_app, host="0.0.0.0", port=ENGINE_PORT)
            )
            await server.serve()
    finally:
        try:
            if engine is not None:
                await engine.close()
        finally:
            record = record.model_copy(update={"state": "stopped"})
            await kv.put(instance_key(instance_id), record.model_dump(mode="json"))


def run_engine_with_backend(
    kv: ModalKeyValueStore,
    executor_reference: str,
    *,
    definition_id: str,
    revision: str,
    instance_id: str,
    backend_env: dict[str, str] | None = None,
    nproc: int = 1,
    max_models: int = 8,
    startup_timeout: float = BACKEND_STARTUP_TIMEOUT,
    operation_timeout: float = BACKEND_OPERATION_TIMEOUT,
) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = {**os.environ, **(backend_env or {})}
    env["PYTHONPATH"] = os.pathsep.join(
        [entry for entry in (env.get("PYTHONPATH"), *sys.path) if entry]
    )
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    env.setdefault("TORCH_NCCL_DUMP_ON_TIMEOUT", "1")
    env.setdefault("TORCH_NCCL_ENABLE_MONITORING", "1")
    env.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", str(NCCL_HEARTBEAT_TIMEOUT))
    env.setdefault("TORCH_NCCL_TRACE_BUFFER_SIZE", "2000")
    launcher = (
        [sys.executable, "-m"]
        if nproc == 1
        else [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc-per-node={nproc}",
            "-m",
        ]
    )
    backend = subprocess.Popen(
        [*launcher, "lilo.engine.backend", executor_reference, str(port)],
        env=env,
        start_new_session=True,
    )

    def signal_backend(sig: signal.Signals) -> None:
        try:
            os.killpg(backend.pid, sig)
        except ProcessLookupError:
            pass

    executor = HttpExecutor(
        f"http://127.0.0.1:{port}",
        read_timeout=operation_timeout,
        on_read_timeout=lambda: signal_backend(signal.SIGKILL),
    )

    async def make_server() -> EngineServer:
        try:
            async with asyncio.timeout(startup_timeout):
                while True:
                    if backend.poll() is not None:
                        raise RuntimeError(
                            f"backend exited with code {backend.returncode}"
                        )
                    try:
                        if (await executor.http.get("/healthz")).is_success:
                            return EngineServer(executor, max_models=max_models)
                    except httpx.TransportError:
                        pass
                    await asyncio.sleep(2)
        except TimeoutError as exc:
            signal_backend(signal.SIGTERM)
            raise TimeoutError(f"backend startup exceeded {startup_timeout:g}s") from exc

    async def serve_until_backend_exits() -> None:
        serving = asyncio.create_task(
            serve_engine(
                kv,
                make_server,
                definition_id=definition_id,
                revision=revision,
                instance_id=instance_id,
            )
        )
        try:
            while backend.poll() is None and not serving.done():
                await asyncio.sleep(2)
            if serving.done():
                await serving
                return
            raise RuntimeError(f"backend exited with code {backend.returncode}")
        finally:
            serving.cancel()
            await asyncio.gather(serving, return_exceptions=True)
            if backend.poll() is None:
                try:
                    await asyncio.wait_for(
                        executor.shutdown_backend(),
                        timeout=BACKEND_SHUTDOWN_TIMEOUT,
                    )
                except Exception:
                    logging.getLogger(__name__).exception("close backend")
            await executor.close()

    try:
        asyncio.run(serve_until_backend_exits())
    finally:
        running = backend.poll() is None
        signal_backend(signal.SIGTERM if running else signal.SIGKILL)
        if running:
            try:
                backend.wait(timeout=30)
            except subprocess.TimeoutExpired:
                signal_backend(signal.SIGKILL)
                backend.wait(timeout=10)
