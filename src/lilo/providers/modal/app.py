from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import modal
from stitch.pools.modal_flash import ModalFlashPool

from lilo.errors import RecordNotFound
from lilo.providers.contracts import (
    Parameterization,
    SamplingTask,
)

from .checkpoint_storage import (
    CHECKPOINT_ROOT,
    CHECKPOINT_VOLUME_NAME,
    checkpoint_volume,
)
from .deployment import (
    trainer_deployment_env,
    trainer_max_containers,
)
from .definitions import (
    qwen3_5_35b_a3b_full_64k,
    qwen3_5_4b_full_64k,
    qwen3_5_9b_full_64k,
    qwen3_6_27b_full_64k,
    qwen3_6_35b_a3b_full_64k,
)
from .engines import ModalEnginePlatform
from .fft_pool import (
    FFTPoolSpec,
    deploy_pool,
    pool_gateway,
    proxy_auth_headers,
    stop_pool,
)
from .image_dependencies import STITCH_PACKAGE
from .kv import (
    ModalSessionKeyValueStores,
    fft_pool_kv,
    shared_kv,
)
from .sampling import ModalSamplingTaskPlatform

APP_NAME = "lilo"
ROUTING_REGION = "us-west"
SESSION_IDLE_TIMEOUT = 300.0
FFT_POOL_IDLE_TIMEOUT = 300.0
FFT_POOL_ORPHAN_TIMEOUT = 1800.0
FFT_POOL_TOUCH_INTERVAL = 60.0
SWEEP_PERIOD = modal.Period(minutes=5)
CHECKPOINT_READ_LOCK = asyncio.Lock()
_pool_touches: dict[str, float] = {}

DEFINITIONS = (
    qwen3_5_4b_full_64k,
    qwen3_5_9b_full_64k,
    qwen3_5_35b_a3b_full_64k,
    qwen3_6_27b_full_64k,
    qwen3_6_35b_a3b_full_64k,
)
TRAINER_MAX_CONTAINERS = trainer_max_containers()
TRAINER_DEPLOYMENT_ENV = trainer_deployment_env()
app = modal.App(APP_NAME)
for definition in DEFINITIONS:
    app.include(definition.app)


def _checkpoint_path(uri: str) -> Path:
    path = Path(uri).resolve()
    if not path.is_relative_to(Path(CHECKPOINT_ROOT).resolve()):
        raise ValueError("checkpoint path is outside configured storage")
    return path


async def _read_checkpoint_metadata(uri: str) -> dict[str, object]:
    path = _checkpoint_path(uri)
    async with CHECKPOINT_READ_LOCK:
        await asyncio.to_thread(checkpoint_volume.reload)
        metadata = json.loads(
            await asyncio.to_thread(
                (path / "metadata.json").read_text,
                encoding="utf-8",
            )
        )
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata must be an object")
    return metadata


def _checkpoint_entry(checkpoint: Path) -> dict[str, object]:
    files = [file for file in checkpoint.rglob("*") if file.is_file()]
    metadata_file = checkpoint / "metadata.json"
    return {
        "model_id": checkpoint.parent.parent.name,
        "name": checkpoint.name,
        "path": str(checkpoint),
        "time": checkpoint.stat().st_mtime,
        "size_bytes": sum(file.stat().st_size for file in files),
        "metadata": (
            json.loads(metadata_file.read_text(encoding="utf-8"))
            if metadata_file.is_file()
            else None
        ),
    }


def _scan_checkpoints(model_id: str | None) -> list[dict[str, object]]:
    root = Path(CHECKPOINT_ROOT)
    if not root.is_dir():
        return []
    model_dirs = [root / model_id] if model_id is not None else list(root.iterdir())
    return [
        _checkpoint_entry(checkpoint)
        for model_dir in model_dirs
        if (model_dir / "weights").is_dir()
        for checkpoint in (model_dir / "weights").iterdir()
        if checkpoint.is_dir()
    ]


async def _list_checkpoints(model_id: str | None) -> list[dict[str, object]]:
    async with CHECKPOINT_READ_LOCK:
        await asyncio.to_thread(checkpoint_volume.reload)
        return await asyncio.to_thread(_scan_checkpoints, model_id)


async def _delete_checkpoint(uri: str) -> None:
    path = _checkpoint_path(uri)
    async with CHECKPOINT_READ_LOCK:
        await asyncio.to_thread(checkpoint_volume.reload)
        try:
            await asyncio.to_thread(shutil.rmtree, path)
        except FileNotFoundError:
            raise RecordNotFound("checkpoint", uri) from None
        await asyncio.to_thread(checkpoint_volume.commit)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_pyproject("pyproject.toml")
    .pip_install(STITCH_PACKAGE)
    .add_local_python_source("lilo")
)
proxy_secret = modal.Secret.from_name(
    "lilo-proxy",
    required_keys=["MODAL_PROXY_TOKEN_ID", "MODAL_PROXY_TOKEN_SECRET"],
)


async def _touch_fft_pool(spec: FFTPoolSpec) -> None:
    if spec.latest:
        return
    now = time.time()
    if now - _pool_touches.get(spec.app_name, 0.0) < FFT_POOL_TOUCH_INTERVAL:
        return
    _pool_touches[spec.app_name] = now
    await fft_pool_kv().put(_touch_key(spec), {"touched_at": now})


def _touch_key(spec: FFTPoolSpec) -> str:
    return f"fft_pool_touch:{spec.app_name}"


async def _last_touched(registry, spec: FFTPoolSpec, record: dict) -> float:
    touch = await registry.get(_touch_key(spec))
    return max(
        float(record["touched_at"]),
        float(touch["touched_at"]) if touch is not None else 0.0,
    )


@app.function(image=image, max_containers=1, timeout=20 * 60, retries=2)
async def ensure_fft_pool(spec: dict) -> str:
    pool = FFTPoolSpec.from_dict(spec)
    gateway = await asyncio.to_thread(deploy_pool, pool)
    await fft_pool_kv().put(
        f"fft_pool:{pool.app_name}",
        {**pool.as_dict(), "touched_at": time.time()},
    )
    return gateway


@app.function(
    image=image,
    min_containers=0,
    timeout=60 * 60,
    retries=2,
    secrets=[proxy_secret],
)
@modal.concurrent(max_inputs=128)
async def execute_sample(task: dict) -> dict:
    from lilo.inference.sampling import sample_task

    definition_id = str(task["engine_definition_id"])
    if parameterization_for(definition_id) != "full":
        raise ValueError(f"unsupported sampling definition: {definition_id}")
    definition = module_for(definition_id)
    rollout_world_size = definition.ROLLOUT_GPUS
    rollout_tensor_parallel_size = definition.ROLLOUT_TENSOR_PARALLEL_SIZE
    if rollout_world_size % rollout_tensor_parallel_size:
        raise ValueError("rollout GPU count must be divisible by tensor parallel size")
    rollout_data_parallel_size = rollout_world_size // rollout_tensor_parallel_size
    spec = (
        FFTPoolSpec.base(definition_id)
        if task["model_id"] is None
        else FFTPoolSpec(
            definition_id=definition_id,
            model_id=str(task["model_id"]),
            latest=bool(task.get("latest")),
            version=int(task["publish_version"]),
        )
    )
    await _touch_fft_pool(spec)
    return await sample_task(
        task,
        await pool_gateway(spec),
        data_parallel_size=rollout_data_parallel_size,
        headers=proxy_auth_headers(),
        on_wait=lambda: _touch_fft_pool(spec),
        context_length=definition.MAX_CONTEXT_LENGTH,
    )


def _latest_pool(model) -> FFTPoolSpec:
    return FFTPoolSpec(
        model.engine_definition_id,
        model.model_id,
        True,
        0,
        **(model.spec.get("rollout") or {}),
    )


async def _model_record(kv, model_id: str):
    from lilo.control_plane.keys import model_key
    from lilo.control_plane.records import ModelRecord

    return ModelRecord.model_validate(await kv.get(model_key(model_id)))


def module_for(definition_id: str):
    for definition in DEFINITIONS:
        if definition.DEFINITION_ID == definition_id:
            return definition
    raise KeyError(definition_id)


def parameterization_for(definition_id: str) -> Parameterization | None:
    try:
        return module_for(definition_id).PARAMETERIZATION
    except KeyError:
        return None


@app.function(
    image=image,
    env=TRAINER_DEPLOYMENT_ENV,
    max_containers=1,
    timeout=20 * 60,
    retries=3,
)
async def trainer_reconciler(delay_seconds: float = 0.0) -> None:
    from .trainer_reconciler import (
        complete_reconcile,
        pending_reconciliations,
        reconcile_trainers,
        release_reconcile_call,
    )

    if delay_seconds > 0:
        await asyncio.sleep(delay_seconds)
    call_id = modal.current_function_call_id()

    async def run(definition_id: str, token: str) -> None:
        parameterization = parameterization_for(definition_id)
        if parameterization is None:
            await complete_reconcile(definition_id, token)
            return
        module = module_for(definition_id)
        maximum_instances = TRAINER_MAX_CONTAINERS
        try:
            await reconcile_trainers(
                shared_kv(),
                ModalEnginePlatform(shared_kv(), _spawn_engine),
                definition_id,
                revision=None,
                maximum_instances=(
                    int(maximum_instances) if maximum_instances is not None else None
                ),
                models_per_instance=module.TRAINER_MODELS_PER_INSTANCE,
                scale_up=parameterization == "full",
            )
        except Exception:
            logging.getLogger(__name__).exception(
                "trainer reconcile %s",
                definition_id,
            )
            raise
        await complete_reconcile(definition_id, token)

    pending = await pending_reconciliations()
    try:
        await asyncio.gather(
            *(run(definition_id, token) for definition_id, token in pending.items())
        )
    finally:
        await release_reconcile_call(call_id)
    remaining = await pending_reconciliations()
    if remaining:
        await kick_trainer_reconciler(next(iter(remaining)))


async def kick_trainer_reconciler(definition_id: str) -> None:
    if parameterization_for(definition_id) is None:
        return
    from .trainer_reconciler import request_reconcile

    async def spawn(delay_seconds: float) -> str:
        call = await trainer_reconciler.spawn.aio(delay_seconds)
        return call.object_id

    await request_reconcile(spawn, definition_id)


async def _spawn_engine(definition_id: str, instance_id: str) -> str:
    engine = module_for(definition_id).ENGINE_FUNCTION
    call = await engine.spawn.aio(instance_id)
    return call.object_id


def _plane():
    from lilo.control_plane import ControlPlane

    kv = shared_kv()
    task_stores = ModalSessionKeyValueStores()
    engines = ModalEnginePlatform(kv, _spawn_engine)

    async def spawn_sampling(task: SamplingTask) -> str:
        call = await execute_sample.spawn.aio(asdict(task))
        return call.object_id

    async def prepare_model(model) -> None:
        if parameterization_for(model.engine_definition_id) == "full":
            await ensure_fft_pool.spawn.aio(_latest_pool(model).as_dict())

    async def ensure_pool(session) -> None:
        definition_id = session.engine_definition_id
        if parameterization_for(definition_id) != "full":
            return
        if session.model_id is None:
            pool = FFTPoolSpec.base(definition_id)
        elif session.publish_version is None:
            return
        else:
            pool = FFTPoolSpec(
                definition_id=definition_id,
                model_id=session.model_id,
                latest=session.latest,
                version=session.publish_version,
            )
        registry = fft_pool_kv()
        key = f"fft_pool:{pool.app_name}"
        record = await registry.get(key)
        if record is None:
            if pool.latest:
                pool = _latest_pool(await _model_record(kv, session.model_id))
            await ensure_fft_pool.remote.aio(pool.as_dict())
        else:
            await _touch_fft_pool(pool)

    async def kick_trainers(definition_id: str) -> bool:
        await kick_trainer_reconciler(definition_id)
        if parameterization_for(definition_id) != "full":
            return False
        maximum = TRAINER_MAX_CONTAINERS
        if maximum is None:
            return True
        instances = await engines.list_instances()
        return sum(
            instance.definition_id == definition_id and not instance.terminal
            for instance in instances
        ) < int(maximum)

    return ControlPlane(
        kv,
        engines,
        sampling_tasks=ModalSamplingTaskPlatform(task_stores, spawn_sampling),
        session_idle_timeout=SESSION_IDLE_TIMEOUT,
        ensure_sampling_pool=ensure_pool,
        prepare_model=prepare_model,
        sampling_task_stores=task_stores,
        read_checkpoint_metadata=_read_checkpoint_metadata,
        list_checkpoints=_list_checkpoints,
        delete_checkpoint=_delete_checkpoint,
        checkpoint_root=CHECKPOINT_ROOT,
        reconcile_trainers=kick_trainers,
        trainer_autoscaling=lambda definition_id: (
            parameterization_for(definition_id) == "full"
        ),
    )


@app.function(
    image=image,
    env=TRAINER_DEPLOYMENT_ENV,
    name="server",
    routing_region=ROUTING_REGION,
    timeout=20 * 60,
    volumes={CHECKPOINT_ROOT: checkpoint_volume},
    secrets=[modal.Secret.from_name("lilo-api", required_keys=["TINKER_API_KEY"])],
)
@modal.concurrent(max_inputs=128)
@modal.asgi_app(requires_proxy_auth=False)
def server():
    from lilo.control_plane import create_control_plane_app

    return create_control_plane_app(
        _plane(),
        DEFINITIONS,
        api_key=os.environ["TINKER_API_KEY"],
        checkpoint_volume=CHECKPOINT_VOLUME_NAME,
    )


async def _lose_undefined_models() -> tuple[str, ...]:
    from lilo.control_plane.keys import placement_key, trainer_demand_key
    from lilo.control_plane.records import ModelRecord

    kv = shared_kv()
    lost = []
    for _, value in await kv.list_items("model:"):
        model = ModelRecord.model_validate(value)
        if parameterization_for(model.engine_definition_id) is not None:
            continue
        await kv.delete(placement_key(model.model_id))
        await kv.delete(trainer_demand_key(model.model_id))
        lost.append(model.model_id)
    return tuple(lost)


async def _cleanup_fft_pools() -> tuple[str, ...]:
    from lilo.control_plane.records import ModelRecord

    active_latest = {
        FFTPoolSpec(
            model.engine_definition_id,
            model.model_id,
            True,
            0,
        ).app_name
        for _, value in await shared_kv().list_items("model:")
        for model in (ModelRecord.model_validate(value),)
        if parameterization_for(model.engine_definition_id) == "full"
    }
    stopped = []
    registry = fft_pool_kv()
    for key, value in await registry.list_items("fft_pool:"):
        spec = FFTPoolSpec.from_dict(value)
        if spec.latest:
            if spec.app_name in active_latest:
                continue
        else:
            idle = time.time() - await _last_touched(registry, spec, value)
            if idle < FFT_POOL_IDLE_TIMEOUT:
                continue
            try:
                replicas = await ModalFlashPool(
                    spec.app_name,
                    "Server",
                ).discover_replicas_async()
            except modal.exception.NotFoundError:
                await registry.delete(key)
                await registry.delete(_touch_key(spec))
                continue
            if replicas and idle < FFT_POOL_ORPHAN_TIMEOUT:
                continue
            await registry.delete(_touch_key(spec))
        await asyncio.to_thread(stop_pool, spec)
        await registry.delete(key)
        stopped.append(spec.app_name)
        if not spec.latest and await registry.get(_touch_key(spec)) is not None:
            await ensure_fft_pool.spawn.aio(spec.as_dict())
    return tuple(stopped)


@app.function(image=image, env=TRAINER_DEPLOYMENT_ENV, schedule=SWEEP_PERIOD)
def cleaner():
    import asyncio

    async def run() -> None:
        plane = _plane()
        await plane.sweep_idle_sessions(SESSION_IDLE_TIMEOUT)
        await plane.sweep_idle_models(SESSION_IDLE_TIMEOUT)
        await plane.sweep_idle_engines()
        await _lose_undefined_models()
        await _cleanup_fft_pools()

    asyncio.run(run())
