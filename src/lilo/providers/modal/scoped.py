"""Factories for a scoped app and its zero-minimum pinned sampler apps."""
from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
import json
import os
import time
from types import SimpleNamespace

import modal

from lilo.engines import Engine, gpu_count


def control_image():
    from .image_dependencies import CORE_PACKAGES, STITCH_PACKAGE, TINKER_PACKAGE
    return (modal.Image.debian_slim(python_version="3.12").apt_install("git")
            .pip_install(*CORE_PACKAGES, STITCH_PACKAGE, TINKER_PACKAGE, "huggingface-hub")
            .add_local_python_source("lilo"))


def register_sampler(app, *, engine, image, assets, bulletin, registry_name,
                     slot, model_id, version, pool, proxy_secret, name):
    from lilo.inference.serving import (
        start_sglang, start_fft_sidecar, wait_http, supervise_children, terminate,
    )
    model_path = engine.training.hf_checkpoint
    # Latest min is activated only after its model has been assigned.
    @app.server(name=name, serialized=True, image=image, gpu=engine.sampler_gpu,
                cpu=engine.sampler_cpu, memory=engine.sampler_memory,
                volumes={"/assets": assets, "/bulletin": bulletin},
                min_containers=0,
                max_containers=pool.max_containers,
                scaledown_window=pool.scaledown_window,
                target_concurrency=engine.sampling.target_concurrency,
                startup_timeout=1800, exit_grace_period=120,
                port=8000, routing_region="us-west")
    class Sampler:
        @modal.enter()
        def start(self):
            run_id = model_id
            if slot is not None:
                run_id = modal.Dict.from_name(registry_name)[f"slot:{slot}"]
            settings = asdict(engine.sampling)
            settings.pop("target_concurrency")
            self.sglang = start_sglang(
                model_path, port=8001, context_length=engine.training.seq_length,
                max_loras_per_batch=1, max_loaded_loras=1, max_lora_rank=1,
                parallel_world_size=gpu_count(engine.sampler_gpu),
                enable_lora=False, enable_cpu_weight_cache=True, **settings,
            )
            wait_http("http://127.0.0.1:8001/health", self.sglang, 1800)
            self.sidecar = start_fft_sidecar(
                port=8000, sglang_port=8001, model_path=model_path,
                bulletin_root="/bulletin", bulletin_volume=bulletin.name,
                run_id=run_id, pinned_version=version,
                scoped_registry=registry_name if slot is not None else None,
            )
            self.supervisor = supervise_children(self.sglang, self.sidecar)
            wait_http("http://127.0.0.1:8000/health", self.sidecar, 1800)

        @modal.exit()
        def stop(self):
            terminate(getattr(self, "sidecar", None))
            terminate(getattr(self, "sglang", None))
    return Sampler


def build_app(engine: Engine, name, registry_name, api_key, max_trainers,
              latest, pinned, checkpoint_volume_name, proxy_secret):
    from .kv import shared_kv, ModalSessionKeyValueStores
    from .engines import ModalEnginePlatform
    from .sampling import ModalSamplingTaskPlatform
    from .megatron_image import image as default_trainer_image
    from .rollout_image import image as default_sampler_image
    from lilo.control_plane import create_control_plane_app
    from .scoped_control import ScopedControlPlane
    from .scoped_pool import ScopedFlashPool
    from .fft_pool import proxy_auth_headers
    from lilo.inference.sampling import sample_task

    engine = replace(engine, training=replace(
        engine.training, hf_checkpoint=engine.training.hf_checkpoint or
        "/assets/" + engine.model.rsplit("/", 1)[-1]))
    from modal.config import config
    environment_name = config.get("environment") or ""
    app = modal.App(name)
    image = control_image()
    trainer_image = engine.trainer_image or default_trainer_image
    sampler_image = engine.sampler_image or default_sampler_image
    assets = modal.Volume.from_name("lilo-model-assets", create_if_missing=True)
    bulletin = modal.Volume.from_name("lilo-snapshot-bulletin", create_if_missing=True, version=2)
    checkpoints = modal.Volume.from_name(checkpoint_volume_name, create_if_missing=True, version=2)
    api_secret = modal.Secret.from_dict({"TINKER_API_KEY": api_key})

    @app.function(name="prepare_assets", image=image, serialized=True,
                  volumes={"/assets": assets}, timeout=3600)
    def prepare_assets():
        from huggingface_hub import snapshot_download
        # Explicit revisions are always resolved by HF; unpinned recipes may reuse
        # the existing local asset cache, matching the shared deployment recipes.
        if engine.revision or not os.path.isfile(engine.training.hf_checkpoint + "/config.json"):
            snapshot_download(engine.model, revision=engine.revision,
                              local_dir=engine.training.hf_checkpoint)
            assets.commit()

    @app.function(name="trainer", image=trainer_image, serialized=True,
                  gpu=engine.trainer_gpu, cpu=engine.trainer_cpu,
                  memory=engine.trainer_memory, timeout=engine.trainer_timeout,
                  min_containers=0, max_containers=max_trainers,
                  single_use_containers=True, retries=0,
                  volumes={"/assets": assets, "/bulletin": bulletin, "/checkpoints": checkpoints},
                  secrets=[api_secret, proxy_secret])
    def trainer(instance_id):
        from modal.config import config
        from .serve import run_engine_with_backend
        backend_config = {"megatron": asdict(engine.training), "checkpoint_dir": "/checkpoints"}
        run_engine_with_backend(
            shared_kv(), "lilo.backends.megatron_fft:build_executor",
            definition_id=engine.name, revision=config["image_id"], instance_id=instance_id,
            nproc=gpu_count(engine.trainer_gpu), max_models=1, notify_reconciler=False,
            backend_env={
                **engine.backend_env,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TORCHINDUCTOR_COMPILE_THREADS": "1",
                "LILO_BACKEND_CONFIG": json.dumps(backend_config),
                "LILO_BASE_MODEL": engine.model,
                "LILO_DEFINITION_ID": engine.name,
                "LILO_DEFINITION_REVISION": config["image_id"],
                "LILO_CHECKPOINT_VOLUME": checkpoint_volume_name,
                "LILO_BULLETIN_ROOT": "/bulletin",
                "LILO_BULLETIN_VOLUME": "lilo-snapshot-bulletin",
                "LILO_SCOPED_REGISTRY": registry_name,
            },
        )

    servers = [register_sampler(
        app, engine=engine, image=sampler_image, assets=assets, bulletin=bulletin,
        registry_name=registry_name, slot=None, model_id="base", version=0,
        pool=pinned, proxy_secret=proxy_secret, name="base_sampler")]
    servers.extend(register_sampler(
        app, engine=engine, image=sampler_image, assets=assets, bulletin=bulletin,
        registry_name=registry_name, slot=i, model_id=None, version=None,
        pool=latest, proxy_secret=proxy_secret, name="latest_sampler" if max_trainers == 1 else f"latest_sampler_{i}")
        for i in range(max_trainers))

    async def spawn_engine(definition_id, instance_id):
        call = await trainer.spawn.aio(instance_id)
        return call.object_id

    @app.function(name="manage", image=image, serialized=True,
                  max_containers=1, timeout=3600, retries=0, secrets=[proxy_secret])
    async def manage(action, model_id=None, version=None, lease=None):
        registry = modal.Dict.from_name(registry_name)
        if action == "close":
            await registry.put.aio("closing", True)
            return
        if await registry.get.aio("closing"):
            raise RuntimeError("deployment is closing")
        engines = ModalEnginePlatform(shared_kv(), spawn_engine)
        if action == "warm":
            instance = await engines.ensure_instance(engine.name)
            deadline = time.monotonic() + 3500
            while time.monotonic() < deadline:
                record = await engines.get_instance(instance.instance_id)
                if record and record.state == "running":
                    return record.instance_id
                if record and record.terminal:
                    raise RuntimeError("trainer failed during warmup")
                await asyncio.sleep(2)
            raise TimeoutError("trainer warmup exceeded deadline")
        if action == "claim":
            from .scoped_assignment import claim_model
            route = await claim_model(registry, shared_kv(), engines, engine.name, model_id)
            if latest.min_containers:
                from lilo.providers.modal.scoped_pool import set_minimum
                await set_minimum.aio(route["function_id"], latest.min_containers)
            return route
        if action == "reap_pinned":
            from .scoped_pins import reap_pins
            from lilo.run import stop_app
            async def stop(child):
                await asyncio.to_thread(stop_app, child)
            return await reap_pins(registry, stop, now=time.time())
        if action == "release_pinned":
            from .scoped_pins import touch_pin
            await touch_pin(registry, f"pinned:{model_id}:{version}",
                            now=time.time(), lease=lease, release=True)
            return
        if action == "pinned":
            from .scoped_pins import touch_pin
            key = f"pinned:{model_id}:{version}"
            existing = await registry.get.aio(key)
            if existing:
                await touch_pin(registry, key, now=time.time(), lease=lease)
                return existing
            # Record intent BEFORE deploying; teardown can find partial deployments.
            import hashlib
            child_name = name + "-pin-" + hashlib.sha256(key.encode()).hexdigest()[:12]
            children = await registry.get.aio("children") or []
            if child_name not in children:
                await registry.put.aio("children", [*children, child_name])
            child = modal.App(child_name)
            pinned_image = modal.Image.from_id(await registry.get.aio("sampler_image_id")).add_local_python_source("lilo")
            server = register_sampler(
                child, engine=engine, image=pinned_image,
                assets=modal.Volume.from_name("lilo-model-assets"),
                bulletin=modal.Volume.from_name("lilo-snapshot-bulletin", version=2),
                registry_name=registry_name, slot=None, model_id=model_id, version=version,
                pool=pinned, proxy_secret=proxy_secret, name="sampler")
            await child.deploy.aio(name=child_name, environment_name=environment_name)
            route = {"url": await server.get_url.aio(), "function_id": server.object_id}
            records = await registry.get.aio("pin_records") or {}
            records[key] = {"app_name": child_name, "last_used": time.time(), "leases": {}}
            await registry.put.aio("pin_records", records)
            await registry.put.aio(key, route)
            await touch_pin(registry, key, now=time.time(), lease=lease)
            return route
        raise ValueError(f"unknown action: {action}")

    async def route_for(model_id, version, is_latest):
        registry = modal.Dict.from_name(registry_name)
        if await registry.get.aio("closing"):
            raise RuntimeError("deployment is closing")
        if model_id is None:
            return (await registry.get.aio("routes"))[0]
        if is_latest:
            if await registry.get.aio("slot:0") != model_id:
                raise ValueError("sampling model was replaced; use a new sampling client")
            route = await registry.get.aio("model:" + model_id)
            if not route:
                raise ValueError("no latest pool assigned to model")
            return route
        return await manage.remote.aio("pinned", model_id, version)

    @app.function(name="execute_sample", image=image, serialized=True,
                  timeout=3600, retries=0, secrets=[proxy_secret])
    @modal.concurrent(max_inputs=128)
    async def execute_sample(task):
        import uuid
        pinned_request = task["model_id"] is not None and not task.get("latest")
        lease = uuid.uuid4().hex if pinned_request else None
        if pinned_request:
            route = await manage.remote.aio("pinned", task["model_id"], task.get("publish_version"), lease)
        else:
            route = await route_for(task["model_id"], task.get("publish_version"), task.get("latest"))
        try:
            return await sample_task(
                task, ScopedFlashPool(route).gateway_url(),
                data_parallel_size=gpu_count(engine.sampler_gpu) // engine.sampling.tensor_parallel_size,
                headers=proxy_auth_headers(), context_length=engine.training.seq_length)
        finally:
            if pinned_request:
                await manage.remote.aio("release_pinned", task["model_id"], task.get("publish_version"), lease)

    @app.function(name="api", image=image, serialized=True,
                  timeout=1200, secrets=[api_secret],
                  volumes={"/checkpoints": checkpoints}, routing_region="us-west")
    @modal.concurrent(max_inputs=128)
    @modal.asgi_app(requires_proxy_auth=False)
    def api():
        async def prepare_model(model):
            if model.spec.get("rollout"):
                raise ValueError("configure sampler pool limits on lilo.run, not model creation")
            await manage.remote.aio("claim", model.model_id)

        async def ensure_pool(session):
            await route_for(session.model_id, session.publish_version, session.latest)

        async def spawn_sampling(task):
            call = await execute_sample.spawn.aio(asdict(task))
            return call.object_id

        stores = ModalSessionKeyValueStores()
        plane = ScopedControlPlane(
            shared_kv(), ModalEnginePlatform(shared_kv(), spawn_engine),
            session_idle_timeout=None, prepare_model=prepare_model,
            ensure_sampling_pool=ensure_pool,
            sampling_task_stores=stores,
            sampling_tasks=ModalSamplingTaskPlatform(stores, spawn_sampling),
        )
        definition = SimpleNamespace(
            CATALOG_VISIBLE=True, DEFINITION_ID=engine.name, MODEL_NAME=engine.model,
            PARAMETERIZATION="full", MAX_CONTEXT_LENGTH=engine.training.seq_length,
        )
        return create_control_plane_app(plane, (definition,), api_key=os.environ["TINKER_API_KEY"],
                                        checkpoint_volume=checkpoint_volume_name)
    return app, api, manage, servers, prepare_assets, sampler_image
