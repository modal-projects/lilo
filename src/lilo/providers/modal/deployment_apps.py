"""Modal app builders shared by all Python-configured model deployments.

Trainers and inference provisioners are independently deployed apps. Pools are
created on demand with the existing LoRA/FFT pool lifecycle.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import modal

from lilo.deployments import DeploymentRecord, gpu_count, validate_frontend
from lilo.backends.deployment import backend_config, serving_options

MANIFEST_ENV = "LILO_DEPLOYMENT_MANIFEST"
POOL_CONFIG_ENV = "LILO_POOL_DEPLOYMENT"


def manifest_from_env():
    data = os.environ.get(MANIFEST_ENV)
    if not data:
        raise ValueError(
            "Missing deployment manifest. Use lilo deploy with your Python config files."
        )
    rows = json.loads(data)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Deployment manifest must be a nonempty list")
    return [DeploymentRecord.model_validate(row) for row in rows]


def frontend_settings():
    deployments = manifest_from_env()
    active = [row.spec for row in deployments if row.active]
    validate_frontend(active)
    if len({row.definition_id for row in deployments}) != len(deployments):
        raise ValueError("duplicate deployment generation")
    return active[0]


def image_for(backend):
    if not modal.is_local():
        return modal.Image.debian_slim()
    if backend == "miles":
        from .miles_image import image
    elif backend == "megatron":
        from .megatron_image import image
    else:
        from .rollout_image import image
    return image


def volumes_for(spec):
    storage = spec.deployment["storage"]
    return {
        "/assets": modal.Volume.from_name(storage["assets"], create_if_missing=True),
        "/checkpoints": modal.Volume.from_name(
            storage["checkpoints"], create_if_missing=True, version=2
        ),
        "/bulletin": modal.Volume.from_name(
            storage["bulletin"], create_if_missing=True, version=2
        ),
    }


def secrets_for(spec, *, training=False):
    names = spec.deployment["secrets"]
    result = [modal.Secret.from_name(names["api"], required_keys=["TINKER_API_KEY"])]
    if training:
        result.append(
            modal.Secret.from_name(
                names["sampler_proxy"],
                required_keys=["MODAL_PROXY_TOKEN_ID", "MODAL_PROXY_TOKEN_SECRET"],
            )
        )
        if names["huggingface"]:
            result.append(modal.Secret.from_name(names["huggingface"]))
    return result


def deployment_env(values):
    """Keep user environment overrides separate from Lilo's deployment wiring."""
    if any(key.startswith("LILO_") for key in values):
        raise ValueError("LILO_ environment variables are managed by Lilo")
    return dict(values)


def build_trainer_app(resolved: DeploymentRecord, *, image=None):
    spec = resolved.spec
    trainer_hash = resolved.trainer_hash
    app = modal.App(resolved.trainer_app_name)
    resource = spec.trainer["resources"]
    from .deployment import trainer_deployment_env

    env = {
        **trainer_deployment_env(),
        **deployment_env(spec.trainer["env"]),
        "LILO_APP_NAME": spec.deployment["frontend"],
    }

    @app.function(
        name="trainer",
        serialized=True,
        image=image if image is not None else image_for(spec.trainer["backend"]),
        gpu=resource["gpu"],
        region=spec.deployment["modal"]["region"],
        cpu=resource["cpu"],
        memory=resource["memory_mib"],
        timeout=resource["timeout_s"],
        # Admission/reconciliation caps each definition. A function-wide cap
        # would block new definitions behind retained jobs sharing this app.
        max_containers=None,
        min_containers=0,
        single_use_containers=True,
        volumes=volumes_for(spec),
        secrets=secrets_for(spec, training=True),
        env=env,
    )
    def trainer(instance_id: str, config_json: str):
        record = DeploymentRecord.model_validate_json(config_json)
        if record.trainer_hash != trainer_hash:
            raise ValueError("trainer settings do not match the deployed app")
        run_trainer(record, instance_id)

    return app, trainer


def run_trainer(resolved, instance_id):
    from modal.config import config
    from .kv import shared_kv
    from .serve import run_engine_with_backend

    spec = resolved.spec
    settings = backend_config(spec, resolved.asset_path)
    # Assets are prepared by the frontend before demand is registered. Reload once
    # on startup to see the committed exact snapshot; never race a trainer download.
    volumes_for(spec)["/assets"].reload()
    env = {
        **deployment_env(spec.trainer["env"]),
        "LILO_APP_NAME": spec.deployment["frontend"],
        "LILO_BACKEND_CONFIG": json.dumps(settings),
        "LILO_BASE_MODEL": spec.model["id"],
        "LILO_BASE_MODEL_REVISION": spec.model["revision"],
        "LILO_DEFINITION_ID": resolved.definition_id,
        "LILO_CHECKPOINT_VOLUME": spec.deployment["storage"]["checkpoints"],
        "LILO_BULLETIN_ROOT": "/bulletin",
        "LILO_BULLETIN_VOLUME": spec.deployment["storage"]["bulletin"],
        "LILO_DEFINITION_REVISION": resolved.generation,
    }
    executor = (
        "lilo.backends.miles_lora:build_executor"
        if spec.trainer["backend"] == "miles"
        else "lilo.backends.megatron_fft:build_executor"
    )

    async def failed(error):
        await shared_kv().put(
            f"deployment_failure:{resolved.definition_id}",
            {"error": str(error), "instance_id": instance_id},
        )

    run_engine_with_backend(
        shared_kv(),
        executor,
        definition_id=resolved.definition_id,
        revision=config["image_id"],
        instance_id=instance_id,
        backend_env=env,
        nproc=1
        if spec.trainer["backend"] == "miles"
        else gpu_count(spec.trainer["resources"]),
        max_models=spec.trainer["engine"]["max_clients_per_instance"],
        sampler_persistence_concurrency=spec.trainer["engine"][
            "sampler_persistence_concurrency"
        ],
        on_startup_error=failed,
    )


def definition_from_spec(resolved, *, register_trainer=True, image=None):
    spec = resolved.spec
    native = serving_options(spec)
    definition = SimpleNamespace(
        DEFINITION_ID=resolved.definition_id,
        MODEL_NAME=spec.model["id"],
        MODEL_REVISION=spec.model["revision"],
        HF_CHECKPOINT=resolved.asset_path,
        PARAMETERIZATION=spec.model["parameterization"],
        CATALOG_VISIBLE=resolved.active,
        ROUTING_DEFAULT=spec.routing["default"],
        SAMPLING_DEFAULT=spec.routing["sampling_default"],
        DEPLOYMENT_NAME=spec.name,
        RESOLVED=resolved,
        MAX_CONTEXT_LENGTH=spec.model["max_context_length"],
        TRAINER_MODELS_PER_INSTANCE=spec.trainer["engine"]["max_clients_per_instance"],
        TRAINER_MAX_CONTAINERS=spec.trainer["scaling"]["max_instances"],
        ROLLOUT_GPUS=gpu_count(spec.inference["resources"]),
        ROLLOUT_TENSOR_PARALLEL_SIZE=native.get(
            "tp_size", gpu_count(spec.inference["resources"])
        )
        // (native.get("dp_size", 1) if native.get("enable_dp_attention") else 1),
    )
    if register_trainer:
        definition.ENGINE_FUNCTION = modal.Function.from_name(
            resolved.trainer_app_name,
            "trainer",
            environment_name=spec.deployment["modal"]["environment"],
        )
    return definition


def pool_deployment(definition_id):
    for row in manifest_from_env():
        if row.definition_id == definition_id:
            return row
    return None


def pool_environment(definition_id):
    resolved = pool_deployment(definition_id)
    if resolved is None:
        raise ValueError(f"missing recorded deployment: {definition_id}")
    return {POOL_CONFIG_ENV: resolved.model_dump_json()}


def build_rollout_app(resolved, pool, *, image=None):
    """Create one frozen-base LoRA pool or one FFT latest/pinned/base pool."""
    spec = resolved.spec
    lora = spec.model["parameterization"] == "lora"
    if pool.definition_id != resolved.definition_id:
        raise ValueError("pool generation does not match deployment")
    app = modal.App(pool.app_name)
    resources, scaling = spec.inference["resources"], spec.inference["scaling"]
    options = {
        "context_length": spec.model["max_context_length"],
        "tp_size": gpu_count(resources),
        "mem_fraction_static": 0.8,
        "max_running_requests": 32,
        "weight_loader_disable_mmap": True,
        **serving_options(spec),
    }
    if lora:
        from lilo.backends.miles_config import MilesBackendConfig

        config = backend_config(spec)["miles"]
        targets = MilesBackendConfig(**config).peft_target_modules
        options.update(enable_lora=True, max_lora_rank=config["max_lora_rank"])
        options.setdefault("lora_target_modules", list(targets))
        options.setdefault("max_loaded_loras", 64)
        options.setdefault("max_loras_per_batch", 8)
    else:
        options["enable_cpu_weight_cache"] = True
    minimum = getattr(pool, "min_containers", None)
    maximum = getattr(pool, "max_containers", None)
    window = getattr(pool, "scaledown_window", None)

    # Serialized class captures the spec; no model-specific module is imported.
    @app.server(
        name="Server",
        serialized=True,
        image=image if image is not None else image_for("sglang"),
        gpu=resources["gpu"],
        cpu=resources["cpu"],
        memory=resources["memory_mib"],
        volumes=volumes_for(spec),
        secrets=secrets_for(spec),
        env=deployment_env(spec.inference["env"]),
        min_containers=scaling["min_replicas"] if minimum is None else minimum,
        max_containers=scaling["max_replicas"] if maximum is None else maximum,
        target_concurrency=scaling["target_concurrency"],
        scaledown_window=scaling["scaledown_window_s"] if window is None else window,
        startup_timeout=1200,
        exit_grace_period=300,
        port=8000,
        routing_region=spec.deployment["modal"]["region"],
        compute_region=spec.deployment["modal"]["region"],
    )
    class Server:
        @modal.enter()
        def start(self):
            import subprocess
            import sys
            from lilo.inference.serving import (
                start_lora_sidecar,
                start_fft_sidecar,
                supervise_children,
                terminate,
                wait_http,
            )

            self.sglang = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "lilo.inference.native_sglang",
                    resolved.asset_path,
                    json.dumps(options),
                ],
                start_new_session=True,
            )
            try:
                wait_http("http://127.0.0.1:8001/health", self.sglang, 1200)
                kwargs = dict(
                    port=8000,
                    sglang_port=8001,
                    bulletin_root="/bulletin",
                    bulletin_volume=spec.deployment["storage"]["bulletin"],
                )
                self.sidecar = (
                    start_lora_sidecar(**kwargs)
                    if lora
                    else start_fft_sidecar(
                        **kwargs,
                        model_path=resolved.asset_path,
                        run_id=pool.model_id,
                        pinned_version=None if pool.latest else pool.version,
                    )
                )
                self.supervisor = supervise_children(self.sglang, self.sidecar)
                wait_http("http://127.0.0.1:8000/health", self.sidecar, 1200)
            except BaseException:
                terminate(getattr(self, "sidecar", None))
                terminate(self.sglang)
                raise

        @modal.exit()
        def stop(self):
            from lilo.inference.serving import terminate

            terminate(getattr(self, "sidecar", None))
            terminate(getattr(self, "sglang", None))

    return app, Server


def provision_pool(record, pool):
    """Ask the saved inference app to create a pool using its original code."""
    provision = modal.Function.from_name(
        record.inference_app_name,
        "provision",
        environment_name=record.spec.deployment["modal"]["environment"],
    )
    return provision.remote(record.model_dump_json(), pool.as_dict())


def build_inference_app(record, *, image=None):
    """Freeze pool-building code so idle pools can restart after frontend upgrades."""
    from .image_dependencies import (
        CORE_PACKAGES,
        STITCH_PACKAGE,
        TINKER_PACKAGE,
        ignore_config_source,
    )

    if image is None:
        image = (
            modal.Image.debian_slim(python_version="3.12")
            .apt_install("git")
            .pip_install(
                *CORE_PACKAGES, STITCH_PACKAGE, TINKER_PACKAGE, "huggingface-hub"
            )
            .add_local_python_source("lilo", copy=True, ignore=ignore_config_source)
        )
    app = modal.App(record.inference_app_name)
    inference_hash = record.inference_hash

    @app.function(name="provision", image=image, serialized=True, timeout=1800)
    def provision(config_json: str, pool_data: dict):
        from .lora_pool import LoraPoolSpec, deploy_pool as deploy_lora
        from .fft_pool import FFTPoolSpec, deploy_pool as deploy_fft

        saved = DeploymentRecord.model_validate_json(config_json)
        if saved.inference_hash != inference_hash:
            raise ValueError("inference settings do not match the deployed app")
        if pool_data["definition_id"] != saved.definition_id:
            raise ValueError("pool definition does not match deployment")
        if saved.spec.model["parameterization"] == "lora":
            return deploy_lora(LoraPoolSpec.from_dict(pool_data), record=saved)
        return deploy_fft(FFTPoolSpec.from_dict(pool_data), record=saved)

    return app, provision
