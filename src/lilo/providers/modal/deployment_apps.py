"""Modal app builders shared by all Python-configured model deployments.

Trainers and inference provisioners are independently deployed apps. Pools are
created on demand with the existing LoRA/FFT pool lifecycle.
"""

from __future__ import annotations

import json
import subprocess
import sys
from importlib import import_module
from types import SimpleNamespace

import modal
import modal.experimental
from modal.config import config

from lilo.deployments import DeploymentRecord, validate_frontend
from lilo.inference.serving import (
    start_fft_sidecar,
    start_lora_sidecar,
    supervise_children,
    terminate,
    wait_http,
)

from .deployment import trainer_deployment_env
from .deployment_records import (
    manifest_from_env,
)
from .fft_pool import FFTPoolSpec
from .fft_pool import deploy_pool as deploy_fft
from .image_dependencies import (
    CORE_PACKAGES,
    STITCH_PACKAGE,
    TINKER_PACKAGE,
    ignore_config_source,
)
from .kv import shared_kv
from .lora_pool import LoraPoolSpec
from .lora_pool import deploy_pool as deploy_lora
from .ray_cluster import start_trainer_cluster
from .serve import run_engine_with_backend


def frontend_settings():
    deployments = manifest_from_env()
    active = [row.spec for row in deployments if row.active]
    validate_frontend(active)
    if len({row.definition_id for row in deployments}) != len(deployments):
        raise ValueError("duplicate deployment generation")
    return deployments[0]


def image_for(backend):
    if not modal.is_local():
        return modal.Image.debian_slim()
    # Image definitions are imported only by the selected worker's deploy process.
    modules = {
        "miles": ".miles_image",
        "megatron": ".megatron_image",
        "sglang": ".rollout_image",
    }
    return import_module(modules[backend], __package__).image


def volumes_for(record):
    storage = record.platform["storage"]
    return {
        "/assets": modal.Volume.from_name(storage["assets"], create_if_missing=True),
        "/checkpoints": modal.Volume.from_name(
            storage["checkpoints"], create_if_missing=True, version=2
        ),
        "/bulletin": modal.Volume.from_name(
            storage["bulletin"], create_if_missing=True, version=2
        ),
    }


def secrets_for(record, *, training=False):
    names = record.platform["secrets"]
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
    resource = spec.trainer

    env = {
        **trainer_deployment_env(),
        **deployment_env(spec.trainer.env),
        "LILO_APP_NAME": resolved.platform["frontend"],
    }

    def trainer(instance_id: str, config_json: str):
        record = DeploymentRecord.model_validate_json(config_json)
        if record.trainer_hash != trainer_hash:
            raise ValueError("trainer settings do not match the deployed app")
        run_trainer(record, instance_id)

    if resource.nodes > 1:
        trainer = modal.experimental.clustered(resource.nodes, rdma=True)(trainer)
    trainer = app.function(
        name="trainer",
        serialized=True,
        image=image if image is not None else image_for(spec.trainer.backend),
        gpu=f"{resource.gpu}:{resource.gpus_per_node}",
        region=resolved.platform["modal"]["region"],
        cpu=resource.cpu,
        memory=resource.memory_mib,
        timeout=spec.trainer.timeout_s,
        # Admission/reconciliation caps each definition. A function-wide cap
        # would block new definitions behind retained jobs sharing this app.
        max_containers=None,
        min_containers=0,
        single_use_containers=True,
        volumes=volumes_for(resolved),
        secrets=secrets_for(resolved, training=True),
        env=env,
        experimental_options={"efa_enabled": True} if resource.nodes > 1 else {},
    )(trainer)

    return app, trainer


def run_trainer(resolved, instance_id):
    spec = resolved.spec
    settings = resolved.trainer_settings
    # Assets are prepared by the frontend before demand is registered. Reload once
    # on startup to see the committed exact snapshot; never race a trainer download.
    assets = volumes_for(resolved)["/assets"]
    if spec.trainer.nodes > 1:
        ray_address = start_trainer_cluster(
            spec.trainer.nodes,
            before_head=assets.reload,
            before_worker_join=assets.reload,
        )
        if ray_address is None:
            return
    else:
        assets.reload()
        ray_address = None
    env = {
        **deployment_env(spec.trainer.env),
        "LILO_APP_NAME": resolved.platform["frontend"],
        "LILO_BACKEND_CONFIG": json.dumps(settings),
        "LILO_BASE_MODEL": spec.model,
        "LILO_BASE_MODEL_REVISION": spec.revision,
        "LILO_DEFINITION_ID": resolved.definition_id,
        "LILO_CHECKPOINT_VOLUME": resolved.platform["storage"]["checkpoints"],
        "LILO_BULLETIN_ROOT": "/bulletin",
        "LILO_BULLETIN_VOLUME": resolved.platform["storage"]["bulletin"],
        "LILO_DEFINITION_REVISION": resolved.generation,
    }
    if ray_address is not None:
        env["LILO_RAY_ADDRESS"] = ray_address
    executor = (
        "lilo.backends.miles_lora:build_executor"
        if spec.trainer.backend == "miles"
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
        nproc=1 if spec.trainer.backend == "miles" else spec.trainer.gpus_per_node,
        max_models=spec.trainer.max_clients_per_instance,
        sampler_persistence_concurrency=spec.trainer.sampler_persistence_concurrency,
        on_startup_error=failed,
    )


def definition_from_spec(resolved, *, register_trainer=True, image=None):
    spec = resolved.spec
    serving = resolved.inference_settings
    definition = SimpleNamespace(
        DEFINITION_ID=resolved.definition_id,
        MODEL_NAME=spec.model,
        MODEL_REVISION=spec.revision,
        HF_CHECKPOINT=resolved.asset_path,
        PARAMETERIZATION=spec.parameterization,
        DEPLOYMENT_NAME=spec.name,
        RESOLVED=resolved,
        MAX_CONTEXT_LENGTH=spec.max_context_length,
        TRAINER_MODELS_PER_INSTANCE=spec.trainer.max_clients_per_instance,
        TRAINER_MAX_CONTAINERS=spec.trainer.max_instances,
        ROLLOUT_GPUS=spec.inference.gpus_per_node,
        ROLLOUT_TENSOR_PARALLEL_SIZE=serving.get(
            "tp_size", spec.inference.gpus_per_node
        )
        // (serving.get("dp_size", 1) if serving.get("enable_dp_attention") else 1),
    )
    if register_trainer:
        definition.ENGINE_FUNCTION = modal.Function.from_name(
            resolved.trainer_app_name,
            "trainer",
            environment_name=resolved.platform["modal"]["environment"],
        )
    return definition


def build_rollout_app(resolved, pool, *, image=None):
    """Create one frozen-base LoRA pool or one FFT latest/pinned/base pool."""
    spec = resolved.spec
    lora = spec.parameterization == "lora"
    if pool.definition_id != resolved.definition_id:
        raise ValueError("pool generation does not match deployment")
    app = modal.App(pool.app_name)
    inference = spec.inference
    options = resolved.inference_settings
    minimum = inference.min_replicas
    maximum = inference.max_replicas
    window = inference.scaledown_window_s
    if isinstance(pool, FFTPoolSpec):
        minimum = minimum if pool.min_containers is None else pool.min_containers
        maximum = maximum if pool.max_containers is None else pool.max_containers
        window = window if pool.scaledown_window is None else pool.scaledown_window

    # Serialized class captures the spec; no model-specific module is imported.
    @app.server(
        name="Server",
        serialized=True,
        image=image if image is not None else image_for("sglang"),
        gpu=f"{inference.gpu}:{inference.gpus_per_node}",
        cpu=inference.cpu,
        memory=inference.memory_mib,
        volumes=volumes_for(resolved),
        secrets=secrets_for(resolved),
        env=deployment_env(inference.env),
        min_containers=minimum,
        max_containers=maximum,
        target_concurrency=inference.target_concurrency,
        scaledown_window=window,
        startup_timeout=inference.startup_timeout_s,
        exit_grace_period=300,
        port=8000,
        routing_region=resolved.platform["modal"]["region"],
        compute_region=resolved.platform["modal"]["region"],
    )
    class Server:
        @modal.enter()
        def start(self):
            self.sidecar = None
            self.sglang = None

            self.sglang = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "lilo.inference.sglang",
                    resolved.asset_path,
                    json.dumps(options),
                ],
                start_new_session=True,
            )
            try:
                wait_http(
                    "http://127.0.0.1:8001/health",
                    self.sglang,
                    inference.startup_timeout_s,
                )
                kwargs = dict(
                    port=8000,
                    sglang_port=8001,
                    bulletin_root="/bulletin",
                    bulletin_volume=resolved.platform["storage"]["bulletin"],
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
                wait_http(
                    "http://127.0.0.1:8000/health",
                    self.sidecar,
                    inference.startup_timeout_s,
                )
            except BaseException:
                terminate(self.sidecar)
                terminate(self.sglang)
                raise

        @modal.exit()
        def stop(self):
            terminate(self.sidecar)
            terminate(self.sglang)

    return app, Server


def build_inference_app(record, *, image=None):
    """Freeze pool-building code so idle pools can restart after frontend upgrades."""

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
        saved = DeploymentRecord.model_validate_json(config_json)
        if saved.inference_hash != inference_hash:
            raise ValueError("inference settings do not match the deployed app")
        if pool_data["definition_id"] != saved.definition_id:
            raise ValueError("pool definition does not match deployment")
        if saved.spec.parameterization == "lora":
            return deploy_lora(LoraPoolSpec.from_dict(pool_data), record=saved)
        return deploy_fft(FFTPoolSpec.from_dict(pool_data), record=saved)

    return app, provision
