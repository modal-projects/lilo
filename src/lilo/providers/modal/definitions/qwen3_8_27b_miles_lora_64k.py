from __future__ import annotations

import os

import modal

from ..checkpoint_storage import (
    CHECKPOINT_ROOT,
    CHECKPOINT_VOLUME_NAME,
    checkpoint_volume,
)
from ..deployment import trainer_deployment_env, trainer_max_containers
from ..kernel_cache import KERNEL_CACHE_ENV, KERNEL_CACHE_ROOT, kernel_cache_volume

MODEL_NAME = "Qwen/Qwen3.8-27B"
HF_CHECKPOINT = "/assets/Qwen3.8-27B"
DEFINITION_ID = "qwen3_8_27b_miles_lora_64k"
PARAMETERIZATION = "lora"
CATALOG_VISIBLE = False
MAX_CONTEXT_LENGTH = 65_536

GPU_TYPE = "H200"
GPUS = 8
TENSOR_MODEL_PARALLEL_SIZE = 4
CONTEXT_PARALLEL_SIZE = 2
MAX_LORA_SLOTS = 6
MAX_LORA_RANK = 32
DEFAULT_LORA_ALPHA = 32
TARGET_MODULES = (
    "linear_qkv",
    "linear_proj",
    "linear_fc1",
    "linear_fc2",
)
TRAINER_MODELS_PER_INSTANCE = MAX_LORA_SLOTS

ROLLOUT_GPU_TYPE = "H200"
ROLLOUT_GPUS = 1
ROLLOUT_TENSOR_PARALLEL_SIZE = 1
ROLLOUT_EXPERT_PARALLEL_SIZE = 1
ROLLOUT_EXPERT_TENSOR_PARALLEL_SIZE = 1
ROLLOUT_MEMORY_FRACTION = 0.8
ROLLOUT_MAX_RUNNING_REQUESTS = 16
ROLLOUT_MAX_QUEUED_REQUESTS = 8
ROLLOUT_TARGET_CONCURRENCY = 8
ROLLOUT_MAX_LOADED_LORAS = 256
ROLLOUT_MIN_CONTAINERS = 8
ROLLOUT_MAX_CONTAINERS = 8
ROLLOUT_MAX_LORAS_PER_BATCH = 8
ROLLOUT_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

BULLETIN_ROOT = "/bulletin"
BULLETIN_VOLUME_NAME = "lilo-snapshot-bulletin"
app = modal.App(f"lilo-{DEFINITION_ID}")

if modal.is_local():
    from ..miles_image import image
else:
    image = modal.Image.debian_slim()

assets = modal.Volume.from_name("lilo-model-assets", create_if_missing=True)
bulletin = modal.Volume.from_name(
    BULLETIN_VOLUME_NAME,
    create_if_missing=True,
    version=2,
)
TRAINER_VOLUMES = {
    "/assets": assets,
    BULLETIN_ROOT: bulletin,
    CHECKPOINT_ROOT: checkpoint_volume,
    KERNEL_CACHE_ROOT: kernel_cache_volume,
}
api_secret = modal.Secret.from_name(
    "lilo-api",
    required_keys=["TINKER_API_KEY"],
)
proxy_secret = modal.Secret.from_name(
    "lilo-proxy",
    required_keys=["MODAL_PROXY_TOKEN_ID", "MODAL_PROXY_TOKEN_SECRET"],
)
huggingface_secret = modal.Secret.from_name("huggingface-secret")


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{GPUS}",
    volumes=TRAINER_VOLUMES,
    secrets=[api_secret, proxy_secret, huggingface_secret],
    env=trainer_deployment_env(),
    timeout=86_400,
    max_containers=trainer_max_containers(),
    single_use_containers=True,
)
def qwen3_8_27b_miles_lora_64k(instance_id: str) -> None:
    run_trainer(instance_id)


def run_trainer(
    instance_id: str,
    *,
    definition_id: str = DEFINITION_ID,
    max_models: int = MAX_LORA_SLOTS,
    deterministic_training: bool = False,
) -> None:
    import json

    from huggingface_hub import snapshot_download
    from modal.config import config

    from lilo.providers.modal.kv import shared_kv
    from lilo.providers.modal.serve import run_engine_with_backend

    if not os.path.exists(HF_CHECKPOINT):
        snapshot_download(repo_id=MODEL_NAME, local_dir=HF_CHECKPOINT)
        assets.commit()
    backend_config = {
        "miles": {
            "hf_checkpoint": HF_CHECKPOINT,
            "model_type": "qwen3.8-27B",
            "actor_num_gpus_per_node": GPUS,
            "tensor_model_parallel_size": TENSOR_MODEL_PARALLEL_SIZE,
            "context_parallel_size": CONTEXT_PARALLEL_SIZE,
            "max_lora_slots": MAX_LORA_SLOTS,
            "max_lora_rank": MAX_LORA_RANK,
            "default_lora_alpha": DEFAULT_LORA_ALPHA,
            "target_modules": TARGET_MODULES,
            "max_tokens_per_gpu": MAX_CONTEXT_LENGTH // CONTEXT_PARALLEL_SIZE,
            "extra_args": (
                "--seq-length",
                str(MAX_CONTEXT_LENGTH),
                "--recompute-granularity",
                "full",
                "--recompute-method",
                "uniform",
                "--recompute-num-layers",
                "1",
            ),
        },
        "checkpoint_dir": CHECKPOINT_ROOT,
    }
    if deterministic_training:
        backend_config["miles"].update(
            tp_reduce_precision="float64", deterministic_attention=True
        )
    run_engine_with_backend(
        shared_kv(),
        "lilo.backends.miles_lora:build_executor",
        definition_id=definition_id,
        revision=config["image_id"],
        instance_id=instance_id,
        backend_env={
            "LILO_BACKEND_CONFIG": json.dumps(backend_config),
            "LILO_BASE_MODEL": MODEL_NAME,
            "LILO_DEFINITION_ID": definition_id,
            "LILO_CHECKPOINT_VOLUME": CHECKPOINT_VOLUME_NAME,
            "LILO_BULLETIN_ROOT": BULLETIN_ROOT,
            "LILO_BULLETIN_VOLUME": BULLETIN_VOLUME_NAME,
            "LILO_DEFINITION_REVISION": config["image_id"],
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            **KERNEL_CACHE_ENV,
        },
        nproc=1,
        max_models=max_models,
        sampler_persistence_concurrency=8,
    )


ENGINE_FUNCTION = qwen3_8_27b_miles_lora_64k
