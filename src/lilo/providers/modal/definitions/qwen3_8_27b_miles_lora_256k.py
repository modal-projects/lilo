from __future__ import annotations

import json
import os

import modal
import modal.experimental
from modal.config import config

from ..checkpoint_storage import (
    CHECKPOINT_ROOT,
    CHECKPOINT_VOLUME_NAME,
    checkpoint_volume,
)
from ..deployment import trainer_deployment_env, trainer_max_containers
from ..kv import shared_kv
from ..serve import run_engine_with_backend

MODEL_NAME = "Qwen/Qwen3.8-27B"
HF_CHECKPOINT = "/assets/Qwen3.8-27B"
DEFINITION_ID = "qwen3_8_27b_miles_lora_256k"
PARAMETERIZATION = "lora"
CATALOG_VISIBLE = False
MAX_CONTEXT_LENGTH = 262_144

GPU_TYPE = "H200"
GPUS = 8
TRAINER_NODES = 2
# 16 = TP2 x CP8 x DP1, the topology raw Miles runs 256k on. TP2 keeps the 27B
# base weights at ~27 GB/GPU of the 141 GB H200, and spending the rest of the
# world size on CP is what shrinks the activation working set: 32k tokens per
# rank instead of 87k under TP8 x CP3.
TENSOR_MODEL_PARALLEL_SIZE = 2
CONTEXT_PARALLEL_SIZE = 8
# Divisible by 2 * cp (zigzag chunks) and by tp (sequence parallelism).
_SEQ_ALIGNMENT = 2 * CONTEXT_PARALLEL_SIZE * TENSOR_MODEL_PARALLEL_SIZE
SEQ_LENGTH = -(-MAX_CONTEXT_LENGTH // _SEQ_ALIGNMENT) * _SEQ_ALIGNMENT
MAX_TOKENS_PER_GPU = SEQ_LENGTH // CONTEXT_PARALLEL_SIZE
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
# Megatron's default 10-minute collective timeout is the NCCL watchdog budget
# for a single collective. At 250k tokens the inter-node context-parallel
# all-to-all runs behind a straggler's recompute, so a rank can sit in one
# collective far longer than ten minutes without anything being wrong.
DISTRIBUTED_TIMEOUT_MINUTES = 120
# Coalescing a whole rollout's datums into one Miles call turns a step into a
# single multi-thousand-collective forward_backward across both nodes, where
# one desynchronized rank wedges every process group. One datum per call keeps
# the collective chains short; gradients still accumulate until optim_step.
MAX_FORWARD_BACKWARD_BATCH = 1

ROLLOUT_GPU_TYPE = "H200"
ROLLOUT_GPUS = 4
ROLLOUT_TENSOR_PARALLEL_SIZE = 4
ROLLOUT_EXPERT_PARALLEL_SIZE = 1
ROLLOUT_EXPERT_TENSOR_PARALLEL_SIZE = 1
ROLLOUT_MEMORY_FRACTION = 0.8
ROLLOUT_MAX_RUNNING_REQUESTS = 4
ROLLOUT_MAX_QUEUED_REQUESTS = 8
ROLLOUT_TARGET_CONCURRENCY = 2
ROLLOUT_MAX_LOADED_LORAS = 256
ROLLOUT_MIN_CONTAINERS = 2
ROLLOUT_MAX_CONTAINERS = 2
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

with image.imports():
    from huggingface_hub import snapshot_download

    from ..ray_cluster import start_trainer_cluster

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


def ensure_assets() -> None:
    if not os.path.exists(HF_CHECKPOINT):
        snapshot_download(repo_id=MODEL_NAME, local_dir=HF_CHECKPOINT)
        assets.commit()


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{GPUS}",
    volumes=TRAINER_VOLUMES,
    secrets=[api_secret, proxy_secret, huggingface_secret],
    env=trainer_deployment_env(),
    timeout=86_400,
    max_containers=trainer_max_containers(),
    single_use_containers=True,
    experimental_options={"efa_enabled": True},
)
@modal.experimental.clustered(TRAINER_NODES, rdma=True)
def qwen3_8_27b_miles_lora_256k(instance_id: str) -> None:
    ray_address = start_trainer_cluster(
        TRAINER_NODES,
        before_head=ensure_assets,
        before_worker_join=assets.reload,
    )
    if ray_address is None:
        return
    run_trainer(instance_id, ray_address=ray_address)


def backend_config(
    instance_id: str,
    *,
    deterministic_training: bool = False,
) -> dict:
    config = {
        "miles": {
            "hf_checkpoint": HF_CHECKPOINT,
            "model_type": "qwen3.8-27B",
            "actor_num_gpus_per_node": GPUS,
            "actor_num_nodes": TRAINER_NODES,
            "tensor_model_parallel_size": TENSOR_MODEL_PARALLEL_SIZE,
            "context_parallel_size": CONTEXT_PARALLEL_SIZE,
            "max_lora_slots": MAX_LORA_SLOTS,
            "max_lora_rank": MAX_LORA_RANK,
            "default_lora_alpha": DEFAULT_LORA_ALPHA,
            "target_modules": TARGET_MODULES,
            "max_tokens_per_gpu": MAX_TOKENS_PER_GPU,
            "extra_args": (
                "--seq-length",
                str(SEQ_LENGTH),
                "--recompute-granularity",
                "full",
                "--recompute-method",
                "uniform",
                "--recompute-num-layers",
                "1",
                "--distributed-timeout-minutes",
                str(DISTRIBUTED_TIMEOUT_MINUTES),
            ),
        },
        "checkpoint_dir": CHECKPOINT_ROOT,
        "capture_dir": f"{CHECKPOINT_ROOT}/.captures/{instance_id}",
    }
    if deterministic_training:
        config["miles"].update(
            tp_reduce_precision="float64", deterministic_attention=True
        )
    return config


def run_trainer(
    instance_id: str,
    *,
    definition_id: str = DEFINITION_ID,
    max_models: int = MAX_LORA_SLOTS,
    deterministic_training: bool = False,
    ray_address: str | None = None,
) -> None:
    ensure_assets()
    config_payload = backend_config(
        instance_id,
        deterministic_training=deterministic_training,
    )
    run_engine_with_backend(
        shared_kv(),
        "lilo.backends.miles_lora:build_executor",
        definition_id=definition_id,
        revision=config["image_id"],
        instance_id=instance_id,
        backend_env={
            "LILO_BACKEND_CONFIG": json.dumps(config_payload),
            "LILO_BASE_MODEL": MODEL_NAME,
            "LILO_DEFINITION_ID": definition_id,
            "LILO_CHECKPOINT_VOLUME": CHECKPOINT_VOLUME_NAME,
            "LILO_BULLETIN_ROOT": BULLETIN_ROOT,
            "LILO_BULLETIN_VOLUME": BULLETIN_VOLUME_NAME,
            "LILO_DEFINITION_REVISION": config["image_id"],
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            **({"LILO_RAY_ADDRESS": ray_address} if ray_address else {}),
        },
        nproc=1,
        max_models=max_models,
        sampler_persistence_concurrency=8,
        max_forward_backward_batch=MAX_FORWARD_BACKWARD_BATCH,
    )


ENGINE_FUNCTION = qwen3_8_27b_miles_lora_256k
