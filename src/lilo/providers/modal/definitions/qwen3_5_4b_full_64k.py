from __future__ import annotations

import os

import modal

from ..checkpoint_storage import (
    CHECKPOINT_ROOT,
    CHECKPOINT_VOLUME_NAME,
    checkpoint_volume,
)
from ..deployment import trainer_max_containers

MODEL_NAME = "Qwen/Qwen3.5-4B"
HF_CHECKPOINT = "/assets/Qwen3.5-4B"
MICRO_BATCH_SIZE = 1
MAX_CONTEXT_LENGTH = 65_536
MAX_TOKENS_PER_MICROBATCH = MAX_CONTEXT_LENGTH
DEFINITION_ID = "qwen3_5_4b_full_64k"
PARAMETERIZATION = "full"
CATALOG_VISIBLE = True
TRAINER_MODELS_PER_INSTANCE = 1
GPU_TYPE = "H100"
GPUS = 4
TENSOR_MODEL_PARALLEL_SIZE = 2
CONTEXT_PARALLEL_SIZE = 2
DATA_PARALLEL_SIZE = GPUS // (TENSOR_MODEL_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE)
MTP_NUM_LAYERS = 0
RECOMPUTE_GRANULARITY = "full"
RECOMPUTE_METHOD = "uniform"
RECOMPUTE_NUM_LAYERS = 1
LOSS_SCALE = 1.0
DEFER_FP32_LOGITS = True
FP32_LM_HEAD = True
ROLLOUT_GPU_TYPE = "H100"
ROLLOUT_GPUS = 1
ROLLOUT_TENSOR_PARALLEL_SIZE = 1
ROLLOUT_EXPERT_PARALLEL_SIZE = 1
ROLLOUT_EXPERT_TENSOR_PARALLEL_SIZE = 1
ROLLOUT_MEMORY_FRACTION = 0.85
ROLLOUT_MAX_RUNNING_REQUESTS = 32
ROLLOUT_MAX_QUEUED_REQUESTS = 4
ROLLOUT_TARGET_CONCURRENCY = 16
ROLLOUT_CPU_WEIGHT_CACHE_MAX_COMPILE_GROUP_GB = 16
BULLETIN_ROOT = "/bulletin"
BULLETIN_VOLUME_NAME = "lilo-snapshot-bulletin"
app = modal.App(f"lilo-{DEFINITION_ID}")

if modal.is_local():
    from ..megatron_image import image
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
}
api_secret = modal.Secret.from_name(
    "lilo-api",
    required_keys=["TINKER_API_KEY"],
)
proxy_secret = modal.Secret.from_name(
    "lilo-proxy",
    required_keys=["MODAL_PROXY_TOKEN_ID", "MODAL_PROXY_TOKEN_SECRET"],
)


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{GPUS}",
    volumes=TRAINER_VOLUMES,
    secrets=[api_secret, proxy_secret],
    timeout=86_400,
    max_containers=trainer_max_containers(),
    single_use_containers=True,
)
def qwen3_5_4b_full_64k(instance_id: str) -> None:
    import json

    from huggingface_hub import snapshot_download
    from modal.config import config

    from lilo.providers.modal.kv import shared_kv
    from lilo.providers.modal.serve import run_engine_with_backend

    if not os.path.exists(HF_CHECKPOINT):
        snapshot_download(repo_id=MODEL_NAME, local_dir=HF_CHECKPOINT)
        assets.commit()
    backend_config = {
        "megatron": {
            "hf_checkpoint": HF_CHECKPOINT,
            "tensor_model_parallel_size": TENSOR_MODEL_PARALLEL_SIZE,
            "context_parallel_size": CONTEXT_PARALLEL_SIZE,
            "sequence_parallel": True,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "max_tokens_per_microbatch": MAX_TOKENS_PER_MICROBATCH,
            "seq_length": MAX_CONTEXT_LENGTH,
            "defer_fp32_logits": DEFER_FP32_LOGITS,
            "fp32_lm_head": FP32_LM_HEAD,
            "use_distributed_optimizer": True,
            "provider_overrides": {
                "mtp_num_layers": MTP_NUM_LAYERS,
                "recompute_granularity": RECOMPUTE_GRANULARITY,
                "recompute_method": RECOMPUTE_METHOD,
                "recompute_num_layers": RECOMPUTE_NUM_LAYERS,
            },
            "optimizer": {
                "lr": 1e-4,
                "min_lr": 1e-4,
                "loss_scale": LOSS_SCALE,
            },
        },
        "checkpoint_dir": CHECKPOINT_ROOT,
    }
    run_engine_with_backend(
        shared_kv(),
        "lilo.backends.megatron_fft:build_executor",
        definition_id=DEFINITION_ID,
        revision=config["image_id"],
        instance_id=instance_id,
        backend_env={
            "LILO_BACKEND_CONFIG": json.dumps(backend_config),
            "LILO_BASE_MODEL": MODEL_NAME,
            "LILO_DEFINITION_ID": DEFINITION_ID,
            "LILO_CHECKPOINT_VOLUME": CHECKPOINT_VOLUME_NAME,
            "LILO_BULLETIN_ROOT": BULLETIN_ROOT,
            "LILO_BULLETIN_VOLUME": BULLETIN_VOLUME_NAME,
            "LILO_DEFINITION_REVISION": config["image_id"],
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
        },
        nproc=GPUS,
        max_models=1,
    )


ENGINE_FUNCTION = qwen3_5_4b_full_64k
