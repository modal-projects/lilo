"""Short deterministic multi-LoRA parity experiments on the existing Miles backend."""

# ruff: noqa: F401
import modal
from ..deployment import trainer_max_containers
from .qwen3_5_9b_base_miles_lora_16k import (
    MODEL_NAME,
    HF_CHECKPOINT,
    PARAMETERIZATION,
    MAX_CONTEXT_LENGTH,
    GPU_TYPE,
    GPUS,
    MAX_LORA_SLOTS,
    MAX_LORA_RANK,
    DEFAULT_LORA_ALPHA,
    TARGET_MODULES,
    TENSOR_MODEL_PARALLEL_SIZE,
    BULLETIN_ROOT,
    BULLETIN_VOLUME_NAME,
    TRAINER_VOLUMES,
    assets,
    bulletin,
    image,
    api_secret,
    proxy_secret,
    run_trainer,
    ROLLOUT_GPU_TYPE,
    ROLLOUT_GPUS,
    ROLLOUT_TENSOR_PARALLEL_SIZE,
    ROLLOUT_EXPERT_PARALLEL_SIZE,
    ROLLOUT_EXPERT_TENSOR_PARALLEL_SIZE,
    ROLLOUT_MEMORY_FRACTION,
    ROLLOUT_MAX_QUEUED_REQUESTS,
    ROLLOUT_TARGET_CONCURRENCY,
    ROLLOUT_MAX_LOADED_LORAS,
    ROLLOUT_MAX_LORAS_PER_BATCH,
    ROLLOUT_LORA_TARGET_MODULES,
)

DEFINITION_ID = "qwen3_5_9b_base_miles_lora_deterministic"
CATALOG_VISIBLE = False
TRAINER_MODELS_PER_INSTANCE = 6
ROLLOUT_MIN_CONTAINERS = 1
ROLLOUT_MAX_CONTAINERS = 1
ROLLOUT_MAX_RUNNING_REQUESTS = 64
ROLLOUT_DETERMINISTIC_INFERENCE = True
ROLLOUT_ATTENTION_BACKEND = "fa3"
app = modal.App(f"lilo-{DEFINITION_ID}")


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}!:{GPUS}",
    volumes=TRAINER_VOLUMES,
    secrets=[api_secret, proxy_secret],
    timeout=86_400,
    max_containers=trainer_max_containers(),
    single_use_containers=True,
)
def qwen3_5_9b_base_miles_lora_deterministic(instance_id: str) -> None:
    run_trainer(
        instance_id,
        definition_id=DEFINITION_ID,
        max_models=TRAINER_MODELS_PER_INSTANCE,
        deterministic_training=True,
    )


ENGINE_FUNCTION = qwen3_5_9b_base_miles_lora_deterministic
