from __future__ import annotations

import importlib
import os

import modal

from lilo.inference.serving import (
    start_lora_sidecar,
    start_sglang,
    supervise_children,
    terminate,
    wait_http,
)

from .rollout_image import image

APP_NAME = os.environ["LILO_LORA_POOL_APP_NAME"]
DEFINITION_ID = os.environ["LILO_LORA_POOL_DEFINITION_ID"]
definition = importlib.import_module(
    f"lilo.providers.modal.definitions.{DEFINITION_ID}"
)
SGLANG_PORT = 8001
SIDECAR_PORT = 8000

api_secret = modal.Secret.from_name(
    "lilo-api",
    required_keys=["TINKER_API_KEY"],
)
pool_secret = modal.Secret.from_dict(
    {
        key: value
        for key, value in os.environ.items()
        if key.startswith("LILO_LORA_POOL_")
    }
)
app = modal.App(APP_NAME)


@app.server(
    image=image,
    gpu=f"{definition.ROLLOUT_GPU_TYPE}:{definition.ROLLOUT_GPUS}",
    volumes={
        "/assets": definition.assets,
        definition.BULLETIN_ROOT: definition.bulletin,
    },
    secrets=[api_secret, pool_secret],
    target_concurrency=definition.ROLLOUT_TARGET_CONCURRENCY,
    min_containers=0,
    max_containers=getattr(definition, "ROLLOUT_MAX_CONTAINERS", None),
    scaledown_window=getattr(definition, "ROLLOUT_SCALEDOWN_WINDOW", 5 * 60),
    startup_timeout=20 * 60,
    exit_grace_period=getattr(
        definition,
        "ROLLOUT_EXIT_GRACE_PERIOD",
        5 * 60,
    ),
    port=SIDECAR_PORT,
    routing_region="us-west",
)
class Server:
    @modal.enter()
    def start(self) -> None:
        self.sglang = start_sglang(
            definition.HF_CHECKPOINT,
            port=SGLANG_PORT,
            context_length=definition.MAX_CONTEXT_LENGTH,
            max_loras_per_batch=getattr(
                definition,
                "ROLLOUT_MAX_LORAS_PER_BATCH",
                8,
            ),
            max_loaded_loras=definition.ROLLOUT_MAX_LOADED_LORAS,
            max_lora_rank=definition.MAX_LORA_RANK,
            max_running_requests=definition.ROLLOUT_MAX_RUNNING_REQUESTS,
            max_queued_requests=definition.ROLLOUT_MAX_QUEUED_REQUESTS,
            tensor_parallel_size=definition.ROLLOUT_TENSOR_PARALLEL_SIZE,
            expert_parallel_size=definition.ROLLOUT_EXPERT_PARALLEL_SIZE,
            expert_tensor_parallel_size=(
                definition.ROLLOUT_EXPERT_TENSOR_PARALLEL_SIZE
            ),
            parallel_world_size=definition.ROLLOUT_GPUS,
            lora_target_modules=definition.ROLLOUT_LORA_TARGET_MODULES,
            enable_lora=True,
            memory_fraction=definition.ROLLOUT_MEMORY_FRACTION,
            schedule_policy="lpm",
        )
        wait_http(
            f"http://127.0.0.1:{SGLANG_PORT}/health",
            self.sglang,
            20 * 60,
        )
        self.sidecar = start_lora_sidecar(
            port=SIDECAR_PORT,
            sglang_port=SGLANG_PORT,
            bulletin_root=definition.BULLETIN_ROOT,
            bulletin_volume=definition.BULLETIN_VOLUME_NAME,
        )
        self.supervisor = supervise_children(self.sglang, self.sidecar)
        wait_http(
            f"http://127.0.0.1:{SIDECAR_PORT}/health",
            self.sidecar,
            20 * 60,
        )

    @modal.exit()
    def stop(self) -> None:
        terminate(getattr(self, "sidecar", None))
        terminate(getattr(self, "sglang", None))
