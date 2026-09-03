from __future__ import annotations

import importlib
import os

import modal

from lilo.inference.serving import (
    start_fft_sidecar,
    start_sglang,
    supervise_children,
    terminate,
    wait_http,
)

from .rollout_image import image

APP_NAME = os.environ["LILO_FFT_POOL_APP_NAME"]
DEFINITION_ID = os.environ["LILO_FFT_POOL_DEFINITION_ID"]
MODEL_ID = os.environ["LILO_FFT_POOL_MODEL_ID"]
LATEST = os.environ["LILO_FFT_POOL_LATEST"] == "1"
VERSION = int(os.environ["LILO_FFT_POOL_VERSION"])
definition = importlib.import_module(
    f"lilo.providers.modal.definitions.{DEFINITION_ID}"
)
ROLLOUT_GPU_TYPE = definition.ROLLOUT_GPU_TYPE
ROLLOUT_GPUS = definition.ROLLOUT_GPUS
ROLLOUT_TENSOR_PARALLEL_SIZE = definition.ROLLOUT_TENSOR_PARALLEL_SIZE
ROLLOUT_EXPERT_PARALLEL_SIZE = definition.ROLLOUT_EXPERT_PARALLEL_SIZE
ROLLOUT_EXPERT_TENSOR_PARALLEL_SIZE = definition.ROLLOUT_EXPERT_TENSOR_PARALLEL_SIZE
ROLLOUT_MEMORY_FRACTION = definition.ROLLOUT_MEMORY_FRACTION
ROLLOUT_MAX_RUNNING_REQUESTS = definition.ROLLOUT_MAX_RUNNING_REQUESTS
ROLLOUT_MAX_QUEUED_REQUESTS = definition.ROLLOUT_MAX_QUEUED_REQUESTS
ROLLOUT_TARGET_CONCURRENCY = definition.ROLLOUT_TARGET_CONCURRENCY


def _pool_setting(name: str, default: int | None) -> int | None:
    value = os.environ.get(f"LILO_FFT_POOL_{name}")
    return default if value is None else int(value)


ROLLOUT_MIN_CONTAINERS = _pool_setting(
    "MIN_CONTAINERS",
    getattr(definition, "ROLLOUT_MIN_CONTAINERS", None),
)
ROLLOUT_MAX_CONTAINERS = _pool_setting(
    "MAX_CONTAINERS",
    getattr(definition, "ROLLOUT_MAX_CONTAINERS", None),
)
ROLLOUT_SCALEDOWN_WINDOW = _pool_setting(
    "SCALEDOWN_WINDOW",
    getattr(definition, "ROLLOUT_SCALEDOWN_WINDOW", 5 * 60),
)
ROLLOUT_EXIT_GRACE_PERIOD = getattr(
    definition,
    "ROLLOUT_EXIT_GRACE_PERIOD",
    5 * 60,
)
ROLLOUT_CPU_WEIGHT_CACHE_MAX_COMPILE_GROUP_GB = (
    definition.ROLLOUT_CPU_WEIGHT_CACHE_MAX_COMPILE_GROUP_GB
)
SGLANG_PORT = 8001
SIDECAR_PORT = 8000

pool_secret = modal.Secret.from_dict(
    {
        key: value
        for key, value in os.environ.items()
        if key.startswith("LILO_FFT_POOL_")
    }
)
api_secret = modal.Secret.from_name(
    "lilo-api",
    required_keys=["TINKER_API_KEY"],
)
app = modal.App(APP_NAME)


@app.server(
    image=image,
    gpu=f"{ROLLOUT_GPU_TYPE}:{ROLLOUT_GPUS}",
    volumes={
        "/assets": definition.assets,
        definition.BULLETIN_ROOT: definition.bulletin,
    },
    secrets=[api_secret, pool_secret],
    target_concurrency=ROLLOUT_TARGET_CONCURRENCY,
    min_containers=ROLLOUT_MIN_CONTAINERS,
    max_containers=ROLLOUT_MAX_CONTAINERS,
    scaledown_window=ROLLOUT_SCALEDOWN_WINDOW,
    startup_timeout=20 * 60,
    exit_grace_period=ROLLOUT_EXIT_GRACE_PERIOD,
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
            max_loras_per_batch=1,
            max_loaded_loras=1,
            max_lora_rank=1,
            max_running_requests=ROLLOUT_MAX_RUNNING_REQUESTS,
            max_queued_requests=ROLLOUT_MAX_QUEUED_REQUESTS,
            tensor_parallel_size=ROLLOUT_TENSOR_PARALLEL_SIZE,
            expert_parallel_size=ROLLOUT_EXPERT_PARALLEL_SIZE,
            expert_tensor_parallel_size=ROLLOUT_EXPERT_TENSOR_PARALLEL_SIZE,
            parallel_world_size=ROLLOUT_GPUS,
            enable_lora=False,
            enable_cpu_weight_cache=True,
            cpu_weight_cache_max_compile_group_gb=(
                ROLLOUT_CPU_WEIGHT_CACHE_MAX_COMPILE_GROUP_GB
            ),
            memory_fraction=ROLLOUT_MEMORY_FRACTION,
            schedule_policy="lpm",
        )
        wait_http(
            f"http://127.0.0.1:{SGLANG_PORT}/health",
            self.sglang,
            20 * 60,
        )
        self.sidecar = start_fft_sidecar(
            port=SIDECAR_PORT,
            sglang_port=SGLANG_PORT,
            model_path=definition.HF_CHECKPOINT,
            bulletin_root=definition.BULLETIN_ROOT,
            bulletin_volume=definition.BULLETIN_VOLUME_NAME,
            run_id=MODEL_ID,
            pinned_version=None if LATEST else VERSION,
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
