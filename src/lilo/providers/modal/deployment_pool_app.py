"""Generic rollout app constructed in the pool deployment subprocess."""

import os

from lilo.deployments import DeploymentRecord

from .deployment_apps import build_rollout_app
from .deployment_records import POOL_CONFIG_ENV
from .fft_pool import FFTPoolSpec
from .lora_pool import LoraPoolSpec

resolved = DeploymentRecord.model_validate_json(os.environ[POOL_CONFIG_ENV])
if resolved.spec.parameterization == "lora":
    pool = LoraPoolSpec(resolved.definition_id, revision=resolved.generation[:16])
else:
    pool = FFTPoolSpec(
        definition_id=resolved.definition_id,
        model_id=os.environ["LILO_FFT_POOL_MODEL_ID"],
        latest=os.environ["LILO_FFT_POOL_LATEST"] == "1",
        version=int(os.environ["LILO_FFT_POOL_VERSION"]),
        **{
            key: int(os.environ[f"LILO_FFT_POOL_{key.upper()}"])
            for key in ("min_containers", "max_containers", "scaledown_window")
            if f"LILO_FFT_POOL_{key.upper()}" in os.environ
        },
    )
app, Server = build_rollout_app(resolved, pool)
