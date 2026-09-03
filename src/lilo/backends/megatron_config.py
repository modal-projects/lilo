from __future__ import annotations

from pathlib import Path
from typing import Any

from .megatron_runtime.common.config import EngineModelConfig, OptimizerConfig


def parse_backend_config(value: dict[str, Any]) -> tuple[EngineModelConfig, Path]:
    megatron = dict(value.get("megatron") or value)
    optimizer = OptimizerConfig(**dict(megatron.pop("optimizer", {})))
    if "target_modules" in megatron:
        megatron["target_modules"] = tuple(megatron["target_modules"])
    config = EngineModelConfig(**megatron, optimizer=optimizer)
    checkpoint_dir = Path(value.get("checkpoint_dir") or "/tmp/lilo")
    return config, checkpoint_dir
