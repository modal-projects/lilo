"""Readable Python recipes with validated model, trainer, and inference settings."""

from dataclasses import field
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, StrictBool
from pydantic.dataclasses import dataclass

PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonnegativeInt = Annotated[int, Field(strict=True, ge=0)]
NonemptyString = Annotated[str, Field(strict=True, min_length=1)]
GPU = Annotated[str, Field(strict=True, pattern=r"^[A-Za-z0-9-]+$")]
CONFIG = ConfigDict(extra="forbid", validate_default=True)


@dataclass(config=CONFIG, frozen=True, kw_only=True)
class Trainer:
    gpu: GPU
    gpus_per_node: PositiveInt = 1
    nodes: PositiveInt = 1
    cpu: Annotated[float, Field(gt=0)] = 8
    memory_mib: PositiveInt = 32768
    backend: Literal["miles", "megatron"] = "miles"
    max_instances: PositiveInt = 1
    max_clients_per_instance: PositiveInt = 1
    sampler_persistence_concurrency: PositiveInt = 8
    timeout_s: PositiveInt = 86400
    config: dict[str, object] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)


@dataclass(config=CONFIG, frozen=True, kw_only=True)
class Inference:
    gpu: GPU
    gpus_per_node: PositiveInt = 1
    cpu: Annotated[float, Field(gt=0)] = 8
    memory_mib: PositiveInt = 32768
    min_replicas: NonnegativeInt = 0
    max_replicas: PositiveInt = 8
    target_concurrency: PositiveInt = 16
    scaledown_window_s: PositiveInt = 300
    startup_timeout_s: PositiveInt = 1200
    config: dict[str, object] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)


@dataclass(config=CONFIG, frozen=True, kw_only=True)
class Deployment:
    name: Annotated[str, Field(strict=True, pattern=r"^[A-Za-z0-9_-]+$")]
    model: NonemptyString
    max_context_length: PositiveInt
    trainer: Trainer
    inference: Inference
    parameterization: Literal["lora", "full"] = "lora"
    revision: NonemptyString = "main"
    default: StrictBool = False
    sampling_default: StrictBool = False
    session_idle_timeout_s: PositiveInt = 300
    pool_idle_timeout_s: PositiveInt = 300
    sweep_interval_s: PositiveInt = 300

    def __post_init__(self):
        if self.inference.min_replicas > self.inference.max_replicas:
            raise ValueError("min_replicas must not exceed max_replicas")
        if self.trainer.backend == "megatron":
            if self.trainer.nodes != 1:
                raise ValueError("multi-node training currently requires Miles")
            if self.parameterization != "full":
                raise ValueError("Megatron requires full parameterization")
            if self.trainer.max_clients_per_instance != 1:
                raise ValueError("FFT trainers admit one client per instance")
            if self.trainer.sampler_persistence_concurrency != 1:
                raise ValueError("Megatron requires sampler_persistence_concurrency: 1")
        elif self.parameterization != "lora":
            raise ValueError("Miles requires lora parameterization")
        for env in (self.trainer.env, self.inference.env):
            if any(key.startswith("LILO_") for key in env):
                raise ValueError("LILO_ environment variables are managed by Lilo")


class BaseConfig(Deployment):
    """Author a recipe with class attributes and ordinary Python inheritance.

    Sections are dictionaries in the recipe and validated values in an instance.
    Overriding a section replaces it; use ``{**Parent.trainer, ...}`` to extend it.
    """

    def __init__(self, **overrides):
        from copy import deepcopy

        values = {}
        for cls in reversed(type(self).__mro__):
            if cls in (object, Deployment, BaseConfig):
                continue
            values.update(
                (key, value)
                for key, value in vars(cls).items()
                if not key.startswith("_")
            )
        super().__init__(**deepcopy(values | overrides))
