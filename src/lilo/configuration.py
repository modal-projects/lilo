"""Validated Python components for model infrastructure.

Only backend config and environment dictionaries accept arbitrary keys.
Use dataclasses.replace to compose variants without mutating another config.
"""

from dataclasses import field
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, StrictBool
from pydantic.dataclasses import dataclass

PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonnegativeInt = Annotated[int, Field(strict=True, ge=0)]
NonemptyString = Annotated[str, Field(strict=True, min_length=1)]
CONFIG = ConfigDict(extra="forbid", validate_default=True)


@dataclass(config=CONFIG, frozen=True, kw_only=True)
class Compute:
    gpu: Annotated[str, Field(pattern=r"^[A-Za-z0-9-]+$")]
    gpus_per_node: PositiveInt = 1
    nodes: PositiveInt = 1
    cpu: Annotated[float, Field(gt=0)] = 8
    memory_mib: PositiveInt = 32768

    @property
    def modal_gpu(self) -> str:
        return f"{self.gpu}:{self.gpus_per_node}"


@dataclass(config=CONFIG, frozen=True, kw_only=True)
class Model:
    id: NonemptyString
    max_context_length: PositiveInt
    parameterization: Literal["lora", "full"] = "lora"
    revision: NonemptyString = "main"


@dataclass(config=CONFIG, frozen=True, kw_only=True)
class Trainer:
    compute: Compute
    backend: Literal["miles", "megatron"] = "miles"
    max_instances: PositiveInt = 1
    max_clients_per_instance: PositiveInt = 1
    sampler_persistence_concurrency: PositiveInt = 8
    timeout_s: PositiveInt = 86400
    config: dict[str, object] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)


@dataclass(config=CONFIG, frozen=True, kw_only=True)
class Inference:
    compute: Compute
    backend: Literal["sglang"] = "sglang"
    min_replicas: NonnegativeInt = 0
    max_replicas: PositiveInt = 8
    target_concurrency: PositiveInt = 16
    scaledown_window_s: PositiveInt = 300
    startup_timeout_s: PositiveInt = 1200
    config: dict[str, object] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)


@dataclass(config=CONFIG, frozen=True, kw_only=True)
class Routing:
    default: StrictBool = False
    sampling_default: StrictBool = False


@dataclass(config=CONFIG, frozen=True, kw_only=True)
class Lifecycle:
    session_idle_timeout_s: PositiveInt = 300
    pool_idle_timeout_s: PositiveInt = 300
    sweep_interval_s: PositiveInt = 300


@dataclass(config=CONFIG, frozen=True, kw_only=True)
class Deployment:
    name: Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]+$")]
    model: Model
    trainer: Trainer
    inference: Inference
    routing: Routing = field(default_factory=Routing)
    lifecycle: Lifecycle = field(default_factory=Lifecycle)

    def __post_init__(self):
        if self.inference.min_replicas > self.inference.max_replicas:
            raise ValueError("min_replicas must not exceed max_replicas")
        if self.inference.compute.nodes != 1:
            raise ValueError("each inference replica uses one node")
        if self.trainer.backend == "megatron":
            if self.trainer.compute.nodes != 1:
                raise ValueError("multi-node training currently requires Miles")
            if self.model.parameterization != "full":
                raise ValueError("Megatron requires full parameterization")
            if self.trainer.max_clients_per_instance != 1:
                raise ValueError("FFT trainers admit one client per instance")
            if self.trainer.sampler_persistence_concurrency != 1:
                raise ValueError("Megatron requires sampler_persistence_concurrency: 1")
        elif self.model.parameterization != "lora":
            raise ValueError("Miles requires lora parameterization")
        for env in (self.trainer.env, self.inference.env):
            if any(key.startswith("LILO_") for key in env):
                raise ValueError("LILO_ environment variables are managed by Lilo")
