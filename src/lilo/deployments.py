"""Python deployment configs and the records saved by the deployment CLI."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import re
import runpy
import sys
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


@dataclass(kw_only=True)
class Model:
    id: str
    max_context_length: int
    revision: str = "main"
    parameterization: Literal["lora", "full"] = "lora"


@dataclass(kw_only=True)
class Routing:
    default: bool = False
    sampling_default: bool = False


@dataclass(kw_only=True)
class Resources:
    gpu: str
    cpu: float = 8
    memory_mib: int = 32768
    timeout_s: int = 86400

    @property
    def gpu_count(self) -> int:
        if not re.fullmatch(r"[A-Za-z0-9-]+(?::[1-9][0-9]*)?", self.gpu):
            raise ValueError(f"invalid GPU resource: {self.gpu}")
        return int(self.gpu.split(":")[1]) if ":" in self.gpu else 1


@dataclass(kw_only=True)
class TrainerScaling:
    min_instances: Literal[0] = 0
    max_instances: int = 1


@dataclass(kw_only=True)
class InferenceScaling:
    min_replicas: int = 0
    max_replicas: int = 8
    target_concurrency: int = 16
    scaledown_window_s: int = 300

    def __post_init__(self):
        if self.min_replicas > self.max_replicas:
            raise ValueError("min_replicas must not exceed max_replicas")


@dataclass(kw_only=True)
class EngineOptions:
    max_clients_per_instance: int = 1
    sampler_persistence_concurrency: int = 8


@dataclass(kw_only=True)
class Trainer:
    resources: Resources
    backend: str = "miles"
    scaling: TrainerScaling = field(default_factory=TrainerScaling)
    engine: EngineOptions = field(default_factory=EngineOptions)
    config: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)


@dataclass(kw_only=True)
class Inference:
    resources: Resources
    backend: str = "sglang"
    scaling: InferenceScaling = field(default_factory=InferenceScaling)
    config: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)


@dataclass(kw_only=True)
class Secrets:
    api: str = "lilo-api"
    sampler_proxy: str = "lilo-proxy"
    huggingface: str | None = "huggingface-secret"


@dataclass(kw_only=True)
class Storage:
    assets: str = "lilo-model-assets"
    checkpoints: str = "lilo-checkpoints"
    bulletin: str = "lilo-snapshot-bulletin"


@dataclass(kw_only=True)
class ModalSettings:
    environment: str | None = None
    region: str = "us-west"


@dataclass(kw_only=True)
class Deployment:
    frontend: str = "lilo-yaml"
    mode: Literal["shared"] = "shared"
    modal: ModalSettings = field(default_factory=ModalSettings)
    secrets: Secrets = field(default_factory=Secrets)
    storage: Storage = field(default_factory=Storage)


@dataclass(kw_only=True)
class Lifecycle:
    session_idle_timeout_s: int = 300
    pool_idle_timeout_s: int = 300
    sweep_interval_s: int = 300


@dataclass(kw_only=True)
class BaseConfig:
    """Subclass in a config file and override defaults or use __post_init__."""

    name: str
    model: Model
    trainer: Trainer
    inference: Inference
    api_version: Literal["lilo/v1"] = "lilo/v1"
    routing: Routing = field(default_factory=Routing)
    deployment: Deployment = field(default_factory=Deployment)
    lifecycle: Lifecycle = field(default_factory=Lifecycle)


class DeploymentRecord(BaseModel):
    """Saved deployment metadata around a Python configuration.

    The CLI resolves the model revision before creating this record. Its hash
    binds jobs to their original code and configuration across later deploys;
    active controls whether new clients can select it.
    """

    model_config = ConfigDict(extra="forbid")

    spec: BaseConfig
    implementation: str
    miles_commit: str | None = None
    generation: str
    active: bool = True

    @classmethod
    def create(
        cls,
        spec: BaseConfig,
        *,
        revision: str,
        implementation: str,
        miles_commit: str | None = None,
    ) -> DeploymentRecord:
        """Record an already-resolved revision without reparsing the configuration."""
        pinned = deepcopy(spec)
        pinned.model.revision = revision
        # Changing routing defaults should not restart an existing trainer.
        identity = asdict(pinned)
        identity.pop("routing")
        generation = hashlib.sha256(
            json.dumps([implementation, identity], sort_keys=True).encode()
        ).hexdigest()
        return cls(
            spec=pinned,
            implementation=implementation,
            generation=generation,
            miles_commit=miles_commit,
        )

    @property
    def definition_id(self) -> str:
        return f"yaml_{self.spec.name}_{self.generation[:16]}"

    @property
    def asset_path(self) -> str:
        digest = hashlib.sha256(
            f"{self.spec.model.id}@{self.spec.model.revision}".encode()
        ).hexdigest()
        return f"/assets/{digest}"


def config_path(name: str) -> Path:
    """Locate an installed example config without maintaining a model catalog."""
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
        raise ValueError("invalid config name")
    return Path(str(files("lilo").joinpath("configs", name.replace("-", "_") + ".py")))


def load(path: str | Path) -> BaseConfig:
    """Execute a Python config file and instantiate its exported Config class."""
    path = Path(path).resolve()
    if path.suffix != ".py":
        raise ValueError("deployment configs must be Python .py files")
    # Let a config import sibling modules using normal Python imports.
    original_path = sys.path[:]
    sys.path.insert(0, str(path.parent))
    try:
        namespace = runpy.run_path(str(path))
        config_class = namespace.get("Config")
        if not isinstance(config_class, type) or not issubclass(
            config_class, BaseConfig
        ):
            raise ValueError(f"{path} must export a Config subclass of BaseConfig")
        return config_class()
    finally:
        sys.path[:] = original_path


def validate_frontend(specs: list[BaseConfig]) -> None:
    if not specs:
        raise ValueError("at least one deployment is required")
    if len({s.name for s in specs}) != len(specs):
        raise ValueError("duplicate deployment name")
    first = specs[0]
    for spec in specs:
        if spec.deployment != first.deployment or spec.lifecycle != first.lifecycle:
            raise ValueError(
                "deployments on one frontend must share deployment and lifecycle settings"
            )
    defaults, sampling = set(), set()
    for spec in specs:
        key = (spec.model.id, spec.model.parameterization)
        if spec.routing.default:
            if key in defaults:
                raise ValueError(f"multiple defaults for {key}")
            defaults.add(key)
        if spec.routing.sampling_default:
            if spec.model.id in sampling:
                raise ValueError(f"multiple sampling defaults for {spec.model.id}")
            sampling.add(spec.model.id)
