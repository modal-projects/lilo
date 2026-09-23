"""Deployment specifications. Loading YAML never contacts Modal or allocates GPUs."""

from __future__ import annotations

import hashlib
import json
import re
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Model(StrictModel):
    id: str = Field(min_length=1)
    revision: str = "main"
    parameterization: Literal["lora", "full"] = "lora"
    max_context_length: int = Field(gt=0)


class Routing(StrictModel):
    default: bool = False
    sampling_default: bool = False


class Resources(StrictModel):
    gpu: str
    cpu: float = Field(default=8, gt=0)
    memory_mib: int = Field(default=32768, gt=0)
    timeout_s: int = Field(default=86400, gt=0, le=86400)

    @property
    def gpu_count(self) -> int:
        if not re.fullmatch(r"[A-Za-z0-9-]+(?::[1-9][0-9]*)?", self.gpu):
            raise ValueError(f"invalid GPU resource: {self.gpu}")
        return int(self.gpu.split(":")[1]) if ":" in self.gpu else 1


class TrainerScaling(StrictModel):
    min_instances: Literal[0] = 0
    max_instances: int = Field(default=1, gt=0)


class InferenceScaling(StrictModel):
    min_replicas: int = Field(default=0, ge=0)
    max_replicas: int = Field(default=8, gt=0)
    target_concurrency: int = Field(default=16, gt=0)
    scaledown_window_s: int = Field(default=300, gt=0)

    @model_validator(mode="after")
    def ordered(self):
        if self.min_replicas > self.max_replicas:
            raise ValueError("min_replicas must not exceed max_replicas")
        return self


class EngineOptions(StrictModel):
    max_clients_per_instance: int = Field(default=1, gt=0)
    sampler_persistence_concurrency: int = Field(default=8, gt=0)


class Trainer(StrictModel):
    backend: str = "miles"
    resources: Resources
    scaling: TrainerScaling = Field(default_factory=TrainerScaling)
    engine: EngineOptions = Field(default_factory=EngineOptions)
    config: dict[str, Any] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)


class Inference(StrictModel):
    backend: str = "sglang"
    resources: Resources
    scaling: InferenceScaling = Field(default_factory=InferenceScaling)
    config: dict[str, Any] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)


class Secrets(StrictModel):
    api: str = "lilo-api"
    sampler_proxy: str = "lilo-proxy"
    huggingface: str | None = "huggingface-secret"


class Storage(StrictModel):
    assets: str = "lilo-model-assets"
    checkpoints: str = "lilo-checkpoints"
    bulletin: str = "lilo-snapshot-bulletin"


class ModalSettings(StrictModel):
    environment: str | None = None
    region: str = "us-west"


class Deployment(StrictModel):
    frontend: str = Field(
        default="lilo-yaml", pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,46}$"
    )
    mode: Literal["shared"] = "shared"
    modal: ModalSettings = Field(default_factory=ModalSettings)
    secrets: Secrets = Field(default_factory=Secrets)
    storage: Storage = Field(default_factory=Storage)


class Lifecycle(StrictModel):
    session_idle_timeout_s: int = Field(default=300, gt=0)
    pool_idle_timeout_s: int = Field(default=300, gt=0)
    sweep_interval_s: int = Field(default=300, gt=0)


class DeploymentSpec(StrictModel):
    api_version: Literal["lilo/v1"] = "lilo/v1"
    name: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
    model: Model
    routing: Routing = Field(default_factory=Routing)
    deployment: Deployment = Field(default_factory=Deployment)
    trainer: Trainer
    inference: Inference
    lifecycle: Lifecycle = Field(default_factory=Lifecycle)


class ResolvedDeployment(StrictModel):
    spec: DeploymentSpec
    implementation: str
    miles_commit: str | None = None
    generation: str
    active: bool = True

    @property
    def definition_id(self) -> str:
        return f"yaml_{self.spec.name}_{self.generation[:16]}"

    @property
    def asset_path(self) -> str:
        digest = hashlib.sha256(
            f"{self.spec.model.id}@{self.spec.model.revision}".encode()
        ).hexdigest()
        return f"/assets/{digest}"


def resolve(
    spec: DeploymentSpec,
    *,
    revision: str,
    implementation: str,
    miles_commit: str | None = None,
) -> ResolvedDeployment:
    if not re.fullmatch(r"[a-fA-F0-9]{40,64}", revision):
        raise ValueError("model revision must resolve to an exact commit")
    value = spec.model_dump()
    value["model"]["revision"] = revision
    pinned = DeploymentSpec.model_validate(value)
    # Include resource and lifecycle policy: each applied version remains self-contained.
    # Checkpoint compatibility uses model/topology metadata, not this generation hash.
    identity = pinned.model_dump(exclude={"routing"})
    digest = hashlib.sha256(
        json.dumps([implementation, identity], sort_keys=True).encode()
    ).hexdigest()
    return ResolvedDeployment(
        spec=pinned,
        implementation=implementation,
        generation=digest,
        miles_commit=miles_commit,
    )


class UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str) or key in result:
            raise ValueError(f"duplicate or non-string YAML key: {key!r}")
        result[key] = loader.construct_object(value_node)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def merge(parent: dict, child: dict) -> dict:
    return {
        key: merge(parent[key], value)
        if isinstance(parent.get(key), dict) and isinstance(value, dict)
        else value
        for key, value in (parent | child).items()
    }


def preset_path(name: str) -> Path:
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
        raise ValueError("invalid preset name")
    return Path(str(files("lilo").joinpath("presets", name + ".yaml")))


def load(path: str | Path) -> DeploymentSpec:
    """Read and merge YAML inheritance, then construct the deployment fields.

    Backend configuration is interpreted when preparing its trainer or pool.
    Parent files may be partial; only the fully merged document is constructed.
    """
    path = Path(path).resolve()
    seen = set()
    documents = []
    while True:
        if path in seen:
            raise ValueError(f"cyclic extends: {path}")
        seen.add(path)
        data = yaml.load(path.read_text(), Loader=UniqueLoader)
        if not isinstance(data, dict):
            raise ValueError("deployment YAML must be a mapping")
        parent = data.pop("extends", None)
        documents.append(data)
        if parent is None:
            break
        if not isinstance(parent, str):
            raise ValueError("extends must be a path or builtin:preset")
        path = (
            preset_path(parent[8:])
            if parent.startswith("builtin:")
            else path.parent / parent
        ).resolve()
    merged = {}
    for document in reversed(documents):
        merged = merge(merged, document)
    return DeploymentSpec(**merged)


def validate_frontend(specs: list[DeploymentSpec]) -> None:
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
