"""Python deployment configs and the records saved by the deployment CLI."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, fields
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import re
import runpy
import sys
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict


# Shared orchestration defaults. Backend option dictionaries have no schema here.
_DEFAULTS = {
    "api_version": "lilo/v1",
    "model": {"revision": "main", "parameterization": "lora"},
    "routing": {"default": False, "sampling_default": False},
    "trainer": {
        "backend": "miles",
        "resources": {"cpu": 8, "memory_mib": 32768, "timeout_s": 86400},
        "scaling": {"min_instances": 0, "max_instances": 1},
        "engine": {"max_clients_per_instance": 1, "sampler_persistence_concurrency": 8},
        "config": {},
        "env": {},
    },
    "inference": {
        "backend": "sglang",
        "resources": {"cpu": 8, "memory_mib": 32768, "timeout_s": 86400},
        "scaling": {
            "min_replicas": 0,
            "max_replicas": 8,
            "target_concurrency": 16,
            "scaledown_window_s": 300,
        },
        "config": {},
        "env": {},
    },
    "deployment": {
        "frontend": "lilo-yaml",
        "mode": "shared",
        "modal": {"environment": None, "region": "us-west"},
        "secrets": {
            "api": "lilo-api",
            "sampler_proxy": "lilo-proxy",
            "huggingface": "huggingface-secret",
        },
        "storage": {
            "assets": "lilo-model-assets",
            "checkpoints": "lilo-checkpoints",
            "bulletin": "lilo-snapshot-bulletin",
        },
    },
    "lifecycle": {
        "session_idle_timeout_s": 300,
        "pool_idle_timeout_s": 300,
        "sweep_interval_s": 300,
    },
}


def _with_defaults(defaults, values):
    """Fill omitted orchestration settings; explicit values win."""
    if not isinstance(defaults, dict) or not isinstance(values, dict):
        return deepcopy(values)
    result = deepcopy(defaults)
    for key, value in values.items():
        result[key] = _with_defaults(defaults.get(key), value)
    return result


@dataclass(kw_only=True, init=False)
class BaseConfig:
    """Declare plain section dictionaries and optional inherited overrides."""

    name: str
    model: dict[str, Any]
    trainer: dict[str, Any]
    inference: dict[str, Any]
    api_version: str
    routing: dict[str, Any]
    deployment: dict[str, Any]
    lifecycle: dict[str, Any]
    overrides: ClassVar[dict[str, Any]] = {}

    def __init__(self, **kwargs):
        names = {item.name for item in fields(BaseConfig)}
        unknown = kwargs.keys() - names
        if unknown:
            raise TypeError(f"unknown config fields: {sorted(unknown)}")
        values = deepcopy(_DEFAULTS)
        for cls in reversed(type(self).__mro__):
            for name in names & vars(cls).keys():
                values[name] = _with_defaults(_DEFAULTS.get(name), vars(cls)[name])
            for path, value in vars(cls).get("overrides", {}).items():
                parts = path.split(".")
                if parts[0] not in names:
                    raise ValueError(f"unknown config override: {path}")
                target = values
                try:
                    for part in parts[:-1]:
                        target = target[part]
                    target[parts[-1]] = deepcopy(value)
                except (KeyError, TypeError) as exc:
                    raise ValueError(f"unknown config override: {path}") from exc
        for name, value in kwargs.items():
            values[name] = _with_defaults(_DEFAULTS.get(name), value)
        missing = names - values.keys()
        if missing:
            raise TypeError(f"missing config fields: {sorted(missing)}")
        self.__dict__.update(values)


def gpu_count(resources):
    gpu = resources["gpu"]
    if not re.fullmatch(r"[A-Za-z0-9-]+(?::[1-9][0-9]*)?", gpu):
        raise ValueError(f"invalid GPU resource: {gpu}")
    return int(gpu.split(":")[1]) if ":" in gpu else 1


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
        pinned.model["revision"] = revision
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
            f"{self.spec.model['id']}@{self.spec.model['revision']}".encode()
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
        key = (spec.model["id"], spec.model["parameterization"])
        if spec.routing["default"]:
            if key in defaults:
                raise ValueError(f"multiple defaults for {key}")
            defaults.add(key)
        if spec.routing["sampling_default"]:
            if spec.model["id"] in sampling:
                raise ValueError(f"multiple sampling defaults for {spec.model['id']}")
            sampling.add(spec.model["id"])
