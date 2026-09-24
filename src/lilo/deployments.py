"""Python deployment configs and the records saved by the deployment CLI."""

from __future__ import annotations

import hashlib
import json
import re
import runpy
import sys
from copy import deepcopy
from dataclasses import asdict, replace
from importlib.resources import files
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from lilo.backends.deployment import resolve_backend_settings
from lilo.configuration import Deployment

LIFECYCLE_FIELDS = ("session_idle_timeout_s", "pool_idle_timeout_s", "sweep_interval_s")

PLATFORM_DEFAULTS = {
    "frontend": "lilo-yaml",
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
}


def settings_hash(settings: dict) -> str:
    """Stable identifier for settings, not a claim that they have been validated."""
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def deployment_generation(
    spec,
    platform,
    miles_commit,
    trainer_release,
    inference_release,
    trainer_settings,
    inference_settings,
):
    identity = asdict(spec)
    return settings_hash(
        {
            "config": identity,
            "platform": platform,
            "miles_commit": miles_commit,
            "trainer_release": trainer_release,
            "inference_release": inference_release,
            "trainer_settings": trainer_settings,
            "inference_settings": inference_settings,
        }
    )


def platform_defaults():
    return deepcopy(PLATFORM_DEFAULTS)


class DeploymentRecord(BaseModel):
    """Saved deployment metadata around a Python configuration.

    The CLI resolves the model revision before creating this record. Its hash
    binds jobs to their original configuration across later deploys;
    active marks recipes in the latest deploy command; retained recipes remain usable.
    """

    model_config = ConfigDict(extra="forbid")

    spec: Deployment
    platform: dict[str, Any] = Field(default_factory=platform_defaults)
    trainer_release: str = "initial"
    inference_release: str = "initial"
    miles_commit: str | None = None
    trainer_settings: dict
    inference_settings: dict
    generation: str
    active: bool = True

    @classmethod
    def create(
        cls,
        spec: Deployment,
        *,
        revision: str,
        miles_commit: str | None = None,
        platform: dict | None = None,
        trainer_release: str = "initial",
        inference_release: str = "initial",
    ) -> DeploymentRecord:
        """Record an already-resolved revision without reparsing the configuration."""
        pinned = replace(deepcopy(spec), revision=revision)
        asset_path = model_asset_path(pinned.model, revision)
        trainer_settings, inference_settings = resolve_backend_settings(
            pinned, asset_path
        )
        platform = deepcopy(PLATFORM_DEFAULTS if platform is None else platform)
        generation = deployment_generation(
            pinned,
            platform,
            miles_commit,
            trainer_release,
            inference_release,
            trainer_settings,
            inference_settings,
        )
        return cls(
            spec=pinned,
            trainer_settings=trainer_settings,
            inference_settings=inference_settings,
            platform=platform,
            trainer_release=trainer_release,
            inference_release=inference_release,
            generation=generation,
            miles_commit=miles_commit,
        )

    def with_releases(self, trainer_release, inference_release):
        generation = deployment_generation(
            self.spec,
            self.platform,
            self.miles_commit,
            trainer_release,
            inference_release,
            self.trainer_settings,
            self.inference_settings,
        )
        return self.model_copy(
            update={
                "trainer_release": trainer_release,
                "inference_release": inference_release,
                "generation": generation,
            }
        )

    @property
    def trainer_hash(self) -> str:
        """Identify trainer settings and the deployment-managed code release."""
        return settings_hash(
            {
                "name": self.spec.name,
                "model": self.spec.model,
                "revision": self.spec.revision,
                "parameterization": self.spec.parameterization,
                "max_context_length": self.spec.max_context_length,
                "trainer": asdict(self.spec.trainer),
                "settings": self.trainer_settings,
                "release": self.trainer_release,
                "platform": self.platform,
                "miles_commit": self.miles_commit,
            }
        )

    @property
    def inference_hash(self) -> str:
        """Identify inference settings, including the adapter shape it must load."""
        return settings_hash(
            {
                "name": self.spec.name,
                "model": self.spec.model,
                "revision": self.spec.revision,
                "parameterization": self.spec.parameterization,
                "max_context_length": self.spec.max_context_length,
                "inference": asdict(self.spec.inference),
                "settings": self.inference_settings,
                "release": self.inference_release,
                "platform": self.platform,
            }
        )

    @property
    def trainer_app_name(self) -> str:
        return f"lilo-trainer-{self.trainer_hash[:24]}"

    @property
    def inference_app_name(self) -> str:
        return f"lilo-inference-{self.inference_hash[:24]}"

    @property
    def definition_id(self) -> str:
        return f"yaml_{self.spec.name}_{self.generation[:16]}"

    @property
    def asset_path(self) -> str:
        return model_asset_path(self.spec.model, self.spec.revision)


def model_asset_path(model_id, revision):
    digest = hashlib.sha256(f"{model_id}@{revision}".encode()).hexdigest()
    return f"/assets/{digest}"


def config_path(name: str) -> Path:
    """Locate an installed example config without maintaining a model catalog."""
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
        raise ValueError("invalid config name")
    return Path(str(files("lilo").joinpath("configs", name.replace("-", "_") + ".py")))


def load(path: str | Path) -> Deployment:
    """Execute a Python config file and read its exported config object."""
    path = Path(path).resolve()
    if path.suffix != ".py":
        raise ValueError("deployment configs must be Python .py files")
    # Let a config import sibling modules using normal Python imports.
    original_path = sys.path[:]
    sys.path.insert(0, str(path.parent))
    try:
        namespace = runpy.run_path(str(path))
        config = namespace.get("config")
        if not isinstance(config, Deployment):
            raise ValueError(f"{path} must export a BaseConfig instance named config")
        return config
    finally:
        sys.path[:] = original_path


def validate_frontend(specs: list[Deployment]) -> None:
    if not specs:
        raise ValueError("at least one deployment is required")
    if len({s.name for s in specs}) != len(specs):
        raise ValueError("duplicate deployment name")
    first = specs[0]
    for spec in specs:
        if any(
            getattr(spec, field) != getattr(first, field) for field in LIFECYCLE_FIELDS
        ):
            raise ValueError(
                "deployments on one frontend must share lifecycle settings"
            )
