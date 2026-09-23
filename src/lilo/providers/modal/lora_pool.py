from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from stitch.pools.modal_flash import ModalFlashPool


@dataclass(frozen=True)
class LoraPoolSpec:
    definition_id: str
    revision: str = ""

    def __post_init__(self) -> None:
        if Path(self.definition_id).name != self.definition_id:
            raise ValueError(f"invalid definition id: {self.definition_id!r}")
        if not self.revision:
            object.__setattr__(
                self,
                "revision",
                _implementation_revision(self.definition_id),
            )

    @classmethod
    def from_dict(cls, value: dict) -> LoraPoolSpec:
        return cls(
            definition_id=str(value["definition_id"]),
            revision=str(value.get("revision", "")),
        )

    @property
    def app_name(self) -> str:
        digest = hashlib.sha256(
            f"{self.definition_id}\0{self.revision}".encode()
        ).hexdigest()[:16]
        return f"lilo-lora-{digest}"

    def as_dict(self) -> dict[str, str]:
        return asdict(self)

    def env(self) -> dict[str, str]:
        return {
            "LILO_LORA_POOL_APP_NAME": self.app_name,
            "LILO_LORA_POOL_DEFINITION_ID": self.definition_id,
        }


async def pool_gateway(spec: LoraPoolSpec) -> str:
    return await ModalFlashPool(spec.app_name, "Server").gateway_url_async()


def deploy_pool(spec: LoraPoolSpec) -> str:
    pool = ModalFlashPool(spec.app_name, "Server")
    try:
        return pool.gateway_url()
    except Exception as exc:
        import modal

        if not isinstance(exc, modal.exception.NotFoundError):
            raise
    modal_cli = shutil.which("modal")
    if modal_cli is None:
        raise RuntimeError("modal CLI is unavailable")
    from .deployment_apps import pool_environment

    recipe_env = pool_environment(spec.definition_id)
    command = [
        modal_cli,
        "deploy",
        "-m",
        "lilo.providers.modal.deployment_pool_app",
        "--name",
        spec.app_name,
    ]
    environment = os.environ.get("MODAL_ENVIRONMENT")
    if environment:
        command.extend(["--env", environment])
    subprocess.run(command, env={**os.environ, **spec.env(), **recipe_env}, check=True)
    return pool.gateway_url()


def stop_pool(spec: LoraPoolSpec) -> None:
    modal_cli = shutil.which("modal")
    if modal_cli is None:
        raise RuntimeError("modal CLI is unavailable")
    command = [modal_cli, "app", "stop", "-y", spec.app_name]
    environment = os.environ.get("MODAL_ENVIRONMENT")
    if environment:
        command.extend(["--env", environment])
    result = subprocess.run(command, capture_output=True, text=True)
    already_stopped = result.returncode == 1 and any(
        line.strip().startswith("App is already stopped.")
        for line in (result.stdout + "\n" + result.stderr).splitlines()
    )
    if not already_stopped:
        result.check_returncode()


def _implementation_revision(definition_id: str) -> str:
    if not definition_id.startswith("yaml_"):
        raise ValueError(f"expected a YAML deployment id: {definition_id}")
    return definition_id.rsplit("_", 1)[-1]
