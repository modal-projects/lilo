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
    command = [
        modal_cli,
        "deploy",
        "-m",
        "lilo.providers.modal.lora_pool_app",
        "--name",
        spec.app_name,
    ]
    environment = os.environ.get("MODAL_ENVIRONMENT")
    if environment:
        command.extend(["--env", environment])
    subprocess.run(command, env={**os.environ, **spec.env()}, check=True)
    return pool.gateway_url()


def stop_pool(spec: LoraPoolSpec) -> None:
    modal_cli = shutil.which("modal")
    if modal_cli is None:
        raise RuntimeError("modal CLI is unavailable")
    command = [modal_cli, "app", "stop", "-y", spec.app_name]
    environment = os.environ.get("MODAL_ENVIRONMENT")
    if environment:
        command.extend(["--env", environment])
    subprocess.run(command, check=True)


def _implementation_revision(definition_id: str) -> str:
    here = Path(__file__)
    files = (
        here,
        here.with_name("lora_pool_app.py"),
        here.with_name("rollout_image.py"),
        here.with_name("image_dependencies.py"),
        here.with_name("definitions") / f"{definition_id}.py",
        here.parents[2] / "inference" / "lora_sidecar.py",
        here.parents[2] / "inference" / "serving.py",
    )
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
