"""Install the pinned deterministic FA3 build into the existing Miles image."""

import hashlib
import json
from pathlib import Path

FA3_REVISION = "8d3a3b80d4758ebde5a867c50d24d4351443cf2b"
WHEEL = "flash_attn_3-3.0.0-cp310-abi3-linux_x86_64.whl"


def validate_artifacts(artifacts: Path) -> None:
    manifest = json.loads((artifacts / "manifest.json").read_text())
    if manifest.get("revision") != FA3_REVISION:
        raise ValueError("Deterministic FA3 requires the pinned source revision")
    if manifest.get("name") != WHEEL:
        raise ValueError("Unexpected deterministic FA3 wheel name")
    if not manifest.get("binary_sha256"):
        raise ValueError("Deterministic FA3 manifest must identify its CUDA binaries")
    digest = hashlib.sha256((artifacts / WHEEL).read_bytes()).hexdigest()
    if digest != manifest.get("wheel_sha256"):
        raise ValueError("Deterministic FA3 wheel does not match its manifest")


def with_deterministic_fa3(base_image, artifacts: Path):
    artifacts = artifacts.expanduser().resolve()
    validate_artifacts(artifacts)
    return (
        base_image.add_local_file(artifacts / WHEEL, "/root/" + WHEEL, copy=True)
        .add_local_file(
            artifacts / "manifest.json", "/root/fa3-manifest.json", copy=True
        )
        .add_local_file(
            Path(__file__).with_name("fa3_determinism_patch.py"),
            "/root/patch_fa3_transformer_engine.py",
            copy=True,
        )
        .run_commands(
            "python -m pip install --no-deps --force-reinstall /root/" + WHEEL,
            "python /root/patch_fa3_transformer_engine.py /root/fa3-manifest.json",
        )
        .env({"LILO_PARITY_FA3_HD256": "1", "TORCHINDUCTOR_COMPILE_THREADS": "1"})
    )
