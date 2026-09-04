from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import modal

from lilo.errors import RecordNotFound

CHECKPOINT_VOLUME_NAME = "lilo-checkpoints"
CHECKPOINT_ROOT = "/checkpoints"
CHECKPOINT_IO_LOCK = asyncio.Lock()

checkpoint_volume = modal.Volume.from_name(
    CHECKPOINT_VOLUME_NAME,
    create_if_missing=True,
    version=2,
)


def checkpoint_path(uri: str, *, root: str = CHECKPOINT_ROOT) -> Path:
    path = Path(uri).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("checkpoint path is outside configured storage")
    return path


async def read_checkpoint_metadata(
    uri: str,
    *,
    root: str = CHECKPOINT_ROOT,
    volume: Any = checkpoint_volume,
) -> dict[str, object]:
    path = checkpoint_path(uri, root=root)
    async with CHECKPOINT_IO_LOCK:
        await asyncio.to_thread(volume.reload)
        metadata = json.loads(
            await asyncio.to_thread(
                (path / "metadata.json").read_text,
                encoding="utf-8",
            )
        )
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata must be an object")
    return metadata


async def write_checkpoint_expiration(
    uri: str,
    expires_at: float | None,
    save_seq_id: int,
    *,
    root: str = CHECKPOINT_ROOT,
    volume: Any = checkpoint_volume,
) -> None:
    path = checkpoint_path(uri, root=root)
    metadata_file = path / "metadata.json"

    def update() -> None:
        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise RecordNotFound("checkpoint", uri) from None
        if not isinstance(metadata, dict):
            raise ValueError("checkpoint metadata must be an object")
        stored_seq_id = metadata.get("checkpoint_save_seq_id", -1)
        if (
            isinstance(stored_seq_id, int)
            and not isinstance(stored_seq_id, bool)
            and stored_seq_id > save_seq_id
        ):
            return
        metadata["checkpoint_save_seq_id"] = save_seq_id
        metadata["expires_at"] = expires_at
        temporary = metadata_file.with_name(f".metadata-{uuid.uuid4().hex}.json")
        try:
            temporary.write_text(
                json.dumps(metadata, sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(metadata_file)
        finally:
            temporary.unlink(missing_ok=True)

    async with CHECKPOINT_IO_LOCK:
        await asyncio.to_thread(volume.reload)
        await asyncio.to_thread(update)
        await asyncio.to_thread(volume.commit)


def _checkpoint_entry(checkpoint: Path) -> dict[str, object]:
    files = [file for file in checkpoint.rglob("*") if file.is_file()]
    metadata_file = checkpoint / "metadata.json"
    return {
        "model_id": checkpoint.parent.parent.name,
        "name": checkpoint.name,
        "path": str(checkpoint),
        "time": checkpoint.stat().st_mtime,
        "size_bytes": sum(file.stat().st_size for file in files),
        "metadata": (
            json.loads(metadata_file.read_text(encoding="utf-8"))
            if metadata_file.is_file()
            else None
        ),
    }


def _scan_checkpoints(
    model_id: str | None,
    *,
    root: str = CHECKPOINT_ROOT,
) -> list[dict[str, object]]:
    checkpoint_root = Path(root)
    if not checkpoint_root.is_dir():
        return []
    model_dirs = (
        [checkpoint_root / model_id]
        if model_id is not None
        else list(checkpoint_root.iterdir())
    )
    return [
        _checkpoint_entry(checkpoint)
        for model_dir in model_dirs
        if (model_dir / "weights").is_dir()
        for checkpoint in (model_dir / "weights").iterdir()
        if checkpoint.is_dir()
    ]


async def list_checkpoints(
    model_id: str | None,
    *,
    root: str = CHECKPOINT_ROOT,
    volume: Any = checkpoint_volume,
) -> list[dict[str, object]]:
    async with CHECKPOINT_IO_LOCK:
        await asyncio.to_thread(volume.reload)
        return await asyncio.to_thread(_scan_checkpoints, model_id, root=root)


async def delete_checkpoint(
    uri: str,
    *,
    root: str = CHECKPOINT_ROOT,
    volume: Any = checkpoint_volume,
) -> None:
    path = checkpoint_path(uri, root=root)
    async with CHECKPOINT_IO_LOCK:
        await asyncio.to_thread(volume.reload)
        try:
            await asyncio.to_thread(shutil.rmtree, path)
        except FileNotFoundError:
            raise RecordNotFound("checkpoint", uri) from None
        await asyncio.to_thread(volume.commit)
