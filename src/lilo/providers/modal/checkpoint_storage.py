from __future__ import annotations

import asyncio
import json
from pathlib import Path
import shutil

import modal

from lilo.errors import RecordNotFound

CHECKPOINT_VOLUME_NAME = "lilo-checkpoints"
CHECKPOINT_ROOT = "/checkpoints"

checkpoint_volume = modal.Volume.from_name(
    CHECKPOINT_VOLUME_NAME,
    create_if_missing=True,
    version=2,
)


def _checkpoint_path(root: str, uri: str) -> Path:
    path = Path(uri).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("checkpoint path is outside configured storage")
    return path


def _checkpoint_entry(checkpoint: Path) -> dict[str, object]:
    files = [file for file in checkpoint.rglob("*") if file.is_file()]
    metadata_file = checkpoint / "metadata.json"
    return {
        "model_id": checkpoint.name,
        "name": checkpoint.parent.name,
        "path": str(checkpoint),
        "time": checkpoint.stat().st_mtime,
        "size_bytes": sum(file.stat().st_size for file in files),
        "metadata": (
            json.loads(metadata_file.read_text(encoding="utf-8"))
            if metadata_file.is_file()
            else None
        ),
    }


def _scan_checkpoints(root: str, model_id: str | None) -> list[dict[str, object]]:
    root = Path(root)
    if not root.is_dir():
        return []
    entries = []
    for name_dir in root.iterdir():
        if not name_dir.is_dir():
            continue
        candidates = (
            [name_dir / model_id] if model_id is not None else name_dir.iterdir()
        )
        for checkpoint in candidates:
            if checkpoint.is_dir() and (checkpoint / "metadata.json").is_file():
                entries.append(_checkpoint_entry(checkpoint))
    return entries


class ModalCheckpointStorage:
    """Checkpoint callbacks for a mounted volume in either Modal deployment."""

    def __init__(self, volume, root: str = CHECKPOINT_ROOT, *, lock=None):
        self.volume = volume
        self.root = root
        self.lock = lock if lock is not None else asyncio.Lock()

    async def read_metadata(self, uri: str) -> dict[str, object]:
        path = _checkpoint_path(self.root, uri)
        async with self.lock:
            await asyncio.to_thread(self.volume.reload)
            metadata = json.loads(
                await asyncio.to_thread(
                    (path / "metadata.json").read_text,
                    encoding="utf-8",
                )
            )
        if not isinstance(metadata, dict):
            raise ValueError("checkpoint metadata must be an object")
        return metadata

    async def list(self, model_id: str | None) -> list[dict[str, object]]:
        async with self.lock:
            await asyncio.to_thread(self.volume.reload)
            return await asyncio.to_thread(_scan_checkpoints, self.root, model_id)

    async def delete(self, uri: str) -> None:
        path = _checkpoint_path(self.root, uri)
        async with self.lock:
            await asyncio.to_thread(self.volume.reload)
            try:
                await asyncio.to_thread(shutil.rmtree, path)
            except FileNotFoundError:
                raise RecordNotFound("checkpoint", uri) from None
            await asyncio.to_thread(self.volume.commit)
