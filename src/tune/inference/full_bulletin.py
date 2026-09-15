from __future__ import annotations

import asyncio
import contextlib
import hashlib
import inspect
import json
import os
import shutil
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from stitch.stores.base import Store
from stitch.types import VersionManifest, VersionRef, decide_pointer_move

_INDEX = "model.safetensors.index.json"
_MANIFEST = "snapshot.json"
_POINTER = "latest"


class FFTSnapshotNotFound(FileNotFoundError):
    pass


class FFTSnapshotBulletin:
    def __init__(
        self,
        root: str | Path,
        *,
        refresh: Callable[[], Any] | None = None,
        commit: Callable[[], None] | None = None,
    ) -> None:
        self.root = Path(root) / "full"
        self._refresh = refresh
        self._commit = commit
        self._lock = threading.RLock()

    async def refresh(self) -> None:
        if self._refresh is None:
            return
        for attempt in range(20):
            try:
                result = await asyncio.to_thread(self._refresh)
                if inspect.isawaitable(result):
                    await result
                return
            except RuntimeError as exc:
                if "open files" not in str(exc) or attempt == 19:
                    raise
                await asyncio.sleep(0.25)

    def read_latest(self, run_id: str) -> VersionRef | None:
        path = self._run_dir(run_id) / _POINTER
        if not path.exists():
            return None
        value = path.read_text(encoding="utf-8").strip()
        return VersionRef.parse(value) if value else None

    def snapshot_dir(self, ref: VersionRef) -> Path:
        self._validate_ref(ref)
        return self._run_dir(ref.run_id) / "updates" / Path(ref.identity).name

    def staging_dir(self, ref: VersionRef) -> Path:
        target = self.snapshot_dir(ref)
        return target.parent / ".staging" / target.name

    def claim(self, run_id: str) -> VersionRef:
        ref = VersionRef(run_id, 0)
        with self._lock:
            current = self.read_latest(run_id)
            if current == ref:
                return ref
            decide_pointer_move(current, ref)
            self._advance(ref)
            self._commit_volume()
        return ref

    def publish(
        self,
        ref: VersionRef,
        source_dir: str | Path,
        *,
        metadata: dict[str, Any],
    ) -> bool:
        self.prepare(ref, source_dir, metadata=metadata)
        self.store(ref, source_dir, metadata=metadata)
        return self.advance(ref)

    def prepare(
        self,
        ref: VersionRef,
        source_dir: str | Path,
        *,
        metadata: dict[str, Any],
    ) -> None:
        source = Path(source_dir)
        _stamp_index(source, ref, metadata)

    def store(
        self,
        ref: VersionRef,
        source_dir: str | Path,
        *,
        metadata: dict[str, Any],
    ) -> None:
        source = Path(source_dir)
        _stamp_index(source, ref, metadata)
        manifest = _manifest(ref, source, metadata)
        (source / _MANIFEST).write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        target = self.snapshot_dir(ref)
        with self._lock:
            if target.exists():
                if _read_manifest(target) != manifest:
                    raise ValueError(f"FFT snapshot is immutable: {ref.identity}")
                shutil.rmtree(source, ignore_errors=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                os.rename(source, target)
            self._commit_volume()

    def advance(self, ref: VersionRef) -> bool:
        with self._lock:
            current = self.read_latest(ref.run_id)
            if current == ref:
                return False
            decide_pointer_move(current, ref)
            self._advance(ref)
            self._commit_volume()
            return True

    def resolve(self, ref: VersionRef) -> Path:
        if ref.version == 0:
            raise FFTSnapshotNotFound(ref.identity)
        target = self.snapshot_dir(ref)
        try:
            manifest = _read_manifest(target)
            files = manifest["files"]
            index = json.loads((target / _INDEX).read_text(encoding="utf-8"))
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
            raise FFTSnapshotNotFound(ref.identity) from exc
        expected = {str(name) for name in (index.get("weight_map") or {}).values()} | {
            _INDEX
        }
        names = set(files)
        try:
            valid = (
                manifest.get("ref") == ref.identity
                and int((index.get("metadata") or {}).get("version")) == ref.version
                and expected.issubset(names)
                and all(_safe_relative_path(name) for name in names)
                and all(_file_manifest(target / name) == files[name] for name in names)
            )
        except (FileNotFoundError, TypeError, ValueError) as exc:
            raise FFTSnapshotNotFound(ref.identity) from exc
        if not valid:
            raise FFTSnapshotNotFound(ref.identity)
        return target

    def metadata(self, ref: VersionRef) -> dict[str, Any]:
        if ref.version == 0:
            return {"optimizer_step": 0}
        try:
            manifest = _read_manifest(self.snapshot_dir(ref))
            metadata = manifest["metadata"]
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
            raise FFTSnapshotNotFound(ref.identity) from exc
        if not isinstance(metadata, dict):
            raise FFTSnapshotNotFound(ref.identity)
        return metadata

    def _advance(self, ref: VersionRef) -> None:
        run_dir = self._run_dir(ref.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(run_dir / _POINTER, ref.identity)

    def _commit_volume(self) -> None:
        if self._commit is not None:
            self._commit()

    def _run_dir(self, run_id: str | None) -> Path:
        if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
            raise ValueError(f"invalid FFT snapshot run: {run_id!r}")
        return self.root / run_id

    @staticmethod
    def _validate_ref(ref: VersionRef) -> None:
        if ref.version <= 0:
            raise ValueError(f"invalid FFT snapshot ref: {ref.identity}")
        run_id = ref.run_id
        if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
            raise ValueError(f"invalid FFT snapshot ref: {ref.identity}")


class FFTSnapshotStore(Store):
    def __init__(self, bulletin: FFTSnapshotBulletin, run_id: str) -> None:
        if not run_id:
            raise ValueError("run_id is required")
        self.bulletin = bulletin
        self.run_id = run_id

    def refresh(self) -> None:
        if self.bulletin._refresh is None:
            return
        result = self.bulletin._refresh()
        if inspect.isawaitable(result):
            asyncio.run(result)

    def read_pointer(self) -> VersionRef | None:
        return self.bulletin.read_latest(self.run_id)

    def advance_pointer(self, ref: VersionRef) -> None:
        self._check_run(ref)
        self.bulletin.advance(ref)

    def claim(self, run_id: str) -> None:
        if run_id != self.run_id:
            raise ValueError(f"store is scoped to run {self.run_id!r}, got {run_id!r}")
        self.bulletin.claim(run_id)

    def read_manifest(self, ref: VersionRef) -> VersionManifest:
        self._check_run(ref)
        source = self.bulletin.resolve(ref)
        return VersionManifest.from_hf_index(source, run_id=ref.run_id)

    def publish(self, manifest: VersionManifest, files_dir: str) -> None:
        self._check_run(manifest.ref)
        source = Path(files_dir)
        self.bulletin.store(
            manifest.ref,
            source,
            metadata=_index_metadata(source),
        )

    def materialize(self, ref: VersionRef) -> str:
        self._check_run(ref)
        return str(self.bulletin.resolve(ref))

    def commit(self) -> None:
        self.bulletin._commit_volume()

    def _check_run(self, ref: VersionRef) -> None:
        if ref.run_id != self.run_id:
            raise ValueError(
                f"store is scoped to run {self.run_id!r}, got {ref.run_id!r}"
            )


class PinnedFFTSnapshotStore(FFTSnapshotStore):
    def __init__(
        self,
        bulletin: FFTSnapshotBulletin,
        ref: VersionRef,
    ) -> None:
        if ref.run_id is None:
            raise ValueError("pinned store requires a run_id")
        super().__init__(bulletin, ref.run_id)
        self.ref = ref

    def read_pointer(self) -> VersionRef:
        return self.ref


def _stamp_index(
    source: Path,
    ref: VersionRef,
    metadata: dict[str, Any],
) -> None:
    path = source / _INDEX
    try:
        index = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        index = _index_safetensors(source)
    weight_map = index.get("weight_map")
    index_metadata = index.get("metadata") or {}
    if not isinstance(weight_map, dict) or (
        not weight_map and not index_metadata.get("delta_encoding")
    ):
        raise ValueError("FFT snapshot has no HF weight map")
    index["metadata"] = {
        **index_metadata,
        **metadata,
        "version": ref.version,
    }
    path.write_text(
        json.dumps(index, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _index_metadata(source: Path) -> dict[str, Any]:
    try:
        metadata = json.loads((source / _INDEX).read_text(encoding="utf-8")).get(
            "metadata"
        )
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid FFT snapshot index: {source / _INDEX}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"invalid FFT snapshot metadata: {source / _INDEX}")
    return metadata


def _index_safetensors(source: Path) -> dict[str, Any]:
    from safetensors import safe_open

    paths = sorted(source.glob("*.safetensors"))
    if not paths:
        raise ValueError(
            "Megatron Bridge produced no complete Hugging Face weight shards"
        )
    weight_map = {}
    for path in paths:
        with safe_open(path, framework="numpy") as weights:
            for name in weights.keys():
                weight_map[name] = path.name
    if not weight_map:
        raise ValueError("FFT snapshot contains no tensors")
    return {
        "metadata": {"total_size": sum(path.stat().st_size for path in paths)},
        "weight_map": weight_map,
    }


def _manifest(
    ref: VersionRef,
    source: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    index = json.loads((source / _INDEX).read_text(encoding="utf-8"))
    expected = {str(name) for name in (index.get("weight_map") or {}).values()} | {
        _INDEX
    }
    names = {
        str(path.relative_to(source))
        for path in source.rglob("*")
        if path.is_file() and path.name != _MANIFEST
    }
    if not expected.issubset(names):
        raise ValueError("FFT snapshot is missing indexed weight files")
    return {
        "ref": ref.identity,
        "metadata": metadata,
        "files": {name: _file_manifest(source / name) for name in sorted(names)},
    }


def _file_manifest(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "size": path.stat().st_size}


def _read_manifest(path: Path) -> dict[str, Any]:
    return json.loads((path / _MANIFEST).read_text(encoding="utf-8"))


def _safe_relative_path(value: str) -> bool:
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts


def _atomic_write(path: Path, value: str) -> None:
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".latest.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise
