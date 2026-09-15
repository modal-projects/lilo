import asyncio
import contextlib
import hashlib
import inspect
import json
import os
import shutil
import tempfile
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from stitch.types import VersionRef, decide_pointer_move

PEFT_FILES = ("adapter_model.safetensors", "adapter_config.json")
_MANIFEST = "snapshot.json"
_POINTER = "latest"


class SnapshotNotFound(FileNotFoundError):
    pass


class ImmutableSnapshotError(RuntimeError):
    pass


class SnapshotBulletin:
    def __init__(
        self,
        root: str | Path,
        *,
        refresh: Callable[[], Any] | None = None,
        commit: Callable[[], None] | None = None,
    ) -> None:
        self.root = Path(root)
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
        text = path.read_text(encoding="utf-8").strip()
        return VersionRef.parse(text) if text else None

    def snapshot_dir(self, ref: VersionRef) -> Path:
        self._validate_ref(ref)
        return self._run_dir(ref.run_id) / Path(ref.identity).name

    def resolve(self, ref: VersionRef) -> Path:
        target = self.snapshot_dir(ref)
        try:
            manifest = json.loads((target / _MANIFEST).read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise SnapshotNotFound(ref.identity) from exc
        files = manifest.get("files")
        try:
            invalid = (
                manifest.get("ref") != ref.identity
                or not isinstance(files, dict)
                or any(
                    not isinstance(files.get(name), dict)
                    or _file_manifest(target / name) != files[name]
                    for name in PEFT_FILES
                )
            )
        except FileNotFoundError as exc:
            raise SnapshotNotFound(ref.identity) from exc
        if invalid:
            raise SnapshotNotFound(ref.identity)
        return target

    def advance(self, ref: VersionRef) -> bool:
        self._validate_ref(ref)
        with self._lock:
            current = self.read_latest(ref.run_id)
            if current == ref:
                return False
            decide_pointer_move(current, ref)
            run_dir = self._run_dir(ref.run_id)
            run_dir.mkdir(parents=True, exist_ok=True)
            _atomic_write(run_dir / _POINTER, ref.identity)
            return True

    def publish(self, ref: VersionRef, source_dir: str | Path) -> bool:
        source = Path(source_dir)
        manifest = _manifest(ref, source)
        target = self.snapshot_dir(ref)
        with self._lock:
            created = self._install(target, source, manifest)
            if not created:
                if _read_manifest(target) != manifest:
                    raise ImmutableSnapshotError(ref.identity)
                current = self.read_latest(ref.run_id)
                if current is not None and current.version > ref.version:
                    return False
            moved = self.advance(ref)
            if self._commit is not None:
                self._commit()
            return created or moved

    def _install(
        self,
        target: Path,
        source: Path,
        manifest: dict[str, Any],
    ) -> bool:
        if target.exists():
            return False
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
        try:
            temp.mkdir()
            for name in PEFT_FILES:
                shutil.copy2(source / name, temp / name)
            (temp / _MANIFEST).write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            try:
                os.rename(temp, target)
            except OSError:
                if not target.exists():
                    raise
                return False
            return True
        finally:
            shutil.rmtree(temp, ignore_errors=True)

    @staticmethod
    def _validate_ref(ref: VersionRef) -> None:
        run_id = ref.run_id
        if (
            not run_id
            or ref.version <= 0
            or Path(run_id).name != run_id
            or run_id in {".", ".."}
        ):
            raise ValueError(f"invalid LoRA snapshot ref: {ref.identity}")

    def _run_dir(self, run_id: str | None) -> Path:
        if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
            raise ValueError(f"invalid LoRA run: {run_id!r}")
        return self.root / run_id


def _manifest(ref: VersionRef, source: Path) -> dict[str, Any]:
    files = {name: _file_manifest(source / name) for name in PEFT_FILES}
    return {"ref": ref.identity, "files": files}


def _file_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "size": path.stat().st_size}


def _read_manifest(target: Path) -> dict[str, Any]:
    try:
        return json.loads((target / _MANIFEST).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ImmutableSnapshotError(str(target)) from exc


def _atomic_write(path: Path, text: str) -> None:
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".latest.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp)
        raise
