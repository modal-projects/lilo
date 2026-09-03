import asyncio
import json

import pytest
from stitch.types import PointerRewind, VersionRef

from lilo.inference.bulletin import (
    ImmutableSnapshotError,
    SnapshotBulletin,
)


def export(tmp_path, name: str, payload: bytes):
    path = tmp_path / name
    path.mkdir()
    (path / "adapter_model.safetensors").write_bytes(payload)
    (path / "adapter_config.json").write_text(json.dumps({"r": 8}))
    return path


def test_publish_is_immutable_monotonic_and_historical(tmp_path):
    commits = []
    board = SnapshotBulletin(tmp_path / "model", commit=lambda: commits.append(1))
    v1 = VersionRef("run-a", 1)
    v2 = VersionRef("run-a", 2)
    source1 = export(tmp_path, "v1", b"one")
    source2 = export(tmp_path, "v2", b"two")

    assert board.publish(v1, source1)
    assert board.publish(v2, source2)
    assert not board.publish(v1, source1)
    assert board.read_latest("run-a") == v2
    assert (
        board.resolve(v1).joinpath("adapter_model.safetensors").read_bytes() == b"one"
    )
    assert (
        board.resolve(v2).joinpath("adapter_model.safetensors").read_bytes() == b"two"
    )
    assert len(commits) == 2

    different = export(tmp_path, "different", b"different")
    with pytest.raises(ImmutableSnapshotError):
        board.publish(v1, different)
    with pytest.raises(PointerRewind):
        board.advance(v1)


def test_runs_have_independent_pointers(tmp_path):
    board = SnapshotBulletin(tmp_path / "model")
    first = VersionRef("run-a", 9)
    second = VersionRef("run-b", 1)

    board.publish(first, export(tmp_path, "first", b"a"))
    board.publish(second, export(tmp_path, "second", b"b"))

    assert board.read_latest("run-a") == first
    assert board.read_latest("run-b") == second
    assert board.resolve(first).is_dir()


def test_publish_retries_commit(tmp_path):
    attempts = 0

    def commit():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("commit interrupted")

    board = SnapshotBulletin(tmp_path / "model", commit=commit)
    ref = VersionRef("run-a", 1)
    source = export(tmp_path, "source", b"one")

    with pytest.raises(RuntimeError, match="commit interrupted"):
        board.publish(ref, source)
    assert not board.publish(ref, source)
    assert attempts == 2


def test_resolve_rejects_corrupted_snapshot(tmp_path):
    board = SnapshotBulletin(tmp_path / "model")
    ref = VersionRef("run-a", 1)
    board.publish(ref, export(tmp_path, "source", b"one"))
    board.snapshot_dir(ref).joinpath("adapter_model.safetensors").write_bytes(
        b"corrupted"
    )

    with pytest.raises(FileNotFoundError):
        board.resolve(ref)


def test_refresh_retries_transient_open_files(tmp_path):
    attempts = 0

    def refresh():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("there are open files preventing the operation")

    asyncio.run(SnapshotBulletin(tmp_path, refresh=refresh).refresh())
    assert attempts == 3
