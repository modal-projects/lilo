import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file
from stitch.publish import claim_run, publish_version
from stitch.types import VersionRef

from lilo.inference.fft_bulletin import (
    FFTSnapshotBulletin,
    FFTSnapshotNotFound,
    FFTSnapshotStore,
    PinnedFFTSnapshotStore,
)


def exported(tmp_path, version: int):
    source = tmp_path / f"export-{version}"
    source.mkdir()
    shard = source / "model-00001-of-00001.safetensors"
    save_file(
        {"model.weight": np.array([version], dtype=np.float32)},
        shard,
    )
    (source / "config.json").write_text('{"model_type":"test"}', encoding="utf-8")
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": shard.stat().st_size},
                "weight_map": {"model.weight": shard.name},
            }
        ),
        encoding="utf-8",
    )
    return source


def test_full_snapshot_uses_payload_then_pointer_commit(tmp_path) -> None:
    commits = []
    board = FFTSnapshotBulletin(
        tmp_path / "bulletin",
        commit=lambda: commits.append(True),
    )
    ref = VersionRef("run-a", 1)

    assert board.publish(
        ref,
        exported(tmp_path, 1),
        metadata={"optimizer_step": 3},
    )

    assert commits == [True, True]
    assert board.read_latest("run-a") == ref
    assert board.resolve(ref) == board.snapshot_dir(ref)
    assert board.metadata(ref)["optimizer_step"] == 3


def test_pinned_store_keeps_exact_pointer(tmp_path) -> None:
    board = FFTSnapshotBulletin(tmp_path / "bulletin")
    first = VersionRef("run-a", 1)
    second = VersionRef("run-a", 2)
    board.publish(
        first,
        exported(tmp_path, 1),
        metadata={"optimizer_step": 1},
    )
    store = PinnedFFTSnapshotStore(board, first)
    board.publish(
        second,
        exported(tmp_path, 2),
        metadata={"optimizer_step": 2},
    )

    assert board.read_latest("run-a") == second
    assert store.read_pointer() == first


def test_fft_snapshot_detects_corruption(tmp_path) -> None:
    board = FFTSnapshotBulletin(tmp_path / "bulletin")
    ref = VersionRef("run-a", 1)
    board.publish(
        ref,
        exported(tmp_path, 1),
        metadata={"optimizer_step": 1},
    )
    config = board.snapshot_dir(ref) / "config.json"
    config.write_text("corrupt", encoding="utf-8")

    with pytest.raises(FFTSnapshotNotFound):
        board.resolve(ref)


def test_full_snapshot_base_claim_is_idempotent(tmp_path) -> None:
    commits = []
    board = FFTSnapshotBulletin(
        tmp_path / "bulletin",
        commit=lambda: commits.append(True),
    )

    assert board.claim("run-a") == VersionRef("run-a", 0)
    assert board.claim("run-a") == VersionRef("run-a", 0)
    assert commits == [True]


def test_full_snapshot_store_publishes_through_stitch(tmp_path) -> None:
    commits = []
    board = FFTSnapshotBulletin(
        tmp_path / "bulletin",
        commit=lambda: commits.append(True),
    )
    store = FFTSnapshotStore(board, "run-a")
    claim_run(store, None, "run-a")
    ref = VersionRef("run-a", 1)
    source = exported(tmp_path, 1)
    staging = board.staging_dir(ref)
    staging.parent.mkdir(parents=True)
    source.rename(staging)
    board.prepare(ref, staging, metadata={"optimizer_step": 1})

    assert publish_version(store, None, str(staging), run_id="run-a") == ref

    assert commits == [True, True, True]
    assert store.read_pointer() == ref
    assert store.read_manifest(ref).ref == ref
    assert Path(store.materialize(ref)) == board.snapshot_dir(ref)
