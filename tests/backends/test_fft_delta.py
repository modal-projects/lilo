import json
import struct
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import pytest
import zstandard
from runtime_stubs import backend_runtime_imports
from safetensors import safe_open
from safetensors.numpy import save_file
from stitch.types import VersionKind, VersionManifest, VersionRef

from lilo.inference.fft_bulletin import FFTSnapshotBulletin

with backend_runtime_imports():
    from lilo.inference import full_delta as delta_module
    from lilo.inference.full_delta import FullDeltaWriter


class Tensor:
    def __init__(self, value):
        self.value = np.ascontiguousarray(value)

    def detach(self):
        return self

    def contiguous(self):
        return self

    def view(self, _dtype):
        return Tensor(self.value.view(np.uint8))

    def reshape(self, *shape):
        return Tensor(self.value.reshape(*shape))

    def numpy(self):
        return self.value

    def numel(self):
        return self.value.size

    def element_size(self):
        return self.value.dtype.itemsize

    def item(self):
        return self.value.item()


class HFWeightTuple(NamedTuple):
    param_name: str
    weight: Tensor
    megatron_param_name: str


class FakeDeltaEncoder:
    def __init__(self, _torch):
        pass

    def prefetch(self, name, current, snapshot):
        if not isinstance(snapshot, Tensor):
            snapshot = Tensor(snapshot)
        current = Tensor(
            current.value.astype(snapshot.value.dtype, copy=False)
            .view(np.uint8)
            .reshape(-1)
        )
        return SimpleNamespace(name=name, current=current, snapshot=snapshot)

    def encode_async(self, chunk, pool):
        return pool.submit(self._encode, chunk)

    @staticmethod
    def _encode(chunk):
        current = chunk.current.value
        snapshot = chunk.snapshot.value.view(np.uint8).reshape(-1)
        positions = np.flatnonzero(current != snapshot)
        snapshot[:] = current
        if not positions.size:
            return None, None
        return (
            delta_module._compress_overwrite(positions, current[positions]),
            delta_module.xxhash.xxh3_128(current).hexdigest(),
        )


def use_fake_runtime(monkeypatch) -> None:
    def read_tensors(path, names):
        with safe_open(path, framework="numpy") as shard:
            return {name: Tensor(shard.get_tensor(name)) for name in names}

    monkeypatch.setattr(delta_module, "DeltaEncoder", FakeDeltaEncoder)
    monkeypatch.setattr(delta_module, "_read_tensors", read_tensors)
    monkeypatch.setattr(
        delta_module,
        "modal",
        SimpleNamespace(
            Volume=SimpleNamespace(
                from_name=lambda name, version: SimpleNamespace(
                    commit=lambda: None,
                )
            )
        ),
    )
    monkeypatch.setenv("LILO_DEFINITION_ID", "full-test")


def test_delta_writer_publishes_first_delta_from_base(tmp_path, monkeypatch) -> None:
    pool = object()
    published_pools = []
    publish_version = delta_module.stitch_publish_version
    monkeypatch.setenv("LILO_DEFINITION_ID", "full-test")
    monkeypatch.setattr(delta_module, "FFTLatestPool", lambda *_: pool)
    monkeypatch.setattr(
        delta_module,
        "stitch_publish_version",
        lambda store, selected, path, **kwargs: (
            published_pools.append(selected),
            publish_version(store, selected, path, **kwargs),
        )[1],
    )
    distributed = SimpleNamespace(
        get_rank=lambda: 0,
        broadcast=lambda tensor, src: None,
    )
    monkeypatch.setattr(
        delta_module,
        "torch",
        SimpleNamespace(
            uint8=object(),
            int32=np.int32,
            cuda=SimpleNamespace(current_device=lambda: 0),
            tensor=lambda value, **kwargs: Tensor(value),
        ),
    )
    monkeypatch.setattr(delta_module, "dist", distributed)
    use_fake_runtime(monkeypatch)
    root = tmp_path / "bulletin"
    board = FFTSnapshotBulletin(root)
    board.claim("run-a")
    source = tmp_path / "base"
    source.mkdir()
    previous = np.array([1.0, 2.0], dtype=np.float16)
    shard = source / "model.safetensors"
    save_file(
        {
            "model.weight": previous,
            "mtp.weight": np.array([9.0], dtype=np.float32),
        },
        shard,
    )
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "model.weight": shard.name,
                    "mtp.weight": shard.name,
                },
            }
        ),
        encoding="utf-8",
    )
    current = np.array([1.0, 3.0], dtype=np.float32)
    canonical_current = current.astype(previous.dtype)

    class Bridge:
        def export_hf_weights(self, model, **kwargs):
            assert model == "model"
            assert kwargs == {
                "cpu": False,
                "show_progress": False,
                "merge_adapter_weights": True,
            }
            yield HFWeightTuple(
                "model.weight",
                Tensor(current),
                "decoder.layers.0.weight",
            )

    writer = FullDeltaWriter()
    snapshot = writer.capture(
        exporter=Bridge(),
        model="model",
        model_id="run-a",
        publish_version=1,
        optimizer_step=1,
        base_model="Qwen/Test",
        hf_checkpoint=str(source),
        bulletin_root=str(root),
    )
    assert isinstance(writer._snapshot["model.weight"], Tensor)
    ref = VersionRef("run-a", 1)
    assert board.read_latest("run-a") == VersionRef("run-a", 0)
    assert not board.staging_dir(ref).exists()

    writer.persist(
        snapshot,
        bulletin_root=str(root),
        bulletin_volume="test-bulletin",
    )

    snapshot = board.resolve(ref)
    manifest = VersionManifest.from_hf_index(snapshot, run_id="run-a")
    assert manifest.kind is VersionKind.DELTA
    index = json.loads(
        (snapshot / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    assert index["metadata"]["base_version"] == 0
    assert index["metadata"]["delta_encoding"] == "overwrite"
    delta_file = snapshot / index["weight_map"]["model.weight"]
    with safe_open(delta_file, framework="numpy") as delta:
        compressed = delta.get_tensor("model.weight").tobytes()
        checksum = delta.metadata()["model.weight"]
    payload = zstandard.ZstdDecompressor().decompress(compressed)
    (count,) = struct.unpack_from("<I", payload)
    positions_end = 4 + 4 * count
    positions = np.frombuffer(payload, dtype="<u4", count=count, offset=4)
    values = np.frombuffer(payload, dtype=np.uint8, offset=positions_end)
    reconstructed = previous.view(np.uint8).reshape(-1).copy()
    reconstructed[positions] = values
    assert reconstructed.tobytes() == canonical_current.tobytes()
    assert len(values) == count
    assert count == np.count_nonzero(
        previous.view(np.uint8).reshape(-1)
        != canonical_current.view(np.uint8).reshape(-1)
    )
    assert (
        checksum
        == delta_module.xxhash.xxh3_128(canonical_current.view(np.uint8)).hexdigest()
    )
    assert board.read_latest("run-a") == ref
    assert published_pools == [pool]


def test_delta_writer_recovers_after_failed_publish(tmp_path, monkeypatch) -> None:
    distributed = SimpleNamespace(
        get_rank=lambda: 0,
        broadcast=lambda tensor, src: None,
    )
    monkeypatch.setattr(
        delta_module,
        "torch",
        SimpleNamespace(
            uint8=object(),
            int32=np.int32,
            cuda=SimpleNamespace(current_device=lambda: 0),
            tensor=lambda value, **kwargs: Tensor(value),
        ),
    )
    monkeypatch.setattr(delta_module, "dist", distributed)
    use_fake_runtime(monkeypatch)
    root = tmp_path / "bulletin"
    board = FFTSnapshotBulletin(root)
    board.claim("run-a")
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    base = np.array([1.0, 2.0], dtype=np.float32)
    shard = base_dir / "model.safetensors"
    save_file({"model.weight": base}, shard)
    (base_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {"model.weight": shard.name},
            }
        ),
        encoding="utf-8",
    )

    current = np.array([1.0, 3.0], dtype=np.float32)

    class Bridge:
        def export_hf_weights(self, model, **kwargs):
            yield HFWeightTuple(
                "model.weight",
                Tensor(current.copy()),
                "decoder.layers.0.weight",
            )

    writer = FullDeltaWriter()
    first = writer.capture(
        exporter=Bridge(),
        model="model",
        model_id="run-a",
        publish_version=1,
        optimizer_step=1,
        base_model="Qwen/Test",
        hf_checkpoint=str(base_dir),
        bulletin_root=str(root),
    )
    writer.persist(
        first,
        bulletin_root=str(root),
        bulletin_volume="test-bulletin",
    )

    current[:] = [4.0, 3.0]
    second = writer.capture(
        exporter=Bridge(),
        model="model",
        model_id="run-a",
        publish_version=2,
        optimizer_step=2,
        base_model="Qwen/Test",
        hf_checkpoint=str(base_dir),
        bulletin_root=str(root),
    )
    publish_version = delta_module.stitch_publish_version

    def fail_after_store(store, _pool, path, *, run_id):
        manifest = VersionManifest.from_hf_index(path, run_id=run_id)
        store.publish(manifest, path)
        raise RuntimeError("publish failed")

    monkeypatch.setattr(
        delta_module,
        "stitch_publish_version",
        fail_after_store,
    )
    with pytest.raises(RuntimeError, match="publish failed"):
        writer.persist(
            second,
            bulletin_root=str(root),
            bulletin_volume="test-bulletin",
        )
    assert board.read_latest("run-a") == VersionRef("run-a", 1)
    assert not board.snapshot_dir(VersionRef("run-a", 2)).exists()

    monkeypatch.setattr(delta_module, "stitch_publish_version", publish_version)
    retry = writer.capture(
        exporter=Bridge(),
        model="model",
        model_id="run-a",
        publish_version=2,
        optimizer_step=2,
        base_model="Qwen/Test",
        hf_checkpoint=str(base_dir),
        bulletin_root=str(root),
    )
    writer.persist(
        retry,
        bulletin_root=str(root),
        bulletin_volume="test-bulletin",
    )

    restored = FullDeltaWriter()
    restored._load_parent(
        board,
        VersionRef("run-a", 2),
        base_checkpoint=base_dir,
    )
    assert restored._snapshot["model.weight"].value.tolist() == [4.0, 3.0]


def test_delta_writer_publishes_unchanged_base_as_version_one(
    tmp_path, monkeypatch
) -> None:
    distributed = SimpleNamespace(
        get_rank=lambda: 0,
        broadcast=lambda tensor, src: None,
    )
    monkeypatch.setattr(
        delta_module,
        "torch",
        SimpleNamespace(
            uint8=object(),
            int32=np.int32,
            cuda=SimpleNamespace(current_device=lambda: 0),
            tensor=lambda value, **kwargs: Tensor(value),
        ),
    )
    monkeypatch.setattr(delta_module, "dist", distributed)
    use_fake_runtime(monkeypatch)
    root = tmp_path / "bulletin"
    source = tmp_path / "base"
    source.mkdir()
    base = np.array([1.0, 2.0], dtype=np.float32)
    shard = source / "model.safetensors"
    save_file({"model.weight": base}, shard)
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {"model.weight": shard.name},
            }
        ),
        encoding="utf-8",
    )

    class Bridge:
        def export_hf_weights(self, model, **kwargs):
            yield HFWeightTuple(
                "model.weight",
                Tensor(base),
                "decoder.layers.0.weight",
            )

    writer = FullDeltaWriter()
    snapshot = writer.capture(
        exporter=Bridge(),
        model="model",
        model_id="run-a",
        publish_version=1,
        optimizer_step=0,
        base_model="Qwen/Test",
        hf_checkpoint=str(source),
        bulletin_root=str(root),
    )
    writer.persist(
        snapshot,
        bulletin_root=str(root),
        bulletin_volume="test-bulletin",
    )

    ref = VersionRef("run-a", 1)
    board = FFTSnapshotBulletin(root)
    snapshot = board.resolve(ref)
    manifest = VersionManifest.from_hf_index(snapshot, run_id="run-a")
    index = json.loads(
        (snapshot / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    assert manifest.kind is VersionKind.DELTA
    assert index["metadata"]["base_version"] == 0
    assert index["metadata"]["optimizer_step"] == 0
    assert index["weight_map"] == {}
    assert board.read_latest("run-a") == ref


def test_delta_writer_loads_bfloat16_parent_as_raw_bytes(tmp_path, monkeypatch) -> None:
    root = tmp_path / "bulletin"
    board = FFTSnapshotBulletin(root)
    ref = VersionRef("run-a", 1)
    source = tmp_path / "full"
    source.mkdir()
    data = b"\x01\x02\x03\x04"
    header = json.dumps(
        {
            "model.weight": {
                "dtype": "BF16",
                "shape": [2],
                "data_offsets": [0, len(data)],
            }
        }
    ).encode()
    header += b" " * (-len(header) % 8)
    shard = source / "model.safetensors"
    shard.write_bytes(struct.pack("<Q", len(header)) + header + data)
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {"model.weight": shard.name},
            }
        ),
        encoding="utf-8",
    )
    board.publish(ref, source, metadata={"optimizer_step": 1})
    monkeypatch.setattr(
        delta_module,
        "_read_tensors",
        lambda _path, _names: {"model.weight": Tensor(np.frombuffer(data, np.uint8))},
    )

    writer = FullDeltaWriter()
    writer._load_full(board.resolve(ref), ref)

    assert writer._snapshot["model.weight"].value.tobytes() == data
