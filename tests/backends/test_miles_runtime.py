import asyncio
from types import SimpleNamespace

import pytest

from lilo.errors import BackendFailed

from lilo.backends.miles_runtime.runtime import (
    MilesRuntime,
    _configure_actor_spec,
    _materialize_capture,
)


def test_capture_detaches_upstream_version_directory(tmp_path):
    version = tmp_path / "_version_capture_123"
    version.mkdir()
    (version / "weights").write_bytes(b"weights")
    capture = tmp_path / "capture"
    capture.symlink_to(version.name, target_is_directory=True)
    _materialize_capture(str(capture))
    assert not capture.is_symlink()
    assert (capture / "weights").read_bytes() == b"weights"
    assert not version.exists()


def test_capture_rejects_unrelated_symlink(tmp_path):
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    capture = tmp_path / "capture"
    capture.symlink_to(unrelated.name, target_is_directory=True)
    with pytest.raises(RuntimeError, match="unexpected Miles capture"):
        _materialize_capture(str(capture))
    assert capture.is_symlink()
    assert unrelated.is_dir()


def test_actor_override_preserves_non_multilora_workers():
    class Spec:
        def __init__(self, worker_class):
            self.worker_class = worker_class

        def model_copy(self, *, update):
            return Spec(update["worker_class"])

    specs = SimpleNamespace(_compute_spec_trainer=lambda name: Spec(name))
    _configure_actor_spec(specs)
    once = specs._compute_spec_trainer
    _configure_actor_spec(specs)
    assert specs._compute_spec_trainer is once
    assert once("fft").worker_class == "fft"
    assert once(
        "miles.backends.megatron_utils.lora.actor.MultiLoRATrainRayActor"
    ).worker_class == ("lilo.backends.miles_runtime.actor.LiloMilesTrainRayActor")


def test_upstream_error_result_invalidates_runtime():
    runtime = MilesRuntime.__new__(MilesRuntime)
    runtime._closed = False
    runtime._failure = None
    runtime._call = asyncio.run

    async def error():
        return {"error": "trainer cell lost"}

    with pytest.raises(BackendFailed, match="trainer cell lost"):
        runtime._run(error())
    with pytest.raises(BackendFailed, match="unavailable"):
        runtime._run(error())


@pytest.mark.parametrize("failed_rank", [0, 1])
def test_weights_only_worker_error_invalidates_runtime_before_capture(
    tmp_path, failed_rank
):
    runtime = MilesRuntime.__new__(MilesRuntime)
    runtime._closed = False
    runtime._failure = None
    runtime._call = asyncio.run
    calls = []

    async def execute(method, **kwargs):
        calls.append(method)
        results = [None, None]
        results[failed_rank] = {"error": "save worker lost"}
        return results

    runtime._trainer = SimpleNamespace(_execute_slots=execute)
    capture = str(tmp_path / "capture")
    with pytest.raises(BackendFailed, match="save worker lost"):
        runtime.save_slot(0, capture, include_optimizer=False)
    with pytest.raises(BackendFailed, match="unavailable"):
        runtime.save_slot(0, capture, include_optimizer=False)
    assert calls == ["save_slot_weights"]
    assert not (tmp_path / "capture").exists()
