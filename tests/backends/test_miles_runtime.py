import asyncio
from types import SimpleNamespace

import pytest

from lilo.backends.miles_runtime import runtime as miles_runtime
from lilo.backends.miles_runtime.runtime import (
    MilesRuntime,
    _allow_context_parallel_multi_lora,
    _configure_actor_spec,
    _materialize_capture,
    _require_cluster_nodes,
    _worker_env,
)
from lilo.errors import BackendFailed


def test_worker_env_carries_volume_names_to_other_nodes(monkeypatch):
    """Actors on worker nodes only see what the pre-existing cluster was given."""
    monkeypatch.setenv("LILO_CHECKPOINT_VOLUME", "lilo-checkpoints")
    monkeypatch.setenv("LILO_BULLETIN_VOLUME", "lilo-bulletin")
    monkeypatch.delenv("LILO_BULLETIN_ROOT", raising=False)
    monkeypatch.setenv("LILO_BACKEND_CONFIG", "{}")
    monkeypatch.setenv("TRITON_CACHE_DIR", "/root/.cache/kernel-cache/triton")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "/root/.cache/kernel-cache/inductor")
    assert _worker_env() == {
        "LILO_CHECKPOINT_VOLUME": "lilo-checkpoints",
        "LILO_BULLETIN_VOLUME": "lilo-bulletin",
        "TRITON_CACHE_DIR": "/root/.cache/kernel-cache/triton",
        "TORCHINDUCTOR_CACHE_DIR": "/root/.cache/kernel-cache/inductor",
    }


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


class _FakeRay:
    def __init__(self, node_states):
        self._node_states = iter(node_states)

    def nodes(self):
        return next(self._node_states)


class _FakeTime:
    def __init__(self):
        self.current = 0.0

    def monotonic(self):
        self.current += 1.0
        return self.current

    def sleep(self, _seconds):
        pass


def test_require_cluster_nodes_waits_for_nodes_to_register(monkeypatch):
    monkeypatch.setattr(miles_runtime, "time", _FakeTime())
    ray = _FakeRay(
        [
            [{"Alive": True, "Resources": {"GPU": 8}}],
            [
                {"Alive": True, "Resources": {"GPU": 8}},
                {"Alive": True, "Resources": {"GPU": 8}},
            ],
        ]
    )

    _require_cluster_nodes(ray, nodes=2, world_size=16, timeout=10.0)


def test_require_cluster_nodes_rejects_gpus_on_one_node(monkeypatch):
    monkeypatch.setattr(miles_runtime, "time", _FakeTime())
    ray = _FakeRay([[{"Alive": True, "Resources": {"GPU": 16}}]])

    with pytest.raises(
        BackendFailed,
        match="Ray cluster exposes 16 GPUs on 1 nodes, trainer needs 16 GPUs on 2 nodes",
    ):
        _require_cluster_nodes(ray, nodes=2, world_size=16, timeout=0.0)


def test_require_cluster_nodes_rejects_short_gpu_total(monkeypatch):
    monkeypatch.setattr(miles_runtime, "time", _FakeTime())
    ray = _FakeRay(
        [
            [
                {"Alive": True, "Resources": {"GPU": 8}},
                {"Alive": True, "Resources": {"GPU": 4}},
            ]
        ]
    )

    with pytest.raises(
        BackendFailed,
        match="Ray cluster exposes 12 GPUs on 2 nodes, trainer needs 16 GPUs on 2 nodes",
    ):
        _require_cluster_nodes(ray, nodes=2, world_size=16, timeout=0.0)


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


@pytest.mark.parametrize("fails", [False, True])
def test_context_parallel_guard_restores_args_and_installs_once(fails):
    seen = []

    def validate(args):
        seen.append(args.context_parallel_size)
        if fails:
            raise ValueError("another LoRA constraint failed")

    lora_arguments = SimpleNamespace(validate_multi_lora_args=validate)
    _allow_context_parallel_multi_lora(lora_arguments)
    once = lora_arguments.validate_multi_lora_args
    _allow_context_parallel_multi_lora(lora_arguments)
    assert lora_arguments.validate_multi_lora_args is once
    args = SimpleNamespace(context_parallel_size=8)
    if fails:
        with pytest.raises(ValueError, match="another LoRA constraint"):
            once(args)
    else:
        once(args)
    assert seen == [1]
    assert args.context_parallel_size == 8
