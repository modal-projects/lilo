import importlib.util
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


class FakeDist:
    """``torch.distributed`` stand-in whose collectives really synchronize ranks.

    Ranks run concurrently and every collective waits for the whole world, so a
    rank that takes a different path through ``_sync_checkpoint_volume`` breaks
    the barrier instead of quietly passing a sequential simulation.
    """

    timeout = 5.0

    def __init__(self, identities: list[str]) -> None:
        self.identities = identities
        self.world_size = len(identities)
        self._ranks = threading.local()
        self._barrier = threading.Barrier(self.world_size, timeout=self.timeout)
        self._gathered: dict[int, object] = {}

    def run_ranks(self, target) -> list:
        """Run ``target`` on every rank at once and return one future per rank."""

        def run(rank):
            self._ranks.rank = rank
            return target(rank)

        with ThreadPoolExecutor(max_workers=self.world_size) as pool:
            return [pool.submit(run, rank) for rank in range(self.world_size)]

    def identity(self) -> str:
        return self.identities[self.get_rank()]

    def is_available(self) -> bool:
        return True

    def is_initialized(self) -> bool:
        return True

    def get_rank(self) -> int:
        return self._ranks.rank

    def get_world_size(self) -> int:
        return self.world_size

    def barrier(self) -> None:
        self._barrier.wait()

    def all_gather_object(self, output: list, obj) -> None:
        self._gathered[self.get_rank()] = obj
        self._barrier.wait()
        output[:] = [self._gathered[rank] for rank in range(self.world_size)]
        self._barrier.wait()


def _module(monkeypatch, name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    parent_name, _, child_name = name.rpartition(".")
    if parent_name:
        parent = sys.modules.get(parent_name) or _module(monkeypatch, parent_name)
        setattr(parent, child_name, module)
    return module


def _distributed_actor(monkeypatch, identities: list[str], via_task_id: bool = True):
    """Load the actor against a concurrent fake world of node ``identities``.

    Ranks share a process, so the per-rank ``MODAL_TASK_ID`` and hostname are
    resolved from the calling thread's rank rather than from the environment.
    """
    actor = _load_actor(monkeypatch)
    fake_dist = FakeDist(identities)
    actor.dist = fake_dist

    class Environ:
        @staticmethod
        def get(key, default=None):
            if key == "LILO_CHECKPOINT_VOLUME":
                return "ckpt"
            if key == "MODAL_TASK_ID" and via_task_id:
                return fake_dist.identity()
            return default

    monkeypatch.setattr(actor, "os", types.SimpleNamespace(environ=Environ))
    monkeypatch.setattr(
        actor.socket,
        "gethostname",
        (lambda: "modal") if via_task_id else fake_dist.identity,
    )
    return actor, fake_dist


def _load_actor(monkeypatch):
    _module(monkeypatch, "modal")
    torch = _module(monkeypatch, "torch")
    _module(monkeypatch, "torch.distributed")
    megatron_bridge = _module(monkeypatch, "megatron.bridge")
    megatron_bridge.AutoBridge = type("AutoBridge", (), {})
    _module(monkeypatch, "megatron.core").dist_checkpointing = types.ModuleType(
        "megatron.core.dist_checkpointing"
    )
    _module(monkeypatch, "megatron.core.dist_checkpointing")

    fsdp_actor = _module(monkeypatch, "miles.backends.fsdp_utils.actor")
    megatron_actor = _module(monkeypatch, "miles.backends.megatron_utils.actor")
    megatron_model = _module(monkeypatch, "miles.backends.megatron_utils.model")
    checkpoint = _module(monkeypatch, "miles.backends.megatron_utils.lora.checkpoint")
    lora_actor = _module(monkeypatch, "miles.backends.megatron_utils.lora.actor")
    lora_actor.MultiLoRATrainRayActor = type("MultiLoRATrainRayActor", (), {})
    training_cp_utils = _module(monkeypatch, "miles.backends.training_utils.cp_utils")
    training_cp_utils.get_logits_and_tokens_offset_with_cp = lambda *args: None
    training_cp_utils.slice_log_prob_with_cp = lambda *args: None
    training_data = _module(monkeypatch, "miles.backends.training_utils.data")
    training_loss = _module(monkeypatch, "miles.backends.training_utils.loss")
    training_mm_data = _module(monkeypatch, "miles.backends.training_utils.mm_data")
    checkpoint_io = _module(monkeypatch, "miles.backends.training_utils.checkpoint_io")
    checkpoint_io.write_checkpoint_dir = lambda *args, **kwargs: None
    snapshot_publisher = _module(
        monkeypatch, "miles.backends.training_utils.weight_update.snapshot_publisher"
    )
    snapshot_publisher.write_checkpoint_dir = checkpoint_io.write_checkpoint_dir
    checkpoint.write_checkpoint_dir = checkpoint_io.write_checkpoint_dir
    loss_hub = _module(monkeypatch, "miles.backends.training_utils.loss_hub")
    loss_hub.logit_processors = types.SimpleNamespace(
        get_log_probs_and_entropy=lambda *args, **kwargs: None
    )
    loss_hub.math_utils = _module(
        monkeypatch, "miles.backends.training_utils.loss_hub.math_utils"
    )
    loss_hub.tinker_losses = _module(
        monkeypatch, "miles.backends.training_utils.loss_hub.tinker_losses"
    )
    _module(
        monkeypatch, "miles.backends.training_utils.parallel"
    ).get_parallel_state = lambda: None
    torch.distributed = sys.modules["torch.distributed"]
    sys.modules["megatron.core"].dist_checkpointing = sys.modules[
        "megatron.core.dist_checkpointing"
    ]
    sys.modules["miles.backends.megatron_utils.lora"].checkpoint = checkpoint
    sys.modules["miles.backends.megatron_utils.lora"].actor = lora_actor
    sys.modules["miles.backends.training_utils"].checkpoint_io = checkpoint_io
    sys.modules["miles.backends.training_utils"].cp_utils = training_cp_utils
    sys.modules["miles.backends.training_utils"].data = training_data
    sys.modules["miles.backends.training_utils"].loss = training_loss
    sys.modules["miles.backends.training_utils"].mm_data = training_mm_data
    sys.modules["miles.backends.training_utils"].loss_hub = loss_hub
    sys.modules["miles.backends.fsdp_utils"].actor = fsdp_actor
    sys.modules["miles.backends.megatron_utils"].actor = megatron_actor
    sys.modules["miles.backends.megatron_utils"].model = megatron_model

    path = Path(__file__).parents[2] / "src/lilo/backends/miles_runtime/actor.py"
    spec = importlib.util.spec_from_file_location(
        "lilo.backends.miles_runtime._test_actor", path
    )
    assert spec is not None and spec.loader is not None
    actor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(actor)
    return actor


class _FakeVolume:
    """A volume whose committed state holds shards rank 0 cannot see locally."""

    def __init__(self, events: list) -> None:
        self.events = events

    def commit(self) -> None:
        self.events.append("commit")

    def reload(self) -> None:
        raise AssertionError("a colocated engine keeps a capture file open")

    def listdir(self, path):
        return [
            types.SimpleNamespace(path=f"{path}/__0_0.distcp"),
            types.SimpleNamespace(path=f"{path}/__8_0.distcp"),
        ]

    def copy_files(self, src_paths, dst_path, recursive=False) -> None:
        self.events.append(("copy", tuple(src_paths), dst_path, recursive))


def test_checkpoint_is_published_from_committed_state(monkeypatch, tmp_path) -> None:
    """Only the committed state holds every node's shards, so publish from there."""
    actor = _load_actor(monkeypatch)
    monkeypatch.setenv("LILO_CHECKPOINT_VOLUME", "lilo-checkpoints")
    monkeypatch.setenv("LILO_CHECKPOINT_ROOT", str(tmp_path))
    events: list = []
    actor.modal.Volume = types.SimpleNamespace(
        from_name=lambda name: _FakeVolume(events)
    )
    actor.dist.is_available = lambda: False
    actor.dist.is_initialized = lambda: False

    def original(path, write_shards, *args, **kwargs):
        raise AssertionError("miles' rank-0 rename cannot see other nodes' shards")

    actor.checkpoint_io.write_checkpoint_dir = original
    actor._publish_checkpoints_across_nodes()
    monkeypatch.setattr(
        actor,
        "_sync_checkpoint_volume",
        lambda action: events.append(("sync", action)),
    )

    path = tmp_path / "000000" / "miles"
    actor.checkpoint_io.write_checkpoint_dir(
        path,
        lambda directory: (directory / "__0_0.distcp").write_text("shard"),
        {"step": 0},
    )

    assert events == [
        ("sync", "commit"),
        "commit",
        (
            "copy",
            ("000000/_tmp_miles/__8_0.distcp",),
            "000000/miles",
            True,
        ),
        ("sync", "commit"),
    ]
    assert not (tmp_path / "000000" / "_tmp_miles").exists()


def test_publishing_leaves_rank_zero_its_own_files(monkeypatch, tmp_path) -> None:
    """A sampler snapshot is read back locally the moment it is published."""
    actor = _load_actor(monkeypatch)
    monkeypatch.setenv("LILO_CHECKPOINT_VOLUME", "lilo-checkpoints")
    monkeypatch.setenv("LILO_CHECKPOINT_ROOT", str(tmp_path))
    actor.modal.Volume = types.SimpleNamespace(from_name=lambda name: _FakeVolume([]))
    actor.dist.is_available = lambda: False
    actor.dist.is_initialized = lambda: False
    actor.checkpoint_io.write_checkpoint_dir = lambda *args, **kwargs: None
    actor._publish_checkpoints_across_nodes()
    monkeypatch.setattr(actor, "_sync_checkpoint_volume", lambda action: None)

    path = tmp_path / "sampler" / "snapshot"
    actor.checkpoint_io.write_checkpoint_dir(
        path,
        lambda directory: (directory / "adapter_model.safetensors").write_text("w"),
        None,
    )

    assert (path / "adapter_model.safetensors").read_text() == "w"


def test_a_checkpoint_missing_a_peer_shard_is_not_published(
    monkeypatch, tmp_path
) -> None:
    """A short checkpoint only fails on the resume that needs it, far too late."""
    actor = _load_actor(monkeypatch)
    monkeypatch.setenv("LILO_CHECKPOINT_VOLUME", "lilo-checkpoints")
    monkeypatch.setenv("LILO_CHECKPOINT_ROOT", str(tmp_path))
    actor.modal.Volume = types.SimpleNamespace(from_name=lambda name: _FakeVolume([]))
    actor.dist.is_available = lambda: False
    actor.dist.is_initialized = lambda: False
    actor.checkpoint_io.write_checkpoint_dir = lambda *args, **kwargs: None
    actor._publish_checkpoints_across_nodes()
    monkeypatch.setattr(actor, "_sync_checkpoint_volume", lambda action: None)
    monkeypatch.setattr(
        actor, "_written_shard_names", lambda tmp: {"__0_0.distcp", "__8_0.distcp"}
    )
    monkeypatch.setattr(
        _FakeVolume, "listdir", lambda self, path: []
    )  # the peer's commit never landed

    with pytest.raises(RuntimeError, match="__8_0.distcp"):
        actor.checkpoint_io.write_checkpoint_dir(
            tmp_path / "000000" / "miles",
            lambda directory: (directory / "__0_0.distcp").write_text("shard"),
            None,
        )


def test_checkpoints_outside_the_volume_keep_miles_publish(monkeypatch) -> None:
    actor = _load_actor(monkeypatch)
    monkeypatch.setenv("LILO_CHECKPOINT_VOLUME", "lilo-checkpoints")
    monkeypatch.setenv("LILO_CHECKPOINT_ROOT", "/checkpoints")
    events: list[str] = []

    def original(path, write_shards, *args, **kwargs):
        events.append("miles-publish")

    actor.checkpoint_io.write_checkpoint_dir = original
    actor.checkpoint.write_checkpoint_dir = original
    actor.snapshot_publisher.write_checkpoint_dir = original
    actor._publish_checkpoints_across_nodes()

    for module in (actor.checkpoint_io, actor.checkpoint, actor.snapshot_publisher):
        module.write_checkpoint_dir("/tmp/scratch/000000", lambda _: None)

    assert events == ["miles-publish"] * 3


def test_sync_checkpoint_volume_commits_once_per_node(monkeypatch) -> None:
    """Modal cluster containers share a hostname; the task id separates them."""
    actor, fake_dist = _distributed_actor(
        monkeypatch, ["ta-node0", "ta-node0", "ta-node1", "ta-node1"]
    )
    actions: list[tuple[int, str]] = []
    lock = threading.Lock()

    def volume_action(name, action):
        with lock:
            actions.append((fake_dist.get_rank(), action))

    actor._volume_action = volume_action

    futures = fake_dist.run_ranks(lambda rank: actor._sync_checkpoint_volume("commit"))

    assert [future.exception() for future in futures] == [None] * 4
    assert sorted(actions) == [(0, "commit"), (2, "commit")]


def test_committing_never_reloads_the_committing_node(monkeypatch) -> None:
    """Reloading fails outright while the colocated engine holds a capture open."""
    actor, fake_dist = _distributed_actor(monkeypatch, ["ta-node0", "ta-node1"])
    lock = threading.Lock()
    actions: list[tuple[int, str]] = []

    def volume_action(name, action):
        if action == "reload":
            raise RuntimeError("there are open files preventing the operation")
        with lock:
            actions.append((fake_dist.get_rank(), action))

    actor._volume_action = volume_action

    futures = fake_dist.run_ranks(lambda rank: actor._sync_checkpoint_volume("commit"))

    assert [future.exception() for future in futures] == [None] * 2
    assert sorted(actions) == [(0, "commit"), (1, "commit")]


def test_sync_checkpoint_volume_falls_back_to_hostnames(monkeypatch) -> None:
    actor, fake_dist = _distributed_actor(
        monkeypatch, ["head", "worker"], via_task_id=False
    )
    actions: list[tuple[int, str]] = []
    lock = threading.Lock()

    def volume_action(name, action):
        with lock:
            actions.append((fake_dist.get_rank(), action))

    actor._volume_action = volume_action

    futures = fake_dist.run_ranks(lambda rank: actor._sync_checkpoint_volume("commit"))

    assert [future.exception() for future in futures] == [None] * 2
    assert sorted(actions) == [(0, "commit"), (1, "commit")]


def test_sync_checkpoint_volume_fails_on_every_rank(monkeypatch) -> None:
    """A representative that raises must not strand the other ranks in a barrier."""
    actor, fake_dist = _distributed_actor(
        monkeypatch, ["ta-node0", "ta-node0", "ta-node1"]
    )

    def volume_action(name, action):
        if fake_dist.get_rank() == 2:
            raise OSError("volume commit failed")

    actor._volume_action = volume_action

    futures = fake_dist.run_ranks(lambda rank: actor._sync_checkpoint_volume("commit"))

    errors = [future.exception() for future in futures]
    assert all(isinstance(error, RuntimeError) for error in errors)
    assert all("volume commit failed" in str(error) for error in errors)
