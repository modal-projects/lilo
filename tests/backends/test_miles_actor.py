import importlib.util
import sys
import types
from pathlib import Path


def _module(monkeypatch, name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    parent_name, _, child_name = name.rpartition(".")
    if parent_name:
        parent = sys.modules.get(parent_name) or _module(monkeypatch, parent_name)
        setattr(parent, child_name, module)
    return module


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
    _module(
        monkeypatch, "miles.backends.training_utils.checkpoint_io"
    ).write_checkpoint_dir = lambda *args: None
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
    sys.modules["miles.backends.training_utils"].cp_utils = training_cp_utils
    sys.modules["miles.backends.training_utils"].data = training_data
    sys.modules["miles.backends.training_utils"].loss = training_loss
    sys.modules["miles.backends.training_utils"].mm_data = training_mm_data
    sys.modules["miles.backends.training_utils"].loss_hub = loss_hub
    sys.modules["miles.backends.fsdp_utils"].actor = fsdp_actor
    sys.modules["miles.backends.megatron_utils"].actor = megatron_actor
    sys.modules["miles.backends.megatron_utils"].model = megatron_model

    path = Path(__file__).parents[2] / "src/lilo/backends/miles_runtime/actor.py"
    spec = importlib.util.spec_from_file_location("test_miles_actor_module", path)
    assert spec is not None and spec.loader is not None
    actor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(actor)
    return actor


def test_sync_checkpoint_volume_commits_then_reloads_each_node(monkeypatch) -> None:
    actor = _load_actor(monkeypatch)
    actions: list[tuple[int, str]] = []

    class FakeDist:
        rank = 0

        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def is_initialized() -> bool:
            return True

        @classmethod
        def get_rank(cls) -> int:
            return cls.rank

        @staticmethod
        def get_world_size() -> int:
            return 2

        @staticmethod
        def barrier() -> None:
            return None

        @staticmethod
        def all_gather_object(hosts, hostname) -> None:
            hosts[:] = ["head", "worker"]

    actor.dist = FakeDist
    actor._volume_action = lambda name, action: actions.append((FakeDist.rank, action))
    monkeypatch.setenv("LILO_CHECKPOINT_VOLUME", "checkpoints")

    for rank, hostname in enumerate(("head", "worker")):
        FakeDist.rank = rank
        monkeypatch.setattr(
            actor.socket, "gethostname", lambda hostname=hostname: hostname
        )
        actor._sync_checkpoint_volume("commit")

    assert actions == [
        (0, "commit"),
        (0, "reload"),
        (1, "commit"),
        (1, "reload"),
    ]
