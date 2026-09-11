import json
import pytest
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lilo.backends import ForwardBatch, ForwardItem, ModelSpec
from lilo.backends.miles_config import MilesBackendConfig, parse_backend_config
from lilo.backends.miles_lora import MilesCommandBackend
from lilo.inference.bulletin import SnapshotBulletin
from stitch.types import VersionRef
from tinker import AdamParams, Datum, LoraConfig, ModelInput, TensorData


class FakeMilesRuntime:
    def __init__(self) -> None:
        self.calls = []
        self.closed = False

    def load_slot(
        self,
        slot,
        rank,
        alpha,
        *,
        checkpoint=None,
        restore_optimizer=True,
    ):
        self.calls.append(("load_slot", slot, rank, alpha, checkpoint, restore_optimizer))

    def unload_slot(self, slot):
        self.calls.append(("unload_slot", slot))

    def forward_backward(
        self,
        slot_rows,
        *,
        loss_fn,
        loss_fn_config,
        forward_only,
    ):
        self.calls.append(
            (
                "forward_backward",
                tuple(slot for slot, _ in slot_rows),
                loss_fn,
                loss_fn_config,
                forward_only,
            )
        )
        return [
            {
                "loss": float(slot + 1),
                "logprobs": [-float(slot + 1)] * row["target_len"],
            }
            for slot, row in slot_rows
        ]

    def optim_step(self, adam_params_by_slot):
        self.calls.append(("optim_step", adam_params_by_slot))
        return {slot: slot + 0.5 for slot in adam_params_by_slot}

    def save_slot(self, slot, path):
        self.calls.append(("save_slot", slot, path))
        destination = Path(path)
        destination.mkdir(parents=True)
        (destination / "adapter_megatron_tp0_pp0.pt").write_bytes(b"weights")
        (destination / "optim_rank0.pt").write_bytes(b"optimizer")

    def export_slot_peft(self, **kwargs):
        self.calls.append(("export_slot_peft", kwargs))
        destination = Path(kwargs["path"])
        destination.mkdir(parents=True)
        (destination / "adapter_model.safetensors").write_bytes(b"adapter")
        (destination / "adapter_config.json").write_text("{}", encoding="utf-8")

    def close(self):
        self.closed = True


def _config(**overrides) -> MilesBackendConfig:
    values = {
        "hf_checkpoint": "/models/qwen",
        "model_type": "qwen3-4B",
        "actor_num_gpus_per_node": 2,
        "tensor_model_parallel_size": 2,
        "max_lora_slots": 2,
    }
    values.update(overrides)
    return MilesBackendConfig(**values)


def _spec(rank=8) -> ModelSpec:
    return ModelSpec(
        base_model="Qwen/Qwen3-4B",
        parameterization="lora",
        lora_config=LoraConfig(rank=rank),
    )


def _datum(tokens, final_token) -> Datum:
    targets = [*tokens[1:], final_token]
    return Datum(
        ModelInput.from_ints(tokens),
        {
            "target_tokens": TensorData(
                data=targets,
                dtype="int64",
                shape=[len(targets)],
            ),
            "weights": TensorData(
                data=[1.0] * len(targets),
                dtype="float32",
                shape=[len(targets)],
            ),
        },
    )


def _backend(tmp_path, runtime=None) -> MilesCommandBackend:
    return MilesCommandBackend(
        _config(),
        checkpoint_dir=tmp_path / "checkpoints",
        capture_dir=tmp_path / "captures",
        base_model="Qwen/Qwen3-4B",
        runtime=runtime or FakeMilesRuntime(),
    )


def test_config_translates_stable_fields_to_miles_arguments() -> None:
    config, checkpoint_dir, capture_dir = parse_backend_config(
        {
            "miles": {
                "hf_checkpoint": "/models/qwen",
                "model_type": "qwen3-4B",
                "actor_num_gpus_per_node": 4,
                "tensor_model_parallel_size": 4,
                "expert_model_parallel_size": 2,
                "max_lora_slots": 4,
                "extra_args": ["--recompute-granularity", "full"],
            },
            "checkpoint_dir": "/checkpoints",
        }
    )

    assert config.world_size == 4
    assert config.peft_target_modules == (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    )
    arguments = config.miles_arguments()
    assert arguments[arguments.index("--multi-lora-n-adapters") + 1] == "4"
    assert "--debug-train-only" in arguments
    assert arguments[-2:] == ["--recompute-granularity", "full"]
    assert checkpoint_dir == Path("/checkpoints")
    assert capture_dir == Path("/tmp/lilo-miles-captures")


def test_config_rejects_data_parallel_topology() -> None:
    config = _config(
        actor_num_gpus_per_node=4,
        tensor_model_parallel_size=2,
    )

    with pytest.raises(ValueError, match="data parallel size 1"):
        config.validate()


def test_backend_routes_batches_and_preserves_per_model_state(tmp_path) -> None:
    runtime = FakeMilesRuntime()
    backend = _backend(tmp_path, runtime)
    backend.accept_model("model-a", _spec())
    backend.accept_model("model-b", _spec())

    batch = ForwardBatch(
        items=(
            ForwardItem("model-b", (_datum([4, 5], 6),)),
            ForwardItem("model-a", (_datum([1, 2, 3], 4),)),
        ),
        loss_fn="cross_entropy",
    )
    outputs = backend.forward_backward(batch)

    assert runtime.calls[-1][0:3] == (
        "forward_backward",
        (0, 1),
        "cross_entropy",
    )
    assert outputs[0].loss_fn_outputs[0]["logprobs"].data == [-2.0, -2.0]
    assert outputs[1].loss_fn_outputs[0]["logprobs"].data == [-1.0, -1.0, -1.0]
    assert backend.jobs["model-a"].accumulating
    assert backend.jobs["model-b"].accumulating

    (result,) = backend.optim_step(
        ("model-a",),
        AdamParams(learning_rate=2e-4),
    )
    assert result.metrics["grad_norm:mean"] == 0.5
    assert not backend.jobs["model-a"].accumulating
    assert backend.jobs["model-b"].accumulating
    assert backend.jobs["model-a"].optimizer_step == 1

    backend.accept_model("model-a", _spec())
    backend.unload_model("model-a")
    backend.unload_model("model-a")
    assert backend.job_to_slot == {"model-b": 1}
    assert 0 in backend.free_slots


def test_checkpoint_capture_persist_and_restore(tmp_path, monkeypatch) -> None:
    runtime = FakeMilesRuntime()
    backend = _backend(tmp_path, runtime)
    monkeypatch.setenv("LILO_DEFINITION_ID", "qwen3_4b_miles_lora_2k")
    backend.accept_model("model-a", _spec())
    backend.jobs["model-a"].optimizer_step = 3

    backend.capture_checkpoint(
        "model-a",
        "capture-a",
        destination="step-3",
        include_optimizer=True,
    )
    uri = backend.persist_checkpoint("capture-a", "step-3")
    metadata = json.loads((Path(uri) / "metadata.json").read_text())

    assert metadata["backend"] == "miles"
    assert metadata["optimizer_step"] == 3
    assert (Path(uri) / "optim_rank0.pt").exists()
    backend.load_checkpoint("model-a", uri, restore_optimizer=True)
    assert runtime.calls[-1] == ("load_slot", 0, 8, 32.0, uri, True)
    assert backend.jobs["model-a"].optimizer_step == 3


def test_checkpoint_restore_rejects_different_lora_targets(
    tmp_path,
    monkeypatch,
) -> None:
    backend = _backend(tmp_path)
    monkeypatch.setenv("LILO_DEFINITION_ID", "qwen3_4b_miles_lora_2k")
    backend.accept_model("model-a", _spec())
    backend.capture_checkpoint(
        "model-a",
        "capture-a",
        destination="step-1",
        include_optimizer=False,
    )
    uri = Path(backend.persist_checkpoint("capture-a", "step-1"))
    metadata = json.loads((uri / "metadata.json").read_text())
    metadata["lora_config"]["train_attn"] = False
    (uri / "metadata.json").write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="LoRA targets"):
        backend.load_checkpoint("model-a", str(uri))


def test_sampler_capture_publishes_existing_lilo_format(tmp_path, monkeypatch) -> None:
    runtime = FakeMilesRuntime()
    backend = _backend(tmp_path, runtime)
    bulletin_root = tmp_path / "bulletin"
    monkeypatch.setenv("LILO_BULLETIN_ROOT", str(bulletin_root))
    monkeypatch.delenv("LILO_BULLETIN_VOLUME", raising=False)
    backend.accept_model("model-a", _spec())

    publication = backend.capture_sampler_snapshot("model-a", "capture-a", 7)
    backend.persist_sampler_snapshot("capture-a")

    assert publication.publish_version == 7
    resolved = SnapshotBulletin(bulletin_root).resolve(VersionRef("model-a", 7))
    assert (resolved / "adapter_model.safetensors").read_bytes() == b"adapter"
    assert "capture-a" not in backend._sampler_captures


def test_sampler_snapshots_persist_concurrently_without_crossing_adapters(tmp_path, monkeypatch) -> None:
    from lilo.backends import miles_lora

    backend = _backend(tmp_path, FakeMilesRuntime())
    root = tmp_path / "bulletin"
    monkeypatch.setenv("LILO_BULLETIN_ROOT", str(root))
    monkeypatch.setenv("LILO_BULLETIN_VOLUME", "test-volume")
    committing = threading.Barrier(2, timeout=5)
    monkeypatch.setattr(miles_lora, "_commit_volume", lambda name: committing.wait())
    for model in ["model-a", "model-b"]:
        backend.accept_model(model, _spec())
        backend.capture_sampler_snapshot(model, model, 1)
        capture = backend._sampler_captures[model]
        (capture["path"] / "adapter_model.safetensors").write_bytes(model.encode())

    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(backend.persist_sampler_snapshot, model) for model in ["model-a", "model-b"]]
        for future in futures:
            future.result(timeout=10)

    bulletin = SnapshotBulletin(root)
    for model in ["model-a", "model-b"]:
        ref = VersionRef(model, 1)
        assert bulletin.read_latest(model) == ref
        assert (bulletin.resolve(ref) / "adapter_model.safetensors").read_bytes() == model.encode()
    assert not backend._sampler_captures


def test_backend_rejects_unsupported_per_model_miles_options(tmp_path) -> None:
    backend = _backend(tmp_path)
    with pytest.raises(ValueError, match="per-model seeds"):
        backend.accept_model(
            "model-a",
            ModelSpec(
                base_model="Qwen/Qwen3-4B",
                parameterization="lora",
                lora_config=LoraConfig(rank=8, seed=7),
            ),
        )


def test_build_executor_uses_single_process_mode(monkeypatch, tmp_path) -> None:
    from lilo.backends import miles_lora

    config = _config()
    captured = {}

    def construct(parsed_config, **kwargs):
        captured.update(config=parsed_config, **kwargs)
        return "backend"

    monkeypatch.setenv("LILO_BACKEND_CONFIG", "{}")
    monkeypatch.setenv("LILO_BASE_MODEL", "Qwen/Qwen3-4B")
    monkeypatch.setattr(
        miles_lora,
        "parse_backend_config",
        lambda _value: (config, tmp_path / "checkpoints", tmp_path / "captures"),
    )
    monkeypatch.setattr(miles_lora, "MilesCommandBackend", construct)

    executor = miles_lora.build_executor()

    assert executor.backend == "backend"
    assert executor.command_group is None
    assert executor.checkpoint_persistence_group is None
    assert executor.sampler_persistence_group is None
    assert captured["config"] == config
