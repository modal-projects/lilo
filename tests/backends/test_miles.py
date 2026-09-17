import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from stitch.types import VersionRef
from tinker import AdamParams, Datum, LoraConfig, ModelInput, TensorData

from lilo.backends import ForwardBatch, ForwardItem, ModelSpec
from lilo.backends.miles_config import MilesBackendConfig, parse_backend_config
from lilo.backends.miles_lora import MilesCommandBackend
from lilo.backends.miles_runtime.data import pad_slot_rows
from lilo.inference.bulletin import SnapshotBulletin


class FakeMilesRuntime:
    revision = "a" * 40

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
        self.calls.append(
            ("load_slot", slot, rank, alpha, checkpoint, restore_optimizer)
        )

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
        self.last_slot_rows = tuple((slot, dict(row)) for slot, row in slot_rows)
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
        return {slot: {"grad_norm": slot + 0.5} for slot in adam_params_by_slot}

    def save_slot(self, slot, path, *, include_optimizer=True):
        self.calls.append(("save_slot", slot, path))
        destination = Path(path)
        destination.mkdir(parents=True)
        (destination / "adapter_megatron_tp0_pp0.pt").write_bytes(b"weights")
        (destination / "metadata.json").write_text('{"sharded_backend": "torch_dist"}')
        if include_optimizer:
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
    assert config.data_parallel_size == 1
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


def test_config_rejects_non_divisible_data_parallel_topology() -> None:
    config = _config(
        actor_num_gpus_per_node=4,
        tensor_model_parallel_size=3,
    )

    with pytest.raises(ValueError, match="must be a multiple"):
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


def test_pad_slot_rows_adds_zero_weight_rows() -> None:
    rows = tuple((0, {"tokens": [1, 2], "target_len": 1}) for _ in range(11))
    padded = pad_slot_rows(rows, 2, "cross_entropy")
    assert len(padded) == 12
    assert padded[-1] == (0, {"tokens": [1, 2], "target_len": 1})

    rows = tuple((0, {"tokens": [1, 2], "target_len": 1}) for _ in range(3))
    assert len(pad_slot_rows(rows, 4, "cross_entropy")) == 4


def test_backend_pads_dp_ragged_batches(tmp_path) -> None:
    runtime = FakeMilesRuntime()
    backend = MilesCommandBackend(
        _config(actor_num_gpus_per_node=4, tensor_model_parallel_size=2),
        checkpoint_dir=tmp_path / "checkpoints",
        capture_dir=tmp_path / "captures",
        base_model="Qwen/Qwen3-4B",
        runtime=runtime,
    )
    backend.accept_model("model-a", _spec())

    outputs = backend.forward_backward(
        ForwardBatch(
            items=(ForwardItem("model-a", (_datum([1, 2], 3),)),),
            loss_fn="cross_entropy",
        )
    )
    assert len(outputs) == 1
    assert len(runtime.last_slot_rows) == 2
    assert runtime.last_slot_rows[-1][1] == {
        "tokens": [1, 2],
        "target_len": 1,
        "weights": [0.0],
    }

    backend.forward_backward(
        ForwardBatch(
            items=(
                ForwardItem(
                    "model-a",
                    (
                        _datum([1, 2], 3),
                        _datum([4, 5], 6),
                    ),
                ),
            ),
            loss_fn="cross_entropy",
        )
    )
    assert len(runtime.last_slot_rows) == 2

    dp1_runtime = FakeMilesRuntime()
    dp1_backend = _backend(tmp_path / "dp1", dp1_runtime)
    dp1_backend.accept_model("model-a", _spec())
    dp1_backend.forward_backward(
        ForwardBatch(
            items=(ForwardItem("model-a", (_datum([1, 2], 3),)),),
            loss_fn="cross_entropy",
        )
    )
    assert len(dp1_runtime.last_slot_rows) == 1


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
    assert Path(uri) == tmp_path / "checkpoints" / "step-3" / "model-a"
    metadata = json.loads((Path(uri) / "metadata.json").read_text())

    assert metadata["miles_revision"] == runtime.revision
    assert metadata["backend"] == "miles"
    assert json.loads((Path(uri) / "miles" / "metadata.json").read_text()) == {
        "sharded_backend": "torch_dist"
    }
    assert metadata["optimizer_step"] == 3
    assert (Path(uri) / "miles" / "optim_rank0.pt").exists()
    backend.load_checkpoint("model-a", uri, restore_optimizer=True)
    assert runtime.calls[-1] == (
        "load_slot",
        0,
        8,
        32.0,
        str(Path(uri) / "miles"),
        True,
    )
    assert backend.jobs["model-a"].optimizer_step == 3


def test_checkpoint_restore_refreshes_files_saved_by_another_container(
    tmp_path, monkeypatch
):
    from lilo.backends import miles_lora

    runtime = FakeMilesRuntime()
    backend = _backend(tmp_path, runtime)
    backend.accept_model("model-a", _spec())
    backend.capture_checkpoint(
        "model-a", "capture-a", destination="step-1", include_optimizer=True
    )
    checkpoint = Path(backend.persist_checkpoint("capture-a", "step-1"))
    # Model the stale mount: the committed checkpoint is absent until reload.
    staged = tmp_path / "remote-checkpoint"
    checkpoint.rename(staged)
    monkeypatch.setenv("LILO_CHECKPOINT_VOLUME", "checkpoint-volume")
    refreshed = []

    def reload_volume(name):
        assert name == "checkpoint-volume"
        staged.rename(checkpoint)
        refreshed.append(name)

    monkeypatch.setattr(miles_lora, "_reload_volume", reload_volume)
    backend.load_checkpoint("model-a", str(checkpoint), restore_optimizer=True)
    assert refreshed == ["checkpoint-volume"]
    assert runtime.calls[-1] == (
        "load_slot",
        0,
        8,
        32.0,
        str(checkpoint / "miles"),
        True,
    )


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
    backend.publish_sampler_snapshot("capture-a")

    assert publication.publish_version == 7
    resolved = SnapshotBulletin(bulletin_root).resolve(VersionRef("model-a", 7))
    assert (resolved / "adapter_model.safetensors").read_bytes() == b"adapter"
    assert "capture-a" not in backend._sampler_captures


def test_sampler_snapshots_persist_concurrently_without_crossing_adapters(
    tmp_path, monkeypatch
) -> None:
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
        futures = [
            workers.submit(backend.publish_sampler_snapshot, model)
            for model in ["model-a", "model-b"]
        ]
        for future in futures:
            future.result(timeout=10)

    bulletin = SnapshotBulletin(root)
    for model in ["model-a", "model-b"]:
        ref = VersionRef(model, 1)
        assert bulletin.read_latest(model) == ref
        assert (
            bulletin.resolve(ref) / "adapter_model.safetensors"
        ).read_bytes() == model.encode()
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


def test_nonfinite_optimizer_skip_does_not_advance_policy(tmp_path):
    runtime = FakeMilesRuntime()
    runtime.optim_step = lambda params: {
        slot: {"skipped_nonfinite": 1.0} for slot in params
    }
    backend = _backend(tmp_path, runtime)
    backend.accept_model("model-a", _spec())
    backend.jobs["model-a"].accumulating = True
    (result,) = backend.optim_step(("model-a",), AdamParams(learning_rate=1e-5))
    assert result.metrics == {
        "update_successful:mean": 0.0,
        "skipped_nonfinite:sum": 1.0,
    }
    assert backend.jobs["model-a"].optimizer_step == 0
    assert not backend.jobs["model-a"].accumulating


def test_datum_preserves_explicit_targets_for_upstream():
    from lilo.backends.miles_runtime.data import _datum_row

    row = _datum_row(_datum([1, 2, 3], 4), "cross_entropy", 0)
    assert row["tokens"] == [1, 2, 3, 4]
    assert row["target_tokens"] == [2, 3, 4]


def test_weights_only_capture_excludes_optimizer(tmp_path):
    backend = _backend(tmp_path)
    backend.accept_model("model-a", _spec())
    backend.capture_checkpoint(
        "model-a", "weights-only", destination="weights", include_optimizer=False
    )
    uri = Path(backend.persist_checkpoint("weights-only", "weights"))
    assert not (uri / "miles" / "optim_rank0.pt").exists()
    assert json.loads((uri / "metadata.json").read_text())["has_optimizer"] is False
    with pytest.raises(ValueError, match="optimizer"):
        backend.load_checkpoint("model-a", str(uri), restore_optimizer=True)


def test_checkpoint_rejects_different_resolved_main_commit(tmp_path):
    runtime = FakeMilesRuntime()
    backend = _backend(tmp_path, runtime)
    backend.accept_model("model-a", _spec())
    backend.capture_checkpoint(
        "model-a", "capture-a", destination="step-1", include_optimizer=True
    )
    uri = backend.persist_checkpoint("capture-a", "step-1")
    runtime.revision = "b" * 40
    calls = list(runtime.calls)
    with pytest.raises(ValueError, match="Miles revision does not match"):
        backend.load_checkpoint("model-a", uri, restore_optimizer=True)
    assert runtime.calls == calls


def test_miles_checkpoint_storage_lifecycle_uses_control_plane_layout(tmp_path):
    import asyncio
    from types import SimpleNamespace
    from lilo.control_plane.service import ControlPlane
    from lilo.providers.modal.checkpoint_storage import ModalCheckpointStorage

    backend = _backend(tmp_path, FakeMilesRuntime())
    storage = ModalCheckpointStorage(
        SimpleNamespace(reload=lambda: None, commit=lambda: None),
        str(backend.checkpoint_dir),
    )
    plane = SimpleNamespace(checkpoint_root=str(backend.checkpoint_dir))
    paths = {}
    for model in ("a", "b"):
        backend.accept_model(model, _spec())
        backend.capture_checkpoint(
            model, model, destination="final", include_optimizer=True
        )
        paths[model] = backend.persist_checkpoint(model, "final")
        path = ControlPlane.tinker_path(plane, paths[model])
        assert path == f"tinker://{model}/weights/final"
        assert ControlPlane.resolve_checkpoint_path(plane, path) == paths[model]
        backend.load_checkpoint(model, paths[model], restore_optimizer=True)

    async def check():
        assert {
            (entry["model_id"], entry["name"]) for entry in await storage.list(None)
        } == {("a", "final"), ("b", "final")}
        assert (await storage.read_metadata(paths["a"]))["backend"] == "miles"
        await storage.delete(paths["a"])
        assert await storage.list("a") == []
        assert len(await storage.list("b")) == 1

    asyncio.run(check())


@pytest.mark.parametrize("loss_fn", ["cross_entropy", "importance_sampling"])
def test_mixed_clients_only_forward_fields_consumed_by_loss(tmp_path, loss_fn):
    from lilo.backends.miles_runtime.data import prepare_batch

    first = _datum([1, 2, 3], 4)
    second = _datum([1, 2, 3], 4)
    inputs = first.loss_fn_inputs
    extra = TensorData(
        data=[0.0] * len(inputs["target_tokens"].data),
        dtype="float32",
        shape=inputs["target_tokens"].shape,
    )
    inputs["advantages"] = extra
    inputs["logprobs"] = extra
    if loss_fn != "cross_entropy":
        second.loss_fn_inputs["advantages"] = extra
        second.loss_fn_inputs["logprobs"] = extra
        second.loss_fn_inputs.pop("weights")
    batch = ForwardBatch(
        items=(ForwardItem("a", (first,)), ForwardItem("b", (second,))),
        loss_fn=loss_fn,
        loss_fn_config={},
    )
    rows = [row for _, row in prepare_batch(batch, {"a": 0, "b": 1}).slot_rows]
    assert rows[0].keys() == rows[1].keys()
    assert ("weights" in rows[0]) == (loss_fn == "cross_entropy")
    assert ("sampling_logprobs" in rows[0]) == (loss_fn != "cross_entropy")


@pytest.mark.parametrize(
    "name", ["learning_rate", "beta1", "beta2", "eps", "weight_decay", "grad_clip_norm"]
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_adam_parameters_rejected_before_runtime(name, value):
    from lilo.backends.miles_lora import _adam_parameters
    from types import SimpleNamespace

    values = dict(
        learning_rate=1e-4,
        beta1=0.9,
        beta2=0.99,
        eps=1e-8,
        weight_decay=0.0,
        grad_clip_norm=1.0,
    )
    values[name] = value
    with pytest.raises(ValueError, match="finite"):
        _adam_parameters(SimpleNamespace(**values))


@pytest.mark.parametrize("outcome", [{"error": "worker update failed"}, {}])
def test_optimizer_worker_failure_is_fatal(tmp_path, outcome):
    from lilo.errors import BackendFailed

    runtime = FakeMilesRuntime()
    backend = _backend(tmp_path, runtime)
    backend.accept_model("a", _spec())
    backend.jobs["a"].accumulating = True
    runtime.optim_step = lambda parameters: {0: outcome}
    with pytest.raises(BackendFailed):
        backend.optim_step(("a",), AdamParams(learning_rate=1e-4))
