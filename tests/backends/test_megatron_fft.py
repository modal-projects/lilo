from __future__ import annotations

import json
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest
from runtime_stubs import backend_runtime_imports
from stitch.types import VersionRef

from lilo.backends import ModelSpec, SamplerPublication

with backend_runtime_imports():
    from lilo.backends import megatron_fft as fft_backend
    from lilo.backends.megatron_fft import FFTMegatronBackend
    from lilo.backends.megatron_runtime.common.config import EngineModelConfig
    from lilo.backends.megatron_runtime.common.forward_backward import (
        add_packing_metrics,
    )
    from lilo.backends.megatron_runtime.fft import checkpoint as fft_checkpoint

BASE_MODEL = "Qwen/Qwen3.5-9B-Base"
DEFINITION_ID = "qwen3_5_9b_base_full_32k"
FULL_MODEL_SPEC = ModelSpec(base_model=BASE_MODEL, parameterization="full")


def test_packing_sum_metrics_are_split_across_outputs() -> None:
    outputs = (SimpleNamespace(metrics={}), SimpleNamespace(metrics={}))

    add_packing_metrics(
        outputs,
        {
            "packing_raw_tokens:sum": 12.0,
            "packing_utilization:mean": 0.75,
        },
    )

    assert [output.metrics for output in outputs] == [
        {
            "packing_raw_tokens:sum": 6.0,
            "packing_utilization:mean": 0.75,
        },
        {
            "packing_raw_tokens:sum": 6.0,
            "packing_utilization:mean": 0.75,
        },
    ]


def backend_state(model_id: str | None = None) -> FFTMegatronBackend:
    backend = FFTMegatronBackend.__new__(FFTMegatronBackend)
    backend.base_model = BASE_MODEL
    backend.model_id = model_id
    backend.world_size = 1
    backend.persistence_group = "persistence"
    backend.accumulating = False
    backend.optimizer_step = 0
    backend._reset_before_accept = False
    backend._delta_writer = None
    backend._sampler_captures = {}
    backend._checkpoint_captures = {}
    backend.config = EngineModelConfig(hf_checkpoint="/model")
    backend.model = [SimpleNamespace(zero_grad_buffer=lambda: None)]
    backend.optimizer = SimpleNamespace(zero_grad=lambda: None)
    return backend


def single_rank_preflight(error, **_kwargs) -> None:
    if error is not None:
        raise error


def publish_sampler_snapshot(
    backend: FFTMegatronBackend,
    model_id: str,
    requested_version: int | None = None,
) -> SamplerPublication:
    capture_id = "capture"
    publication = backend.capture_sampler_snapshot(
        model_id,
        capture_id,
        requested_version or 1,
    )
    backend.persist_sampler_snapshot(capture_id)
    return publication


def test_full_backend_resets_before_reusing_a_warm_worker() -> None:
    backend = backend_state()
    resets = []
    backend._reset_model = lambda: resets.append(True)

    backend.accept_model("first", FULL_MODEL_SPEC)
    backend.unload_model("first")
    backend.accept_model("second", FULL_MODEL_SPEC)

    assert resets == [True]
    assert backend.model_id == "second"
    assert backend.optimizer_step == 0


def test_full_backend_hosts_only_one_model() -> None:
    backend = backend_state("first")

    with pytest.raises(ValueError, match="already hosts model first"):
        backend.accept_model("second", FULL_MODEL_SPEC)


def test_full_backend_dummy_dp_rank_contributes_zero_tokens(monkeypatch) -> None:
    backend = backend_state("model")
    backend.rank = 1
    backend.config = SimpleNamespace(
        seq_length=8,
        packed_token_capacity=lambda: 8,
        sequence_padding_multiple=lambda: 1,
        defer_fp32_logits=False,
    )
    backend._zero_grad = lambda: None
    tensor = SimpleNamespace(numel=lambda: 4)
    output = SimpleNamespace(metrics={})
    monkeypatch.setattr(
        fft_backend,
        "prepare_microbatches",
        lambda *args, **kwargs: ([{"tokens": tensor}], {}),
    )
    monkeypatch.setattr(
        fft_backend,
        "run_megatron_pipeline",
        lambda *args, **kwargs: ({}, {}),
    )
    monkeypatch.setattr(
        fft_backend,
        "synchronize_collectors",
        lambda outputs, metrics: ({}, {0: {"tokens": 4.0}}),
    )
    monkeypatch.setattr(
        fft_backend,
        "build_outputs",
        lambda *args, **kwargs: (output, output),
    )

    result = backend.forward_backward(
        SimpleNamespace(
            items=(SimpleNamespace(model_id="model"),) * 2,
            forward_only=False,
        )
    )

    assert result == (output, output)
    assert backend.accumulating is True


def test_full_backend_resets_weights_in_place(monkeypatch) -> None:
    backend = backend_state()
    loaded = []
    replacement_optimizer = SimpleNamespace(zero_grad=lambda: None)
    backend.bridge = SimpleNamespace(
        load_hf_weights=lambda model: loaded.append(model),
    )
    monkeypatch.setattr(
        fft_backend,
        "create_fft_optimizer",
        lambda config, model: replacement_optimizer,
    )
    monkeypatch.setattr(
        fft_backend,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
    )

    backend._reset_model()

    assert loaded == [backend.model]
    assert backend.optimizer is replacement_optimizer
    assert backend._reset_before_accept is False


@pytest.mark.parametrize(("rank", "expected"), [(0, {"weight": "tensor"}), (1, None)])
def test_capture_hf_weights_keeps_only_rank_zero(rank, expected, monkeypatch) -> None:
    distributed = ModuleType("torch.distributed")
    distributed.is_initialized = lambda: True
    distributed.get_rank = lambda: rank
    monkeypatch.setattr(fft_checkpoint, "dist", distributed)
    calls = []

    class Bridge:
        def export_hf_weights(self, model, **kwargs):
            calls.append((model, kwargs))
            yield SimpleNamespace(param_name="weight", weight="tensor")

    assert fft_checkpoint.capture_hf_weights(Bridge(), "model") == expected
    assert calls == [
        (
            "model",
            {
                "cpu": rank == 0,
                "show_progress": False,
                "merge_adapter_weights": True,
            },
        )
    ]


def test_write_fft_checkpoint_includes_hf_weights(tmp_path, monkeypatch) -> None:
    writes = []
    distributed = ModuleType("torch.distributed")
    distributed.barrier = lambda: None
    distributed.get_rank = lambda: 0
    torch = ModuleType("torch")
    torch.save = lambda payload, path: writes.append(("native", payload, path))
    monkeypatch.setattr(fft_checkpoint, "torch", torch)
    monkeypatch.setattr(fft_checkpoint, "dist", distributed)
    monkeypatch.setattr(
        fft_checkpoint,
        "save_torch_state_dict",
        lambda state, path, **kwargs: writes.append(("hf", state, path, kwargs)),
    )
    output = tmp_path / "output"
    monkeypatch.setattr(
        fft_checkpoint,
        "fft_checkpoint_path",
        lambda uri: output / "checkpoint_rank.pt",
    )
    monkeypatch.setattr(
        fft_checkpoint,
        "fft_checkpoint_metadata_path",
        lambda uri: output / fft_checkpoint.CHECKPOINT_METADATA_FILENAME,
    )
    commits = []
    monkeypatch.setattr(
        fft_checkpoint,
        "commit_checkpoint_volume",
        lambda group: commits.append(group),
    )
    source = tmp_path / "base"
    source.mkdir()
    (source / "config.json").write_text("config", encoding="utf-8")
    metadata = fft_checkpoint.create_fft_checkpoint_metadata(
        EngineModelConfig(hf_checkpoint=str(source)),
        checkpoint_id="snapshot-4",
        base_model=BASE_MODEL,
        include_optimizer=True,
        world_size=1,
    )
    payload = {"metadata": metadata.to_dict(), "model": "state"}

    result = fft_checkpoint.write_fft_checkpoint(
        "/checkpoint",
        payload,
        hf_weights={"model.weight": "weight"},
        hf_checkpoint=str(source),
        persistence_group="persistence",
    )

    assert result == "/checkpoint"
    assert writes == [
        ("native", payload, output / "checkpoint_rank.pt"),
        (
            "hf",
            {"model.weight": "weight"},
            output,
            {"safe_serialization": True},
        ),
    ]
    assert (output / "config.json").read_text(encoding="utf-8") == "config"
    assert (
        json.loads(
            (output / fft_checkpoint.CHECKPOINT_METADATA_FILENAME).read_text(
                encoding="utf-8"
            )
        )
        == metadata.to_dict()
    )
    assert commits == ["persistence"]


def test_fft_backend_captures_and_persists_checkpoint(tmp_path, monkeypatch) -> None:
    backend = backend_state("run-a")
    backend.checkpoint_dir = tmp_path
    backend.bridge = "bridge"
    backend.config = EngineModelConfig(hf_checkpoint="/base-checkpoint")
    backend.model = "model"
    backend.optimizer = "optimizer"
    backend.optimizer_step = 4
    backend.accumulating = True
    monkeypatch.setenv("LILO_DEFINITION_ID", DEFINITION_ID)
    writes = []
    monkeypatch.setattr(
        fft_backend,
        "capture_hf_weights",
        lambda bridge, model: {"model.weight": "weight"},
    )
    monkeypatch.setattr(
        fft_backend,
        "capture_fft_checkpoint",
        lambda model, optimizer, **kwargs: {"captured": (model, optimizer, kwargs)},
    )
    monkeypatch.setattr(
        fft_backend,
        "write_fft_checkpoint",
        lambda path, checkpoint, **kwargs: (
            writes.append((path, checkpoint, kwargs)) or path
        ),
    )
    monkeypatch.setattr(
        fft_backend,
        "fft_checkpoint_path",
        lambda path: tmp_path / "missing-checkpoint.pt",
    )

    backend.capture_checkpoint(
        "run-a",
        "snapshot",
        destination="step-4",
        include_optimizer=True,
    )
    result = backend.persist_checkpoint("snapshot", "step-4")

    assert result == str(tmp_path / "run-a" / "weights" / "step-4")
    assert writes[0][0] == result
    assert writes[0][1]["optimizer_step"] == 4
    captured_metadata = writes[0][1]["captured"][2]["metadata"]
    assert captured_metadata.checkpoint_id == "snapshot"
    assert captured_metadata.base_model == BASE_MODEL
    assert captured_metadata.has_optimizer is True
    assert writes[0][2] == {
        "hf_weights": {"model.weight": "weight"},
        "hf_checkpoint": "/base-checkpoint",
        "metadata": {
            "schema_version": 1,
            "base_model": backend.base_model,
            "engine_definition_id": DEFINITION_ID,
            "parameterization": {"type": "full"},
            "lora_config": None,
        },
        "persistence_group": "persistence",
    }
    assert backend._checkpoint_captures == {}


def test_distributed_optimizer_capture_includes_detached_parameter_state(
    monkeypatch,
) -> None:
    class Tensor:
        def __init__(self, values):
            self.values = list(values)

        def detach(self):
            return self

        def cpu(self):
            return self

        def clone(self):
            return Tensor(self.values)

        def tolist(self):
            return list(self.values)

        def data_ptr(self):
            return id(self)

        def fill_(self, value):
            self.values[:] = [value] * len(self.values)

    monkeypatch.setattr(
        fft_checkpoint,
        "torch",
        SimpleNamespace(is_tensor=lambda value: isinstance(value, Tensor)),
    )
    master = Tensor([1.25])
    exp_avg = Tensor([2.5])
    exp_avg_sq = Tensor([3.75])
    calls = []

    class Optimizer:
        def state_dict(self):
            raise AssertionError("distributed capture must not use plain state_dict")

        def sharded_state_dict(self, **kwargs):
            calls.append(kwargs)
            return {
                "optimizer": fft_checkpoint.ShardedObject(
                    {"param_groups": [{"step": 7}]}
                ),
                "param_state": {
                    0: {
                        "float32": [
                            [
                                {
                                    "param": fft_checkpoint.ShardedTensor(master),
                                    "exp_avg": fft_checkpoint.ShardedTensor(exp_avg),
                                    "exp_avg_sq": fft_checkpoint.ShardedTensor(
                                        exp_avg_sq
                                    ),
                                    "padding": (
                                        fft_checkpoint.LocalNonpersistentObject(False)
                                    ),
                                }
                            ]
                        ]
                    }
                },
                "param_state_sharding_type": "dp_reshardable",
            }

    captured = fft_checkpoint.capture_fft_optimizer_state(
        Optimizer(),
        use_distributed_optimizer=True,
    )

    assert calls == [
        {
            "model_sharded_state_dict": {},
            "is_loading": False,
            "metadata": {"distrib_optim_sharding_type": "dp_reshardable"},
        }
    ]
    assert captured["format"] == (
        fft_checkpoint.DISTRIBUTED_OPTIMIZER_STATE_FORMAT
    )
    tensors = captured["state_dict"]["param_state"][0]["float32"][0][0]
    assert tensors["param"].tolist() == [1.25]
    assert tensors["exp_avg"].tolist() == [2.5]
    assert tensors["exp_avg_sq"].tolist() == [3.75]
    assert tensors["padding"] is False
    assert tensors["param"].data_ptr() != master.data_ptr()
    master.fill_(0)
    exp_avg.fill_(0)
    exp_avg_sq.fill_(0)
    assert tensors["param"].tolist() == [1.25]
    assert tensors["exp_avg"].tolist() == [2.5]
    assert tensors["exp_avg_sq"].tolist() == [3.75]


def test_distributed_optimizer_restore_loads_parameter_state() -> None:
    restored = []
    state = {
        "optimizer": {"param_groups": [{"step": 7}]},
        "param_state": {"exp_avg": [2.5]},
        "param_state_sharding_type": "dp_reshardable",
    }
    optimizer = SimpleNamespace(
        load_state_dict=lambda loaded: restored.append(loaded),
    )

    fft_checkpoint.restore_fft_optimizer_state(
        optimizer,
        {
            "format": fft_checkpoint.DISTRIBUTED_OPTIMIZER_STATE_FORMAT,
            "state_dict": state,
        },
        use_distributed_optimizer=True,
    )

    assert restored == [state]


def test_distributed_optimizer_restore_accepts_chained_state() -> None:
    restored = []

    def state(step: int) -> dict:
        return {
            "optimizer": {"param_groups": [{"step": step}]},
            "param_state": {"exp_avg": [float(step)]},
            "param_state_sharding_type": "dp_reshardable",
        }

    chained = {0: state(7), 1: state(8)}
    optimizer = SimpleNamespace(
        load_state_dict=lambda loaded: restored.append(loaded),
    )

    fft_checkpoint.restore_fft_optimizer_state(
        optimizer,
        {
            "format": fft_checkpoint.DISTRIBUTED_OPTIMIZER_STATE_FORMAT,
            "state_dict": chained,
        },
        use_distributed_optimizer=True,
    )

    assert restored == [chained]


def test_fft_backend_native_resume_never_invokes_hf(monkeypatch) -> None:
    backend = backend_state("run-a")
    restored = []
    checkpoint_reads = []
    backend.model = [
        SimpleNamespace(
            load_state_dict=lambda state, **kwargs: restored.append(("model", state))
        )
    ]
    backend.optimizer = SimpleNamespace(
        reload_model_params=lambda: restored.append("reload"),
        load_state_dict=lambda state: restored.append(("optimizer", state)),
        zero_grad=lambda: None,
    )
    backend._zero_grad = lambda: restored.append("zero")
    backend._delta_writer = object()
    checkpoint = {
        "model": ["model-state"],
        "optimizer": {
            "format": fft_checkpoint.REGULAR_OPTIMIZER_STATE_FORMAT,
            "state_dict": {"step": 4},
        },
        "optimizer_step": 4,
    }
    metadata = SimpleNamespace(identity=lambda: "snapshot:identity")
    monkeypatch.setattr(
        fft_backend,
        "reload_checkpoint_volume",
        lambda: checkpoint_reads.append("reload"),
    )

    def load_checkpoint(uri, *args, **kwargs):
        checkpoint_reads.append(("load", uri))
        return checkpoint, metadata

    monkeypatch.setattr(
        fft_backend,
        "load_fft_training_checkpoint",
        load_checkpoint,
    )
    monkeypatch.setattr(
        fft_backend,
        "synchronize_checkpoint_preflight",
        single_rank_preflight,
    )

    backend.load_checkpoint(
        "run-a",
        "/checkpoint",
        restore_optimizer=True,
    )

    assert checkpoint_reads == ["reload", ("load", "/checkpoint")]
    assert restored == [
        ("model", "model-state"),
        "reload",
        ("optimizer", {"step": 4}),
        "zero",
    ]
    assert backend.optimizer_step == 4
    assert backend.accumulating is False
    assert backend._delta_writer is None


def test_fft_backend_hf_load_never_reads_native(monkeypatch) -> None:
    backend = backend_state("run-a")
    backend.optimizer = "old-optimizer"
    backend.optimizer_step = 9
    backend.accumulating = True
    replacement = SimpleNamespace(zero_grad=lambda: None)
    restored = []
    bridge = SimpleNamespace(
        load_hf_weights=lambda model: restored.append(("hf", model)),
    )
    monkeypatch.setattr(
        fft_backend,
        "AutoBridge",
        SimpleNamespace(
            from_hf_pretrained=lambda uri, **kwargs: (
                restored.append(("preflight", uri, kwargs)) or bridge
            )
        ),
    )
    monkeypatch.setattr(
        fft_backend,
        "load_fft_training_checkpoint",
        lambda uri: pytest.fail("weights-only load read native metadata"),
    )
    monkeypatch.setattr(
        fft_backend,
        "create_fft_optimizer",
        lambda config, model: replacement,
    )
    monkeypatch.setattr(fft_backend, "reload_checkpoint_volume", lambda: None)
    monkeypatch.setattr(
        fft_backend,
        "synchronize_checkpoint_preflight",
        single_rank_preflight,
    )
    monkeypatch.setattr(
        fft_backend,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
    )

    backend.bridge = base_bridge = object()
    backend.load_checkpoint("run-a", "/checkpoint")

    assert restored == [
        (
            "preflight",
            "/checkpoint",
            {"trust_remote_code": False, "local_files_only": True},
        ),
        ("hf", backend.model),
    ]
    assert backend.bridge is base_bridge
    assert backend.optimizer is replacement
    assert backend.optimizer_step == 0
    assert backend.accumulating is False


@pytest.mark.parametrize(
    ("metadata_change", "message"),
    [
        (
            lambda metadata: replace(metadata, tensor_model_parallel_size=2),
            "tensor_model_parallel_size",
        ),
        (
            lambda metadata: replace(
                metadata,
                optimizer_config={**metadata.optimizer_config, "lr": 0.5},
            ),
            "optimizer_config",
        ),
    ],
)
def test_native_metadata_mismatch_prevents_model_mutation(
    tmp_path,
    metadata_change,
    message,
    monkeypatch,
) -> None:
    backend = backend_state("run-a")
    metadata = fft_checkpoint.create_fft_checkpoint_metadata(
        backend.config,
        checkpoint_id="snapshot",
        base_model=BASE_MODEL,
        include_optimizer=True,
        world_size=1,
    )
    (tmp_path / fft_checkpoint.CHECKPOINT_METADATA_FILENAME).write_text(
        json.dumps(metadata_change(metadata).to_dict()),
        encoding="utf-8",
    )
    mutations = []
    backend.optimizer = SimpleNamespace(
        reload_model_params=lambda: mutations.append("reload"),
        load_state_dict=lambda state: mutations.append("optimizer"),
        zero_grad=lambda: mutations.append("zero"),
    )
    monkeypatch.setattr(fft_backend, "reload_checkpoint_volume", lambda: None)
    monkeypatch.setattr(
        fft_backend,
        "synchronize_checkpoint_preflight",
        single_rank_preflight,
    )
    with pytest.raises(ValueError, match=message):
        backend.load_checkpoint(
            "run-a",
            str(tmp_path),
            restore_optimizer=True,
        )

    assert mutations == []


def test_native_resume_requires_global_metadata(tmp_path, monkeypatch) -> None:
    backend = backend_state("run-a")
    monkeypatch.setattr(fft_backend, "reload_checkpoint_volume", lambda: None)
    monkeypatch.setattr(
        fft_backend,
        "synchronize_checkpoint_preflight",
        single_rank_preflight,
    )

    with pytest.raises(
        FileNotFoundError,
        match=fft_checkpoint.CHECKPOINT_METADATA_FILENAME,
    ):
        backend.load_checkpoint(
            "run-a",
            str(tmp_path),
            restore_optimizer=True,
        )


def test_native_metadata_rejects_optimizer_state_absent(tmp_path) -> None:
    config = EngineModelConfig(hf_checkpoint="/model")
    metadata = fft_checkpoint.create_fft_checkpoint_metadata(
        config,
        checkpoint_id="snapshot",
        base_model=BASE_MODEL,
        include_optimizer=False,
        world_size=1,
    )
    (tmp_path / fft_checkpoint.CHECKPOINT_METADATA_FILENAME).write_text(
        json.dumps(metadata.to_dict()),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no optimizer state"):
        fft_checkpoint.load_fft_training_checkpoint(
            str(tmp_path),
            config,
            base_model=BASE_MODEL,
            world_size=1,
        )


def test_native_resume_rejects_incomplete_distributed_optimizer_state(
    tmp_path,
    monkeypatch,
) -> None:
    config = EngineModelConfig(
        hf_checkpoint="/model",
        use_distributed_optimizer=True,
    )
    metadata = fft_checkpoint.create_fft_checkpoint_metadata(
        config,
        checkpoint_id="snapshot",
        base_model=BASE_MODEL,
        include_optimizer=True,
        world_size=1,
    )
    (tmp_path / fft_checkpoint.CHECKPOINT_METADATA_FILENAME).write_text(
        json.dumps(metadata.to_dict()),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        fft_checkpoint.torch,
        "load",
        lambda *args, **kwargs: {
            "metadata": metadata.to_dict(),
            "model": [],
            "optimizer": {
                "format": fft_checkpoint.DISTRIBUTED_OPTIMIZER_STATE_FORMAT,
                "state_dict": {"optimizer": {"param_groups": [{"step": 4}]}},
            },
        },
        raising=False,
    )
    monkeypatch.setattr(
        fft_checkpoint,
        "fft_checkpoint_path",
        lambda uri: tmp_path / "checkpoint_rank.pt",
    )

    with pytest.raises(ValueError, match="dp_reshardable parameter state"):
        fft_checkpoint.load_fft_training_checkpoint(
            str(tmp_path),
            config,
            base_model=BASE_MODEL,
            world_size=1,
        )


def test_preflight_synchronizes_remote_rank_error(monkeypatch) -> None:
    distributed = ModuleType("torch.distributed")
    distributed.get_world_size = lambda: 2
    distributed.all_gather_object = lambda gathered, local: gathered.__setitem__(
        slice(None),
        [
            local,
            {
                "error": "FileNotFoundError: missing shard",
                "metadata_identity": None,
            },
        ],
    )
    monkeypatch.setattr(fft_checkpoint, "dist", distributed)

    with pytest.raises(
        ValueError,
        match="rank 1: FileNotFoundError: missing shard",
    ):
        fft_checkpoint.synchronize_checkpoint_preflight(None)


def test_preflight_synchronizes_metadata_identity(monkeypatch) -> None:
    distributed = ModuleType("torch.distributed")
    distributed.get_world_size = lambda: 2
    distributed.all_gather_object = lambda gathered, local: gathered.__setitem__(
        slice(None),
        [
            local,
            {"error": None, "metadata_identity": "other"},
        ],
    )
    monkeypatch.setattr(fft_checkpoint, "dist", distributed)

    with pytest.raises(ValueError, match="identity differs across ranks"):
        fft_checkpoint.synchronize_checkpoint_preflight(
            None,
            metadata_identity="local",
        )


def test_full_backend_publishes_initial_policy_as_root_delta(
    tmp_path, monkeypatch
) -> None:
    backend = backend_state("run-a")
    backend.bridge = "bridge"
    backend.model = "model"
    publications = []
    backend._delta_writer = SimpleNamespace(
        capture=lambda **kwargs: (
            publications.append(kwargs)
            or SimpleNamespace(ref=VersionRef("run-a", kwargs["publish_version"]))
        ),
        persist=lambda snapshot, **kwargs: None,
    )
    monkeypatch.setenv("LILO_BULLETIN_ROOT", str(tmp_path))
    monkeypatch.setenv("LILO_BULLETIN_VOLUME", "test-bulletin")

    result = publish_sampler_snapshot(backend, "run-a")

    assert result.publish_version == 1
    assert result.optimizer_step == 0
    assert publications[0]["publish_version"] == 1
    assert publications[0]["optimizer_step"] == 0


def test_full_backend_publishes_after_loading_an_older_step(monkeypatch) -> None:
    backend = backend_state("run-a")
    backend.optimizer_step = 4
    backend.bridge = "bridge"
    backend.model = "model"
    captures = []

    class Bulletin:
        def read_latest(self, run_id):
            return VersionRef(run_id, 3)

        def metadata(self, ref):
            return {"optimizer_step": 5}

    monkeypatch.setattr(fft_backend, "FFTSnapshotBulletin", lambda root: Bulletin())
    monkeypatch.setenv("LILO_BULLETIN_ROOT", "/bulletin")
    backend._delta_writer = SimpleNamespace(
        is_aligned_with=lambda ref: False,
        capture=lambda **kwargs: (
            captures.append(kwargs)
            or SimpleNamespace(ref=VersionRef("run-a", kwargs["publish_version"]))
        ),
    )

    publication = backend.capture_sampler_snapshot("run-a", "capture", 99)

    assert publication.publish_version == 4
    assert publication.optimizer_step == 4
    assert captures[0]["publish_version"] == 4


def test_fft_sampler_failure_discards_pending_capture(monkeypatch) -> None:
    backend = backend_state("run-a")
    backend._sampler_captures["capture"] = object()
    monkeypatch.setenv("LILO_BULLETIN_ROOT", "/bulletin")
    monkeypatch.setenv("LILO_BULLETIN_VOLUME", "test-bulletin")

    def fail(*args, **kwargs):
        raise RuntimeError("persist failed")

    backend._delta_writer = SimpleNamespace(persist=fail)

    with pytest.raises(RuntimeError, match="persist failed"):
        backend.persist_sampler_snapshot("capture")

    assert backend._sampler_captures == {}
