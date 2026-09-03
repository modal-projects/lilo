from pathlib import Path
from types import SimpleNamespace

from runtime_stubs import backend_runtime_imports
from tinker import (
    AdamParams,
    Datum,
    ForwardBackwardOutput,
    LoraConfig,
    ModelInput,
    OptimStepResponse,
    TensorData,
)

from lilo.backends import (
    ForwardBatch,
    ForwardItem,
    ModelSpec,
    SamplerPublication,
)

with backend_runtime_imports():
    from lilo.backends.megatron_fft import FFTMegatronBackend
    from lilo.backends.megatron_lora import LoraJobState, LoraMegatronBackend


def test_lora_job_state_does_not_restore_optimizer_by_default() -> None:
    state = LoraJobState(rank=8, alpha=16.0)

    assert state.load_optimizer is False


def test_lora_checkpoint_is_loaded_once_before_replacing_slot(monkeypatch) -> None:
    from lilo.backends import megatron_lora

    backend = LoraMegatronBackend.__new__(LoraMegatronBackend)
    backend.jobs = {"model-a": LoraJobState(rank=8, alpha=16.0)}
    backend.job_to_slot = {"model-a": 0}
    checkpoint = {"adapter_megatron": {"weight": object()}}
    reads = []
    loaded = []
    monkeypatch.setattr(
        megatron_lora,
        "load_training_checkpoint",
        lambda uri: reads.append(uri) or checkpoint,
    )
    monkeypatch.setattr(
        backend,
        "_offload_job_from_slot",
        lambda model_id: backend.job_to_slot.pop(model_id),
    )
    monkeypatch.setattr(
        backend,
        "_load_job_to_slot",
        lambda model_id, state: loaded.append((model_id, state)),
    )

    backend.load_checkpoint("model-a", "/checkpoint", restore_optimizer=True)

    assert reads == ["/checkpoint"]
    assert loaded == [("model-a", checkpoint)]
    assert backend.jobs["model-a"].load_optimizer is True


def test_lora_backend_implements_generic_contract(monkeypatch) -> None:
    backend = LoraMegatronBackend.__new__(LoraMegatronBackend)
    backend.base_model = "Qwen/Qwen3-4B"
    backend.config = SimpleNamespace(default_lora_alpha=32)
    backend.jobs = {}
    backend.job_to_slot = {}
    backend._sampler_captures = {}
    commands = []

    def register(model_id, **kwargs):
        commands.append(SimpleNamespace(model_id=model_id, **kwargs))
        backend.jobs[model_id] = LoraJobState(
            rank=kwargs["rank"],
            alpha=kwargs["alpha"],
            seed=kwargs["seed"],
            train_attn=kwargs["train_attn"],
            train_mlp=kwargs["train_mlp"],
            train_unembed=kwargs["train_unembed"],
        )

    def load(model_id):
        commands.append(SimpleNamespace(model_id=model_id))
        backend.job_to_slot[model_id] = 0
        return 0

    def run_forward_backward(batch):
        commands.append(batch)
        return tuple(
            ForwardBackwardOutput(
                "MegatronSFTLoss",
                [
                    {
                        "logprobs": TensorData(
                            data=[-1.0],
                            dtype="float32",
                        )
                    }
                ],
                {"loss:mean": 1.0},
            )
            for _item in batch.items
        )

    def run_optim_step(model_ids, adam):
        commands.append((model_ids, adam))
        return tuple(
            OptimStepResponse(metrics={"grad_norm:mean": 0.5})
            for _model_id in model_ids
        )

    def capture(model_id, capture_id, publish_version):
        commands.append(("capture_sampler_snapshot", model_id, capture_id))
        backend._sampler_captures[capture_id] = "snapshot"
        return SamplerPublication(publish_version, backend.base_model, 1)

    def offload(model_id):
        backend.job_to_slot.pop(model_id, None)
        return backend.jobs[model_id]

    persisted = []
    monkeypatch.setattr(backend, "_register_job", register)
    monkeypatch.setattr(backend, "_load_job_to_slot", load)
    monkeypatch.setattr(backend, "_forward_backward_batch", run_forward_backward)
    monkeypatch.setattr(backend, "_optim_step_batch", run_optim_step)
    monkeypatch.setattr(backend, "capture_sampler_snapshot", capture)
    monkeypatch.setattr(backend, "_offload_job_from_slot", offload)
    from lilo.backends import megatron_lora

    monkeypatch.setattr(
        megatron_lora,
        "persist_adapter_snapshot",
        lambda snapshot: persisted.append(snapshot),
    )

    spec = ModelSpec(
        base_model=backend.base_model,
        parameterization="lora",
        lora_config=LoraConfig(rank=16, seed=7),
    )
    backend.accept_model("model-a", spec)
    assert commands[0].rank == 16
    assert commands[0].alpha == 32
    backend.accept_model("model-a", spec)

    datum = Datum(
        ModelInput.from_ints([1, 2, 3]),
        {
            "target_tokens": TensorData(data=[2, 3, 4], dtype="int64"),
            "weights": TensorData(data=[1.0, 1.0, 1.0], dtype="float32"),
        },
    )
    output = backend.forward_backward(
        ForwardBatch(
            items=(ForwardItem("model-a", (datum,)),),
            loss_fn="cross_entropy",
        )
    )
    assert output[0].loss_fn_output_type == "MegatronSFTLoss"
    assert commands[-1].items[0].data[0].model_input.to_ints() == [1, 2, 3]

    optim = backend.optim_step(
        ("model-a",),
        AdamParams(learning_rate=1e-4),
    )
    assert optim[0].metrics == {"grad_norm:mean": 0.5}
    publication = backend.capture_sampler_snapshot("model-a", "capture-a", 4)
    assert publication.publish_version == 4
    backend.persist_sampler_snapshot("capture-a")
    assert persisted == ["snapshot"]
    backend.unload_model("model-a")
    assert "model-a" not in backend.jobs


def test_fft_backend_implements_generic_contract(monkeypatch) -> None:
    backend = FFTMegatronBackend.__new__(FFTMegatronBackend)
    backend.base_model = "Qwen/Qwen3.5-9B-Base"
    backend.model_id = None
    backend.accumulating = False
    backend.optimizer_step = 0
    backend._reset_before_accept = False
    backend._delta_writer = None
    backend._sampler_captures = {}
    backend._checkpoint_captures = {}
    backend.model = [SimpleNamespace(zero_grad_buffer=lambda: None)]
    backend.optimizer = SimpleNamespace(zero_grad=lambda: None)

    calls = []

    def capture(model_id, capture_id, requested_version):
        calls.append(("capture", model_id, capture_id))
        backend._sampler_captures[capture_id] = None
        return SamplerPublication(2, backend.base_model, 2)

    monkeypatch.setattr(backend, "capture_sampler_snapshot", capture)
    backend.accept_model(
        "model-a",
        ModelSpec(
            base_model=backend.base_model,
            parameterization="full",
        ),
    )
    publication = backend.capture_sampler_snapshot("model-a", "capture-a", 11)
    assert publication.publish_version == 2
    backend.persist_sampler_snapshot("capture-a")
    assert "capture-a" not in backend._sampler_captures
    backend.unload_model("model-a")
    assert calls == [("capture", "model-a", "capture-a")]
    assert backend.model_id is None


def test_lora_build_executor_constructs_backend_directly(monkeypatch) -> None:
    from lilo.backends import megatron_lora

    backend_config = object()
    checkpoint_dir = Path("/checkpoints")
    captured = {}

    def construct(config, **kwargs):
        captured.update(config=config, **kwargs)
        return "backend"

    monkeypatch.setenv("LILO_BACKEND_CONFIG", '{"megatron": {}}')
    monkeypatch.setenv("LILO_BASE_MODEL", "Qwen/Qwen3-4B")
    monkeypatch.setattr(
        megatron_lora,
        "initialize_distributed_runtime",
        lambda: ("a", "b", "c"),
    )
    monkeypatch.setattr(
        megatron_lora,
        "parse_backend_config",
        lambda config: (backend_config, checkpoint_dir),
    )
    monkeypatch.setattr(megatron_lora, "LoraMegatronBackend", construct)

    result = megatron_lora.build_executor()

    assert result.backend == "backend"
    assert result.command_group == "a"
    assert result.checkpoint_persistence_group == "b"
    assert result.sampler_persistence_group == "c"
    assert captured == {
        "config": backend_config,
        "checkpoint_dir": checkpoint_dir,
        "base_model": "Qwen/Qwen3-4B",
        "persistence_group": "b",
    }


def test_fft_build_executor_constructs_backend_directly(monkeypatch) -> None:
    from lilo.backends import megatron_fft

    backend_config = object()
    captured = {}

    def construct(config, **kwargs):
        captured.update(config=config, **kwargs)
        return "backend"

    monkeypatch.setenv("LILO_BACKEND_CONFIG", '{"megatron": {}}')
    monkeypatch.setenv("LILO_BASE_MODEL", "Qwen/Qwen3.5-9B-Base")
    monkeypatch.setattr(
        megatron_fft,
        "initialize_distributed_runtime",
        lambda: ("a", "b", "c"),
    )
    monkeypatch.setattr(
        megatron_fft,
        "parse_backend_config",
        lambda config: (backend_config, Path("/unused")),
    )
    monkeypatch.setattr(megatron_fft, "FFTMegatronBackend", construct)

    result = megatron_fft.build_executor()

    assert result.backend == "backend"
    assert result.command_group == "a"
    assert result.checkpoint_persistence_group == "b"
    assert result.sampler_persistence_group == "c"
    assert captured == {
        "config": backend_config,
        "checkpoint_dir": Path("/unused"),
        "base_model": "Qwen/Qwen3.5-9B-Base",
        "persistence_group": "b",
    }
