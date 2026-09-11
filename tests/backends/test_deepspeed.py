import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from runtime_stubs import backend_runtime_imports
from tinker import (
    AdamParams,
    Datum,
    ForwardBackwardOutput,
    ModelInput,
    OptimStepResponse,
    TensorData,
)

from lilo.backends import ForwardBatch, ForwardItem, ModelSpec
from lilo.backends.deepspeed_config import (
    DeepSpeedBackendConfig,
    parse_deepspeed_backend_config,
)


@contextmanager
def deepspeed_runtime_imports():
    deepspeed = ModuleType("deepspeed")
    deepspeed.initialize = MagicMock()
    transformers = ModuleType("transformers")
    transformers.AutoModelForCausalLM = MagicMock()
    transformers.AutoModelForImageTextToText = MagicMock()
    with (
        backend_runtime_imports(),
        patch.dict(
            sys.modules,
            {
                "deepspeed": deepspeed,
                "transformers": transformers,
            },
        ),
    ):
        yield


with deepspeed_runtime_imports():
    from lilo.backends import deepspeed_full
    from lilo.backends.deepspeed_full import DeepSpeedFullBackend
    from lilo.backends.deepspeed_runtime.training import (
        _pad_token_id,
        apply_adam_params,
    )


def test_deepspeed_config_uses_unmanaged_zero_two() -> None:
    config, checkpoint_dir = parse_deepspeed_backend_config(
        {
            "deepspeed": {
                "hf_checkpoint": "/model",
                "zero_stage": 2,
                "micro_batch_size": 2,
                "max_sequence_length": 4096,
            },
            "checkpoint_dir": "/state",
        }
    )

    assert checkpoint_dir == Path("/state")
    assert config.hf_checkpoint == "/model"
    assert config.engine_config(4) == {
        "train_micro_batch_size_per_gpu": 2,
        "train_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "managed_gradient_accumulation": False,
        "bf16": {"enabled": True},
        "zero_allow_untested_optimizer": True,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
        },
    }


def test_minimal_backend_rejects_zero_three() -> None:
    with pytest.raises(ValueError, match="ZeRO stages 1 and 2"):
        DeepSpeedBackendConfig(hf_checkpoint="/model", zero_stage=3)


def test_deepspeed_config_rejects_unknown_auto_model_class() -> None:
    with pytest.raises(ValueError, match="unsupported auto_model_class"):
        DeepSpeedBackendConfig(
            hf_checkpoint="/model",
            auto_model_class="unknown",
        )


def test_multimodal_config_uses_text_pad_token() -> None:
    config = SimpleNamespace(
        text_config=SimpleNamespace(pad_token_id=None, eos_token_id=248044)
    )

    assert _pad_token_id(config) == 248044


def test_adam_params_do_not_replace_engine_gradient_clipping_method() -> None:
    groups = [{"lr": 0.0, "betas": (0.0, 0.0), "eps": 0.0, "weight_decay": 0.0}]
    zero = SimpleNamespace(
        optimizer=SimpleNamespace(param_groups=groups),
        clip_grad=0.0,
    )
    engine = SimpleNamespace(
        optimizer=zero,
        gradient_clipping=lambda: 1.0,
    )

    apply_adam_params(
        engine,
        AdamParams(
            learning_rate=1e-5,
            beta1=0.8,
            beta2=0.9,
            eps=1e-7,
            weight_decay=0.1,
            grad_clip_norm=2.0,
        ),
    )

    assert engine.gradient_clipping() == 1.0
    assert zero.clip_grad == 2.0
    assert groups == [
        {
            "lr": 1e-5,
            "betas": (0.8, 0.9),
            "eps": 1e-7,
            "weight_decay": 0.1,
        }
    ]


def test_deepspeed_backend_implements_training_contract(monkeypatch) -> None:
    backend = DeepSpeedFullBackend.__new__(DeepSpeedFullBackend)
    backend.base_model = "Qwen/Qwen3-4B"
    backend.model_id = None
    backend.accumulating = False
    backend.optimizer_step = 0
    backend._reset_before_accept = False
    backend._delta_writer = None
    backend._checkpoint_captures = {}
    backend._sampler_captures = {}
    backend.rank = 0
    backend.world_size = 2
    backend.config = SimpleNamespace(max_sequence_length=2048)
    calls = []
    backend.engine = SimpleNamespace(
        zero_grad=lambda: calls.append("zero_grad"),
        step=lambda: calls.append("step"),
    )
    output = ForwardBackwardOutput(
        "DeepSpeedSFTLoss",
        [
            {
                "logprobs": TensorData(
                    data=[-1.0, -2.0],
                    dtype="float32",
                    shape=[2],
                )
            }
        ],
        {"loss:mean": 1.5},
    )
    monkeypatch.setattr(
        deepspeed_full,
        "run_forward_backward",
        lambda engine, batch, **kwargs: (calls.append((batch, kwargs)) or (output,)),
    )
    monkeypatch.setattr(
        deepspeed_full,
        "apply_adam_params",
        lambda engine, adam: calls.append(adam),
    )
    monkeypatch.setattr(
        deepspeed_full,
        "optimizer_grad_norm",
        lambda engine: 0.5,
    )

    backend.accept_model(
        "model-a",
        ModelSpec(base_model=backend.base_model, parameterization="full"),
    )
    datum = Datum(
        ModelInput.from_ints([1, 2]),
        {
            "target_tokens": TensorData(data=[2, 3], dtype="int64"),
            "weights": TensorData(data=[1.0, 1.0], dtype="float32"),
        },
    )
    batch = ForwardBatch(
        items=(ForwardItem("model-a", (datum,)),),
        loss_fn="cross_entropy",
    )

    assert backend.forward_backward(batch) == (output,)
    assert backend.accumulating
    result = backend.optim_step(
        ("model-a",),
        AdamParams(learning_rate=1e-5),
    )

    assert result == (
        OptimStepResponse(
            metrics={
                "grad_norm:mean": 0.5,
                "update_successful:mean": 1.0,
            }
        ),
    )
    assert backend.optimizer_step == 1
    assert not backend.accumulating
    assert calls.count("step") == 1


def test_checkpoint_load_reuses_initialized_engine(tmp_path, monkeypatch) -> None:
    source = tmp_path / "checkpoint"
    source.mkdir()
    (source / "deepspeed_checkpoint.json").write_text(
        json.dumps(
            {
                "base_model": "Qwen/Qwen3-8B",
                "world_size": 4,
                "zero_stage": 2,
                "has_optimizer": True,
            }
        ),
        encoding="utf-8",
    )
    calls = []
    base_optimizer = SimpleNamespace(state={"old": object()})
    engine = SimpleNamespace(
        optimizer=SimpleNamespace(optimizer=base_optimizer),
        load_checkpoint=lambda *args, **kwargs: (
            calls.append((args, kwargs)) or (str(source), {"optimizer_step": 3})
        ),
        zero_grad=lambda: calls.append("zero_grad"),
    )
    backend = DeepSpeedFullBackend.__new__(DeepSpeedFullBackend)
    backend.model_id = "model-a"
    backend.base_model = "Qwen/Qwen3-8B"
    backend.world_size = 4
    backend.config = SimpleNamespace(zero_stage=2)
    backend.engine = engine
    backend.optimizer_step = 9
    backend.accumulating = True
    backend._delta_writer = object()
    backend._sampler_captures = {"capture": object()}
    monkeypatch.setattr(backend, "_reload_checkpoint_volume", lambda: None)
    monkeypatch.setattr(
        backend,
        "_create_engine",
        lambda checkpoint: pytest.fail("load recreated DeepSpeed"),
    )

    backend.load_checkpoint("model-a", str(source), restore_optimizer=False)

    assert calls[0][1]["load_optimizer_states"] is False
    assert base_optimizer.state == {}
    assert backend.optimizer_step == 0
    assert not backend.accumulating
    assert backend._delta_writer is None
    assert backend._sampler_captures == {}


def test_deepspeed_build_executor_constructs_backend(monkeypatch) -> None:
    config = object()
    captured = {}

    def construct(value, **kwargs):
        captured.update(config=value, **kwargs)
        return "backend"

    monkeypatch.setenv("LILO_BACKEND_CONFIG", '{"deepspeed": {}}')
    monkeypatch.setenv("LILO_BASE_MODEL", "Qwen/Qwen3-4B")
    monkeypatch.setattr(
        deepspeed_full,
        "initialize_distributed_runtime",
        lambda: ("command", "checkpoint", "sampler"),
    )
    monkeypatch.setattr(
        deepspeed_full,
        "parse_deepspeed_backend_config",
        lambda value: (config, Path("/checkpoints")),
    )
    monkeypatch.setattr(deepspeed_full, "DeepSpeedFullBackend", construct)

    executor = deepspeed_full.build_executor()

    assert executor.backend == "backend"
    assert executor.command_group == "command"
    assert executor.checkpoint_persistence_group == "checkpoint"
    assert executor.sampler_persistence_group == "sampler"
    assert captured == {
        "config": config,
        "checkpoint_dir": Path("/checkpoints"),
        "base_model": "Qwen/Qwen3-4B",
        "persistence_group": "checkpoint",
    }
