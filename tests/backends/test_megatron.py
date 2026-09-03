from contextlib import contextmanager

import pytest
from runtime_stubs import backend_runtime_imports

from lilo.backends.megatron_config import parse_backend_config
from lilo.backends.megatron_runtime.common.config import EngineModelConfig

with backend_runtime_imports():
    from lilo.backends.megatron_runtime.lora import peft as peft_module


@pytest.mark.parametrize(
    (
        "context_parallel_size",
        "tensor_parallel_size",
        "sequence_parallel",
        "expected",
    ),
    [
        (1, 4, True, 4),
        (2, 4, True, 8),
        (4, 2, True, 8),
        (1, 4, False, 1),
        (2, 1, False, 4),
    ],
)
def test_sequence_padding_covers_parallel_splits(
    context_parallel_size: int,
    tensor_parallel_size: int,
    sequence_parallel: bool,
    expected: int,
) -> None:
    config = EngineModelConfig(
        hf_checkpoint="/model",
        context_parallel_size=context_parallel_size,
        tensor_model_parallel_size=tensor_parallel_size,
        sequence_parallel=sequence_parallel,
    )

    assert config.sequence_padding_multiple() == expected


def test_packed_capacity_is_separate_from_model_context() -> None:
    config = EngineModelConfig(
        hf_checkpoint="/model",
        max_tokens_per_microbatch=4_096,
        seq_length=65_536,
    )

    assert config.packed_token_capacity() == 4_096
    assert config.seq_length == 65_536


def test_backend_config_parses_runtime_settings() -> None:
    config, checkpoint_dir = parse_backend_config(
        {
            "megatron": {
                "hf_checkpoint": "/models/qwen",
                "max_tokens_per_microbatch": 8192,
                "tensor_model_parallel_size": 2,
                "split_qkv": True,
                "split_gdn": True,
                "split_mamba": True,
                "gpu_memory_fraction": 0.9,
                "use_distributed_optimizer": True,
                "target_modules": ["linear_qkv"],
                "provider_overrides": {"mtp_num_layers": 0},
                "optimizer": {"lr": 2e-4},
            },
            "checkpoint_dir": "/checkpoints",
        }
    )
    assert config.hf_checkpoint == "/models/qwen"
    assert config.max_tokens_per_microbatch == 8192
    assert config.tensor_model_parallel_size == 2
    assert config.split_qkv is True
    assert config.split_gdn is True
    assert config.split_mamba is True
    assert config.gpu_memory_fraction == 0.9
    assert config.use_distributed_optimizer is True
    assert config.target_modules == ("linear_qkv",)
    assert config.provider_overrides == {"mtp_num_layers": 0}
    assert config.optimizer.lr == 2e-4
    assert str(checkpoint_dir) == "/checkpoints"


class FakeTensor:
    def __init__(self, shape, maximum=0):
        self.shape = shape
        self.maximum = maximum

    def __getitem__(self, indexes):
        if not isinstance(indexes, tuple):
            indexes = (indexes,)
        shape = []
        for size, index in zip(self.shape, indexes, strict=True):
            if isinstance(index, slice):
                start, stop, step = index.indices(size)
                shape.append(len(range(start, stop, step)))
            else:
                shape.append(size)
        return FakeTensor(tuple(shape), self.maximum)

    def abs(self):
        return self

    def max(self):
        return self

    def item(self):
        return self.maximum

    def clone(self):
        return FakeTensor(self.shape, self.maximum)


def test_peft_export_slices_rank_and_respects_train_flags(monkeypatch) -> None:
    _peft_target_modules = peft_module._peft_target_modules
    collect_adapter_weights = peft_module.collect_adapter_weights
    peft_target_modules = peft_module.peft_target_modules
    slice_lora_to_rank = peft_module.slice_lora_to_rank

    assert peft_target_modules(
        [
            "decoder.*.linear_qkv",
            "decoder.*.linear_proj",
            "decoder.*.in_proj",
            "decoder.*.out_proj",
            "decoder.*.linear_fc1",
            "decoder.*.linear_fc2",
            "output_layer",
        ],
        train_attn=True,
        train_mlp=False,
        train_unembed=True,
    ) == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "in_proj_q",
        "in_proj_k",
        "in_proj_v",
        "in_proj_z",
        "out_proj",
        "lm_head",
    ]
    assert peft_target_modules(
        [
            "linear_qkv",
            "linear_proj",
            "decoder.layers.*.mixer.in_proj",
            "decoder.layers.*.mixer.out_proj",
            "output_layer",
        ],
        train_attn=True,
        train_mlp=False,
        train_unembed=True,
    ) == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "in_proj",
        "out_proj",
        "lm_head",
    ]
    assert _peft_target_modules(
        ["decoder.layers.*.mixer.in_proj"],
        train_attn=True,
        train_mlp=False,
        train_unembed=False,
        split_mamba_weights=True,
    ) == ["gate_proj", "x_proj"]
    assert slice_lora_to_rank("layer.lora_A.weight", FakeTensor((4, 3)), 2).shape == (
        2,
        3,
    )
    assert slice_lora_to_rank("layer.lora_B.weight", FakeTensor((3, 4)), 2).shape == (
        3,
        2,
    )
    assert slice_lora_to_rank(
        "layer.k_proj.lora_A.weight",
        FakeTensor((6, 3), 1),
        2,
    ).shape == (2, 3)
    assert slice_lora_to_rank(
        "layer.v_proj.lora_B.weight",
        FakeTensor((3, 6), 1),
        2,
    ).shape == (3, 2)
    assert slice_lora_to_rank(
        "layer.v_proj.lora_A.weight",
        FakeTensor((4, 3)),
        2,
        fused_projection_rank_layout=False,
    ).shape == (2, 3)
    with pytest.raises(ValueError, match="smaller than projection"):
        slice_lora_to_rank(
            "layer.up_proj.lora_A.weight",
            FakeTensor((3, 3)),
            2,
        )
    with pytest.raises(ValueError, match="non-zero"):
        slice_lora_to_rank("layer.lora_A.weight", FakeTensor((4, 3), 1), 2)
    with pytest.raises(ValueError, match="smaller"):
        slice_lora_to_rank("layer.lora_B.weight", FakeTensor((3, 1)), 2)

    class Bridge:
        calls = 0

        def export_adapter_weights(self, model, **kwargs):
            self.calls += 1
            yield "layer.lora_A.weight", FakeTensor((4, 3)), "layer"

    @contextmanager
    def exposed(*args):
        yield

    monkeypatch.setattr(peft_module, "expose_adapter_slot", exposed)
    monkeypatch.setattr(peft_module, "patch_megatron_model", exposed)

    bridge = Bridge()
    state = collect_adapter_weights(
        bridge,
        object(),
        slot=3,
        adapter_rank=2,
    )
    assert bridge.calls == 1
    assert state["layer.lora_A.weight"].shape == (2, 3)

    class SplitGDNBridge:
        def export_adapter_weights(self, model, **kwargs):
            yield (
                "layer.in_proj_k.lora_A.weight",
                FakeTensor((2, 3)),
                "layer",
            )

    state = collect_adapter_weights(
        SplitGDNBridge(),
        object(),
        slot=3,
        adapter_rank=2,
        split_gdn=True,
    )
    assert state["layer.in_proj_k.lora_A.weight"].shape == (2, 3)


def test_peft_export_splits_qwen35_linear_attention() -> None:
    _split_gdn_adapter_weights = peft_module._split_gdn_adapter_weights
    slice_lora_to_rank = peft_module.slice_lora_to_rank

    prefix = "model.layers.0.linear_attn"
    state = _split_gdn_adapter_weights(
        {
            f"{prefix}.in_proj_qkv.lora_A.weight": FakeTensor((8, 5), 1),
            f"{prefix}.in_proj_qkv.lora_B.weight": FakeTensor((10, 8), 1),
            f"{prefix}.in_proj_z.lora_A.weight": FakeTensor((8, 5), 1),
            f"{prefix}.in_proj_z.lora_B.weight": FakeTensor((4, 8), 1),
            f"{prefix}.in_proj_b.lora_A.weight": FakeTensor((8, 5), 1),
            f"{prefix}.in_proj_b.lora_B.weight": FakeTensor((1, 8)),
            f"{prefix}.in_proj_a.lora_A.weight": FakeTensor((8, 5), 1),
            f"{prefix}.in_proj_a.lora_B.weight": FakeTensor((1, 8)),
        }
    )

    assert set(state) == {
        f"{prefix}.{target}.{kind}.weight"
        for target in ("in_proj_q", "in_proj_k", "in_proj_v", "in_proj_z")
        for kind in ("lora_A", "lora_B")
    }
    assert state[f"{prefix}.in_proj_q.lora_B.weight"].shape == (3, 8)
    assert state[f"{prefix}.in_proj_k.lora_B.weight"].shape == (3, 8)
    assert state[f"{prefix}.in_proj_v.lora_B.weight"].shape == (4, 8)
    assert slice_lora_to_rank(
        f"{prefix}.in_proj_z.lora_A.weight",
        state[f"{prefix}.in_proj_z.lora_A.weight"],
        2,
    ).shape == (2, 5)
    assert slice_lora_to_rank(
        f"{prefix}.in_proj_z.lora_B.weight",
        state[f"{prefix}.in_proj_z.lora_B.weight"],
        2,
    ).shape == (4, 2)
