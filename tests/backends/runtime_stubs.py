from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager
from types import ModuleType
from unittest.mock import MagicMock


@contextmanager
def backend_runtime_imports():
    """Stub GPU-only packages while importing backend modules in unit tests."""

    added: list[str] = []

    def module(name: str, *, package: bool = False, **attributes) -> ModuleType:
        value = ModuleType(name)
        if package:
            value.__path__ = []
        for attribute, member in attributes.items():
            setattr(value, attribute, member)
        sys.modules[name] = value
        added.append(name)
        return value

    if importlib.util.find_spec("torch") is None:
        distributed = module("torch.distributed")
        functional = module("torch.nn.functional")
        nn = module("torch.nn", package=True, functional=functional)
        module(
            "torch",
            package=True,
            distributed=distributed,
            nn=nn,
            Tensor=type("Tensor", (), {}),
            device=MagicMock(),
        )
        module("safetensors.torch", save_file=MagicMock())

    if importlib.util.find_spec("megatron") is None:
        parallel_state = module("megatron.core.parallel_state")
        tensor_parallel = module("megatron.core.tensor_parallel", package=True)
        mpu = module("megatron.core.mpu")
        core = module(
            "megatron.core",
            package=True,
            parallel_state=parallel_state,
            tensor_parallel=tensor_parallel,
            mpu=mpu,
        )
        module(
            "megatron.core.distributed",
            DistributedDataParallelConfig=MagicMock(),
            finalize_model_grads=MagicMock(),
        )
        dist_checkpointing = module(
            "megatron.core.dist_checkpointing",
            package=True,
        )

        class ShardedTensor:
            def __init__(self, data):
                self.data = data

        class ShardedObject:
            def __init__(self, data):
                self.data = data

        class LocalNonpersistentObject:
            def __init__(self, value):
                self.value = value

            def unwrap(self):
                return self.value

        class ShardedTensorFactory:
            pass

        mapping = module(
            "megatron.core.dist_checkpointing.mapping",
            LocalNonpersistentObject=LocalNonpersistentObject,
            ShardedObject=ShardedObject,
            ShardedTensor=ShardedTensor,
            ShardedTensorFactory=ShardedTensorFactory,
        )
        dist_checkpointing.mapping = mapping
        module(
            "megatron.core.optimizer",
            OptimizerConfig=MagicMock(),
            get_megatron_optimizer=MagicMock(),
        )
        transformer = module("megatron.core.transformer", package=True)
        transformer_enums = module(
            "megatron.core.transformer.enums",
            AttnBackend=MagicMock(),
        )
        transformer.enums = transformer_enums
        fusions = module("megatron.core.fusions", package=True)
        fused_cross_entropy = module(
            "megatron.core.fusions.fused_cross_entropy",
            fused_vocab_parallel_cross_entropy=MagicMock(),
        )
        fusions.fused_cross_entropy = fused_cross_entropy
        module(
            "megatron.core.pipeline_parallel",
            get_forward_backward_func=MagicMock(),
        )
        module(
            "megatron.core.utils",
            get_model_config=MagicMock(),
            unwrap_model=MagicMock(),
        )
        random = module(
            "megatron.core.tensor_parallel.random",
            _MODEL_PARALLEL_RNG_TRACKER_NAME="model-parallel-rng",
            _fork_rng=MagicMock(),
            get_cuda_rng_tracker=MagicMock(),
            get_expert_parallel_rng_tracker_name=MagicMock(),
        )
        tensor_parallel.random = random

        bridge = module(
            "megatron.bridge",
            package=True,
            AutoBridge=MagicMock(),
        )
        peft = module("megatron.bridge.peft", package=True)
        module(
            "megatron.bridge.peft.multi_lora",
            MultiLoRA=MagicMock(),
        )
        multi_lora_layers = module(
            "megatron.bridge.peft.multi_lora_layers",
            clear_adapter_slot=MagicMock(),
            expose_adapter_slot=MagicMock(),
            load_adapter=MagicMock(),
            set_tokens_per_adapter_slot=MagicMock(),
        )
        peft.multi_lora_layers = multi_lora_layers
        training = module("megatron.bridge.training", package=True)
        training_utils = module("megatron.bridge.training.utils", package=True)
        packed = module(
            "megatron.bridge.training.utils.packed_seq_utils",
            get_packed_seq_cp_partition_indices=MagicMock(),
            get_packed_seq_params=MagicMock(),
            get_packed_seq_q_cu_seqlens=MagicMock(),
        )
        training_utils.packed_seq_utils = packed
        training.utils = training_utils
        bridge.peft = peft
        bridge.training = training
        module("megatron", package=True, bridge=bridge, core=core)

    try:
        yield
    finally:
        for name in reversed(added):
            sys.modules.pop(name, None)
