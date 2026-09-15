from __future__ import annotations

from .config import EngineModelConfig


def initialize_megatron(config: EngineModelConfig) -> None:
    import random

    import numpy as np
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state, tensor_parallel

    if not dist.is_initialized():
        raise RuntimeError("initialize the torch distributed process group first")
    if parallel_state.model_parallel_is_initialized():
        raise RuntimeError("Megatron model parallel state is already initialized")

    config.validate(dist.get_world_size())
    if config.gpu_memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(config.gpu_memory_fraction)
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=config.tensor_model_parallel_size,
        pipeline_model_parallel_size=config.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=(
            config.virtual_pipeline_model_parallel_size
        ),
        context_parallel_size=config.context_parallel_size,
        expert_model_parallel_size=config.expert_model_parallel_size,
        expert_tensor_parallel_size=config.expert_tensor_parallel_size,
    )

    seed = config.seed + 100 * parallel_state.get_pipeline_model_parallel_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tensor_parallel.model_parallel_cuda_manual_seed(seed)
