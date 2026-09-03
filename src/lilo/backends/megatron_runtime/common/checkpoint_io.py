from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import modal
import torch.distributed as dist
from megatron.core import parallel_state


def rank_tag() -> str:
    ranks = [
        f"tp{parallel_state.get_tensor_model_parallel_rank()}",
        f"pp{parallel_state.get_pipeline_model_parallel_rank()}",
    ]
    if parallel_state.get_context_parallel_world_size() > 1:
        ranks.append(f"cp{parallel_state.get_context_parallel_rank()}")
    if parallel_state.get_expert_model_parallel_world_size() > 1:
        ranks.append(f"ep{parallel_state.get_expert_model_parallel_rank()}")
    if parallel_state.get_expert_tensor_parallel_world_size() > 1:
        ranks.append(f"etp{parallel_state.get_expert_tensor_parallel_rank()}")
    ranks.append(f"dp{parallel_state.get_data_parallel_rank()}")
    return "_".join(ranks)


def write_checkpoint_metadata(uri: str, metadata: dict[str, Any] | None) -> None:
    if metadata is None or dist.get_rank() != 0:
        return
    path = Path(uri) / "metadata.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def commit_checkpoint_volume(group) -> None:
    """Commit after every checkpoint-persistence rank finishes writing."""
    if group is not None:
        dist.barrier(group=group)
    if group is None or dist.get_rank(group=group) == 0:
        modal.Volume.from_name(os.environ["LILO_CHECKPOINT_VOLUME"]).commit()


def reload_checkpoint_volume() -> None:
    """Reload while every rank is synchronized on the command lane."""
    dist.barrier()
    if dist.get_rank() == 0:
        modal.Volume.from_name(os.environ["LILO_CHECKPOINT_VOLUME"]).reload()
    dist.barrier()
