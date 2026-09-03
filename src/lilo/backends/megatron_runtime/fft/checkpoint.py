"""Save and restore FFT training state and portable Hugging Face weights."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import torch
import torch.distributed as dist
from huggingface_hub import save_torch_state_dict
from megatron.core.dist_checkpointing.mapping import (
    LocalNonpersistentObject,
    ShardedObject,
    ShardedTensor,
    ShardedTensorFactory,
)

from ..common.checkpoint_io import (
    commit_checkpoint_volume,
    rank_tag,
    write_checkpoint_metadata,
)

CHECKPOINT_FORMAT = "lilo.megatron_fft.checkpoint"
CHECKPOINT_FORMAT_VERSION = 2
CHECKPOINT_METADATA_FILENAME = "checkpoint_metadata.json"
REGULAR_OPTIMIZER_STATE_FORMAT = "megatron.optimizer_state_dict.v1"
DISTRIBUTED_OPTIMIZER_STATE_FORMAT = "megatron.dp_reshardable.local.v1"


@dataclass(frozen=True, slots=True)
class FFTCheckpointMetadata:
    format: str
    version: int
    checkpoint_id: str
    base_model: str
    world_size: int
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    virtual_pipeline_model_parallel_size: int | None
    context_parallel_size: int
    expert_model_parallel_size: int
    expert_tensor_parallel_size: int
    data_parallel_size: int
    precision: str
    use_distributed_optimizer: bool
    optimizer_config: dict[str, Any]
    has_optimizer: bool
    optimizer_state_format: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def identity(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return f"{self.checkpoint_id}:{hashlib.sha256(encoded).hexdigest()}"


def create_fft_checkpoint_metadata(
    config,
    *,
    checkpoint_id: str,
    base_model: str,
    include_optimizer: bool,
    world_size: int,
) -> FFTCheckpointMetadata:
    return FFTCheckpointMetadata(
        format=CHECKPOINT_FORMAT,
        version=CHECKPOINT_FORMAT_VERSION,
        checkpoint_id=checkpoint_id,
        base_model=base_model,
        world_size=world_size,
        tensor_model_parallel_size=config.tensor_model_parallel_size,
        pipeline_model_parallel_size=config.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=(
            config.virtual_pipeline_model_parallel_size
        ),
        context_parallel_size=config.context_parallel_size,
        expert_model_parallel_size=config.expert_model_parallel_size,
        expert_tensor_parallel_size=config.expert_tensor_parallel_size,
        data_parallel_size=config.data_parallel_size(world_size),
        precision="bf16" if config.bf16 else "fp16" if config.fp16 else "fp32",
        use_distributed_optimizer=config.use_distributed_optimizer,
        optimizer_config=asdict(config.optimizer),
        has_optimizer=include_optimizer,
        optimizer_state_format=(
            DISTRIBUTED_OPTIMIZER_STATE_FORMAT
            if include_optimizer and config.use_distributed_optimizer
            else REGULAR_OPTIMIZER_STATE_FORMAT if include_optimizer else None
        ),
    )


def capture_hf_weights(bridge, model) -> dict[str, Any] | None:
    """Gather Hugging Face weights and keep them on global rank zero."""
    writer = dist.get_rank() == 0
    weights = {}
    for exported in bridge.export_hf_weights(
        model,
        cpu=writer,
        show_progress=False,
        merge_adapter_weights=True,
    ):
        if writer:
            weights[exported.param_name] = exported.weight
    if writer and not weights:
        raise ValueError("Megatron Bridge exported no Hugging Face weights")
    return weights if writer else None


def capture_fft_checkpoint(
    model,
    optimizer,
    *,
    metadata: FFTCheckpointMetadata,
    include_optimizer: bool,
) -> dict[str, Any]:
    return {
        "metadata": metadata.to_dict(),
        "model": [_copy_to_cpu(chunk.state_dict()) for chunk in model],
        "optimizer": (
            capture_fft_optimizer_state(
                optimizer,
                use_distributed_optimizer=metadata.use_distributed_optimizer,
            )
            if include_optimizer
            else None
        ),
    }


def capture_fft_optimizer_state(
    optimizer,
    *,
    use_distributed_optimizer: bool,
) -> dict[str, Any]:
    if use_distributed_optimizer:
        sharded_state = optimizer.sharded_state_dict(
            model_sharded_state_dict={},
            is_loading=False,
            metadata={"distrib_optim_sharding_type": "dp_reshardable"},
        )
        return {
            "format": DISTRIBUTED_OPTIMIZER_STATE_FORMAT,
            "state_dict": _materialize_local_sharded_state(sharded_state),
        }
    return {
        "format": REGULAR_OPTIMIZER_STATE_FORMAT,
        "state_dict": _copy_to_cpu(optimizer.state_dict()),
    }


def restore_fft_optimizer_state(
    optimizer,
    state: Any,
    *,
    use_distributed_optimizer: bool,
) -> None:
    expected_format = (
        DISTRIBUTED_OPTIMIZER_STATE_FORMAT
        if use_distributed_optimizer
        else REGULAR_OPTIMIZER_STATE_FORMAT
    )
    if not isinstance(state, dict) or state.get("format") != expected_format:
        raise ValueError(
            "checkpoint optimizer state format mismatch: "
            f"expected {expected_format!r}"
        )
    state_dict = state.get("state_dict")
    if not isinstance(state_dict, dict | list):
        raise TypeError("checkpoint optimizer state_dict must be a dict or list")
    if use_distributed_optimizer:
        _validate_distributed_optimizer_state_dict(state_dict)
    optimizer.load_state_dict(state_dict)


def fft_checkpoint_path(uri: str) -> Path:
    return Path(uri) / f"checkpoint_{rank_tag()}.pt"


def fft_checkpoint_metadata_path(uri: str) -> Path:
    return Path(uri) / CHECKPOINT_METADATA_FILENAME


def write_fft_checkpoint(
    uri: str,
    checkpoint: dict[str, Any],
    *,
    hf_weights: dict[str, Any] | None,
    hf_checkpoint: str,
    metadata: dict[str, Any] | None = None,
    persistence_group,
) -> str:
    path = fft_checkpoint_path(uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    if hf_weights is not None:
        save_torch_state_dict(
            hf_weights,
            path.parent,
            safe_serialization=True,
        )
        for name in ("config.json", "generation_config.json"):
            source = Path(hf_checkpoint) / name
            if source.is_file():
                shutil.copy2(source, path.parent / name)

    if dist.get_rank() == 0:
        metadata_path = fft_checkpoint_metadata_path(uri)
        temporary = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=metadata_path.parent,
                prefix=f".{metadata_path.name}.",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(
                    checkpoint["metadata"],
                    handle,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, metadata_path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    write_checkpoint_metadata(uri, metadata)
    commit_checkpoint_volume(persistence_group)
    return uri


def load_fft_training_checkpoint(
    uri: str,
    config,
    *,
    base_model: str,
    world_size: int,
) -> tuple[dict[str, Any], FFTCheckpointMetadata]:
    """Load and check every rank's checkpoint before model state is changed."""

    metadata_path = fft_checkpoint_metadata_path(uri)
    try:
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"exact FFT resume requires {CHECKPOINT_METADATA_FILENAME}: {metadata_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid FFT checkpoint metadata: {metadata_path}") from exc
    metadata = FFTCheckpointMetadata(**value)
    expected = create_fft_checkpoint_metadata(
        config,
        checkpoint_id=metadata.checkpoint_id,
        base_model=base_model,
        include_optimizer=metadata.has_optimizer,
        world_size=world_size,
    )
    comparable_fields = (
        "format",
        "version",
        "base_model",
        "world_size",
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "virtual_pipeline_model_parallel_size",
        "context_parallel_size",
        "expert_model_parallel_size",
        "expert_tensor_parallel_size",
        "data_parallel_size",
        "precision",
        "use_distributed_optimizer",
        "optimizer_config",
        "optimizer_state_format",
    )
    for name in comparable_fields:
        saved = getattr(metadata, name)
        current = getattr(expected, name)
        if saved != current:
            raise ValueError(
                f"checkpoint metadata mismatch for {name}: "
                f"saved={saved!r} current={current!r}"
            )
    if not metadata.has_optimizer:
        raise ValueError("checkpoint metadata reports no optimizer state")

    checkpoint_path = fft_checkpoint_path(uri)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("metadata") != metadata.to_dict():
        raise ValueError(
            "native checkpoint metadata/checkpoint ID does not match "
            f"{CHECKPOINT_METADATA_FILENAME}"
        )
    optimizer_state = checkpoint.get("optimizer")
    if (
        not isinstance(optimizer_state, dict)
        or optimizer_state.get("format") != metadata.optimizer_state_format
        or not isinstance(optimizer_state.get("state_dict"), dict | list)
    ):
        raise ValueError(
            "native checkpoint does not contain the declared optimizer state format"
        )
    if metadata.use_distributed_optimizer:
        _validate_distributed_optimizer_state_dict(optimizer_state["state_dict"])

    return checkpoint, metadata


def synchronize_checkpoint_preflight(
    error: Exception | None,
    *,
    metadata_identity: str | None = None,
) -> None:
    """Make every rank reject a bad checkpoint before any rank mutates state."""

    local = {
        "error": (f"{type(error).__name__}: {error}" if error is not None else None),
        "metadata_identity": metadata_identity,
    }
    gathered: list[dict[str, str | None] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    failures = [
        f"rank {rank}: {item['error']}"
        for rank, item in enumerate(gathered)
        if item is not None and item["error"] is not None
    ]
    if failures:
        raise ValueError("checkpoint preflight failed: " + "; ".join(failures))
    identities = {item["metadata_identity"] for item in gathered if item is not None}
    if identities == {None}:
        return
    if len(identities) != 1 or None in identities:
        raise ValueError(
            "checkpoint metadata identity differs across ranks: "
            f"{sorted(str(identity) for identity in identities)}"
        )


def _materialize_local_sharded_state(value):
    """Detach Megatron's local torch_dist representation from live optimizer buffers."""
    if isinstance(value, ShardedTensorFactory):
        raise TypeError("distributed optimizer state contains an unresolved tensor factory")
    if isinstance(value, ShardedTensor):
        if value.data is None:
            raise ValueError("distributed optimizer sharded tensor has no local data")
        return _copy_to_cpu(value.data)
    if isinstance(value, ShardedObject):
        return _materialize_local_sharded_state(value.data)
    if isinstance(value, LocalNonpersistentObject):
        return _materialize_local_sharded_state(value.unwrap())
    if isinstance(value, dict):
        return {
            key: _materialize_local_sharded_state(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_materialize_local_sharded_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_materialize_local_sharded_state(item) for item in value)
    return _copy_to_cpu(value)


def _validate_distributed_optimizer_state_dict(state_dict: dict | list) -> None:
    """Require every DistributedOptimizer leaf to contain its local tensor shards."""
    if isinstance(state_dict, list):
        if not state_dict:
            raise ValueError("distributed optimizer checkpoint has no optimizer entries")
        for item in state_dict:
            if not isinstance(item, dict | list):
                raise TypeError("distributed optimizer entry must be a dict or list")
            _validate_distributed_optimizer_state_dict(item)
        return

    if state_dict.get("param_state_sharding_type") == "dp_reshardable":
        if not isinstance(state_dict.get("optimizer"), dict):
            raise ValueError("distributed optimizer checkpoint has no optimizer metadata")
        if not isinstance(state_dict.get("param_state"), dict) or not state_dict[
            "param_state"
        ]:
            raise ValueError("distributed optimizer checkpoint has no parameter state")
        return

    if state_dict and all(isinstance(key, int) for key in state_dict):
        for item in state_dict.values():
            if not isinstance(item, dict | list):
                raise TypeError("chained distributed optimizer entry must be a dict or list")
            _validate_distributed_optimizer_state_dict(item)
        return

    raise ValueError(
        "distributed optimizer checkpoint must use dp_reshardable parameter state"
    )


def _copy_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _copy_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_to_cpu(item) for item in value)
    return deepcopy(value)
