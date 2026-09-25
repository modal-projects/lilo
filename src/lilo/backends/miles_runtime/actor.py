from __future__ import annotations

import json
import os
import shutil
import socket
from pathlib import Path

import modal
import torch
import torch.distributed as dist
from megatron.bridge import AutoBridge
from megatron.core import dist_checkpointing
from miles.backends.fsdp_utils import actor as fsdp_actor
from miles.backends.megatron_utils import actor as megatron_actor
from miles.backends.megatron_utils import model as megatron_model
from miles.backends.megatron_utils.lora import checkpoint
from miles.backends.megatron_utils.lora import model as lora_model
from miles.backends.megatron_utils.lora.actor import MultiLoRATrainRayActor
from miles.backends.training_utils import checkpoint_io, cp_utils, data, loss, mm_data
from miles.backends.training_utils.loss_hub import (
    logit_processors,
    math_utils,
    tinker_losses,
)
from miles.backends.training_utils.parallel import get_parallel_state
from miles.backends.training_utils.replay_data import fill_replay_data
from miles.backends.training_utils.weight_update import snapshot_publisher
from miles.utils.replay_base import routing_replay_manager

from .profiling import RankProfiler, TorchProfileConfig
from .replay import install_replay_hooks


def _pad_local_shard(
    tensor,
    total_length: int,
    response_length: int,
    *,
    qkv_format: str,
    max_seq_len,
):
    """Place this CP rank's zigzag logprob shard into a full-length response.

    Replicates the placement logic of miles'
    ``all_gather_with_cp`` without its differentiable ``dist.nn.all_reduce``.
    """
    _, _, logits_offset, _ = cp_utils.get_logits_and_tokens_offset_with_cp(
        total_length, response_length, qkv_format, max_seq_len
    )
    prompt_length = total_length - response_length

    chunk_0 = tensor[: logits_offset[0][1] - logits_offset[0][0]]
    chunk_1 = tensor[logits_offset[0][1] - logits_offset[0][0] :]
    assert chunk_1.shape[0] == logits_offset[1][1] - logits_offset[1][0]

    def zero(length: int):
        return torch.zeros(
            [length] + list(tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
            requires_grad=True,
        )

    if chunk_0.shape[0] == 0 and chunk_1.shape[0] == 0:
        padded = zero(response_length)
    elif chunk_0.shape[0] != 0 and chunk_1.shape[0] == 0:
        left = zero(logits_offset[0][0] - (prompt_length - 1))
        right = zero(total_length - 1 - logits_offset[0][1])
        padded = torch.cat([left, chunk_0, right], dim=0)
    elif chunk_0.shape[0] == 0 and chunk_1.shape[0] != 0:
        left = zero(logits_offset[1][0] - (prompt_length - 1))
        right = zero(total_length - 1 - logits_offset[1][1])
        padded = torch.cat([left, chunk_1, right], dim=0)
    else:
        left = zero(logits_offset[0][0] - (prompt_length - 1))
        mid = zero(logits_offset[1][0] - logits_offset[0][1])
        right = zero(total_length - 1 - logits_offset[1][1])
        padded = torch.cat([left, chunk_0, mid, chunk_1, right], dim=0)

    assert padded.shape[0] == response_length, (
        f"Expected {response_length}, got {padded.shape}"
    )
    return padded


def _gather_tinker_logprobs_across_cp() -> None:
    """Reassemble full-response logprobs for Miles' Tinker loss path under CP>1."""

    original = logit_processors.get_log_probs_and_entropy
    if getattr(original, "__lilo_gathers_cp__", False):
        return

    def get_log_probs_and_entropy(
        logits,
        *,
        args,
        unconcat_tokens,
        total_lengths,
        response_lengths,
        with_entropy=False,
        entropy_requires_grad=True,
        non_loss_data=True,
        max_seq_lens=None,
        rollout_sampling_mask=None,
    ):
        out = original(
            logits,
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            with_entropy=with_entropy,
            entropy_requires_grad=entropy_requires_grad,
            non_loss_data=non_loss_data,
            max_seq_lens=max_seq_lens,
            rollout_sampling_mask=rollout_sampling_mask,
        )
        parallel_state = get_parallel_state()
        if parallel_state.cp.size == 1 or args.allgather_cp:
            return out
        log_probs = []
        for index, (lp, total_length, response_length) in enumerate(
            zip(out["log_probs"], total_lengths, response_lengths, strict=True)
        ):
            max_seq_len = max_seq_lens[index] if max_seq_lens is not None else None
            padded = _pad_local_shard(
                lp,
                total_length,
                response_length,
                qkv_format=args.qkv_format,
                max_seq_len=max_seq_len,
            )
            summed = padded.detach().clone()
            dist.all_reduce(summed, group=parallel_state.cp.group)
            log_probs.append(padded + (summed - padded.detach()))
        out["log_probs"] = log_probs
        return out

    get_log_probs_and_entropy.__lilo_gathers_cp__ = True
    logit_processors.get_log_probs_and_entropy = get_log_probs_and_entropy

    # Callers on the multi-LoRA path that bound the name at import time must
    # be re-pointed. Miles' non-Tinker losses handle CP natively.
    for module in (
        loss,
        tinker_losses,
        megatron_actor,
        fsdp_actor,
    ):
        if module.get_log_probs_and_entropy is original:
            module.get_log_probs_and_entropy = get_log_probs_and_entropy

    # get_rollout_data slices rollout_log_probs/teacher_log_probs to the CP
    # shard for the native (non-tinker) losses. The tinker loss path pairs
    # them with full-response log_probs (gathered above) and full-length
    # advantages/loss_weights, so the slice must be disabled on this path.
    original_slice = cp_utils.slice_log_prob_with_cp
    if getattr(original_slice, "__lilo_unslices_cp__", False):
        return

    def slice_log_prob_with_cp(
        value, total_length, response_length, qkv_format, max_seq_len=None
    ):
        return value

    slice_log_prob_with_cp.__lilo_unslices_cp__ = True
    cp_utils.slice_log_prob_with_cp = slice_log_prob_with_cp
    for module in (data, mm_data, math_utils):
        if module.slice_log_prob_with_cp is original_slice:
            module.slice_log_prob_with_cp = slice_log_prob_with_cp


_gather_tinker_logprobs_across_cp()

install_replay_hooks(
    megatron_model=megatron_model,
    lora_model=lora_model,
    logit_processors=logit_processors,
    tinker_losses=tinker_losses,
    fill_replay_data=fill_replay_data,
    manager=routing_replay_manager,
)


def _checkpoint_volume_path(path: Path) -> str | None:
    """Locate ``path`` inside the checkpoint volume, or ``None`` if outside."""
    root = _checkpoint_root()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return None


def _checkpoint_root() -> Path:
    return Path(os.environ.get("LILO_CHECKPOINT_ROOT") or "/checkpoints")


def _write_checkpoint_dir_on_volume(
    volume_name: str,
    path: Path,
    relative: str,
    write_shards,
    metadata: dict | None,
) -> None:
    """Assemble a checkpoint from every node's committed shards."""
    tmp = path.parent / f"_tmp_{path.name}"
    tmp.mkdir(parents=True, exist_ok=True)
    tmp_relative = str(Path(relative).parent / tmp.name)
    _barrier()
    write_shards(tmp)
    if _rank() == 0 and metadata is not None:
        (tmp / "META.json").write_text(json.dumps(metadata, indent=2))
    written = _written_shard_names(tmp)
    _sync_checkpoint_volume("commit")

    def publish() -> None:
        volume = modal.Volume.from_name(volume_name)
        path.mkdir(parents=True, exist_ok=True)
        volume.commit()
        local = {shard.name: shard for shard in tmp.iterdir()}
        remote = [
            entry.path
            for entry in volume.listdir(tmp_relative)
            if entry.path.rsplit("/", 1)[-1] not in local
        ]
        missing = written - set(local) - {p.rsplit("/", 1)[-1] for p in remote}
        if missing:
            raise RuntimeError(
                f"checkpoint {relative} is missing {len(missing)} shard(s) "
                f"written by peers: {', '.join(sorted(missing))}"
            )
        if remote:
            volume.copy_files(remote, relative, recursive=True)
        for name, shard in local.items():
            shutil.copy2(shard, path / name)
        print(
            f"lilo_checkpoint_publish path={relative} "
            f"local={len(local)} remote={len(remote)}",
            flush=True,
        )

    _on_rank_zero(publish)
    shutil.rmtree(tmp, ignore_errors=True)
    _sync_checkpoint_volume("commit")


def _publish_checkpoints_across_nodes() -> None:
    """Publish checkpoint shards that are spread across nodes.

    Miles writes shards to a temp dir and has rank 0 rename it into place. On a
    Modal Volume each container sees only its own uncommitted writes, so that
    rename publishes rank 0's node and silently drops every other node's shards.
    """

    original = checkpoint_io.write_checkpoint_dir
    if getattr(original, "__lilo_publishes_across_nodes__", False):
        return

    def write_checkpoint_dir(path, write_shards, metadata=None, **kwargs):
        name = os.environ.get("LILO_CHECKPOINT_VOLUME")
        checkpoint_path = Path(path)
        relative = _checkpoint_volume_path(checkpoint_path)
        if name is not None and relative is None:
            print(
                f"lilo_checkpoint_publish path={path} volume={name} "
                f"reason=outside_checkpoint_root root={_checkpoint_root()}",
                flush=True,
            )
        if name is None or relative is None:
            return original(path, write_shards, metadata, **kwargs)
        return _write_checkpoint_dir_on_volume(
            name, checkpoint_path, relative, write_shards, metadata
        )

    write_checkpoint_dir.__lilo_publishes_across_nodes__ = True
    checkpoint_io.write_checkpoint_dir = write_checkpoint_dir
    for module in (checkpoint, snapshot_publisher):
        if module.write_checkpoint_dir is original:
            module.write_checkpoint_dir = write_checkpoint_dir


def _node_identity() -> str:
    """Identify the container a rank runs in."""
    return os.environ.get("MODAL_TASK_ID") or socket.gethostname()


def _sync_checkpoint_volume(action: str) -> None:
    """Commit or reload the checkpoint volume once per node."""
    name = os.environ.get("LILO_CHECKPOINT_VOLUME")
    if name is None:
        if "MODAL_TASK_ID" in os.environ:
            print(
                "lilo_checkpoint_volume action=skipped reason=unset",
                flush=True,
            )
        return
    if not dist.is_available() or not dist.is_initialized():
        _volume_action(name, action)
        return

    dist.barrier()
    hosts: list[str] = [""] * dist.get_world_size()
    dist.all_gather_object(hosts, _node_identity())
    representative = hosts.index(hosts[dist.get_rank()]) == dist.get_rank()
    _volume_action_on_representatives(name, action, representative)


def _written_shard_names(tmp: Path) -> set[str]:
    """All-gather the shard names each rank wrote."""
    mine = sorted(shard.name for shard in tmp.iterdir())
    if not _distributed():
        return set(mine)
    gathered: list[list[str]] = [[]] * dist.get_world_size()
    dist.all_gather_object(gathered, mine)
    return {name for names in gathered for name in names}


def _rank() -> int:
    return dist.get_rank() if _distributed() else 0


def _distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _barrier() -> None:
    if _distributed():
        dist.barrier()


def _on_rank_zero(work) -> None:
    """Run ``work`` on rank 0 and raise its failure on every rank.

    Ranks that skipped a failed step would otherwise carry on into the next
    collective and hang the job until the distributed timeout instead of
    surfacing the error.
    """
    failure: str | None = None
    if _rank() == 0:
        try:
            work()
        except Exception as exc:  # re-raised on every rank below
            failure = f"{type(exc).__name__}: {exc}"
    if _distributed():
        payload: list[str | None] = [failure]
        dist.broadcast_object_list(payload, src=0)
        failure = payload[0]
    if failure is not None:
        raise RuntimeError(f"checkpoint publish failed on rank 0: {failure}")


def _volume_action_on_representatives(
    name: str, action: str, representative: bool
) -> None:
    """Run ``action`` on one rank per node and fail on every rank or none."""
    failure: str | None = None
    if representative:
        try:
            _volume_action(name, action)
        except Exception as exc:  # re-raised on every rank below
            failure = f"rank {dist.get_rank()}: {type(exc).__name__}: {exc}"

    failures: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(failures, failure)
    reported = [f for f in failures if f is not None]
    if reported:
        raise RuntimeError(
            f"checkpoint volume {action} failed on {len(reported)} node(s): "
            + "; ".join(reported)
        )


def _volume_action(name: str, action: str) -> None:
    volume = modal.Volume.from_name(name)
    if action == "commit":
        volume.commit()
    else:
        volume.reload()
    print(f"lilo_checkpoint_volume action={action} volume={name}", flush=True)


_publish_checkpoints_across_nodes()


class LiloMilesTrainRayActor(MultiLoRATrainRayActor):
    """Upstream multi-LoRA actor with Qwen MTP and weights-only save support."""

    def init(self, args, role, **kwargs):
        # Miles's LoRA builder inherits checkpoint MTP heads without honoring
        # enable_mtp_training. Qwen3.5 then injects an auxiliary backward loss
        # even for a client datum whose weights are all zero.
        if args.enable_mtp_training:
            return super().init(args, role, **kwargs)
        original = AutoBridge.to_megatron_provider

        def provider_without_mtp(bridge, *provider_args, **provider_kwargs):
            provider = original(bridge, *provider_args, **provider_kwargs)
            provider.mtp_num_layers = None
            provider.mtp_hybrid_override_pattern = None
            return provider

        AutoBridge.to_megatron_provider = provider_without_mtp
        try:
            return super().init(args, role, **kwargs)
        finally:
            AutoBridge.to_megatron_provider = original

    def torch_profile_start(self) -> None:
        if not TorchProfileConfig.from_env().profiles_rank(dist.get_rank()):
            self._lilo_profiler = None
            return
        self._lilo_profiler = RankProfiler()
        self._lilo_profiler.start()

    def torch_profile_stop(self, output_dir: str) -> dict | None:
        profiler = self._lilo_profiler
        if profiler is None:
            return None
        self._lilo_profiler = None
        return profiler.stop(output_dir, f"rank{dist.get_rank()}")

    def forward_backward(self, *args, **kwargs):
        return self._profiled("forward_backward", *args, **kwargs)

    def optim_step(self, *args, **kwargs):
        return self._profiled("optim_step", *args, **kwargs)

    def forward_only(self, *args, **kwargs):
        return self._profiled("forward_only", *args, **kwargs)

    def load_slot(self, *args, **kwargs):
        if kwargs.get("ckpt_path") or len(args) > 3:
            _sync_checkpoint_volume("reload")
        return super().load_slot(*args, **kwargs)

    def save_slot(self, *args, **kwargs):
        result = super().save_slot(*args, **kwargs)
        _sync_checkpoint_volume("commit")
        return result

    def export_slot(self, *args, **kwargs):
        result = super().export_slot(*args, **kwargs)
        _sync_checkpoint_volume("commit")
        return result

    def _profiled(self, operation: str, *args, **kwargs):
        with torch.profiler.record_function(f"lilo/{operation}"):
            result = getattr(super(), operation)(*args, **kwargs)
        self._log_peak_memory(operation)
        return result

    @staticmethod
    def _log_peak_memory(operation: str) -> None:
        gib = 1024**3
        print(
            f"lilo_memory op={operation} "
            f"allocated_gb={torch.cuda.memory_allocated() / gib:.2f} "
            f"max_allocated_gb={torch.cuda.max_memory_allocated() / gib:.2f} "
            f"reserved_gb={torch.cuda.memory_reserved() / gib:.2f} "
            f"max_reserved_gb={torch.cuda.max_memory_reserved() / gib:.2f}",
            flush=True,
        )

    def export_slot_peft(
        self,
        *,
        slot: int,
        path: str,
        rank: int,
        alpha: float,
        base_model: str,
        target_modules: tuple[str, ...],
        lora_dropout: float,
    ) -> None:
        with torch.profiler.record_function("lilo/export_slot_peft"):
            return super().export_slot_peft(
                slot=slot,
                path=path,
                rank=rank,
                alpha=alpha,
                base_model=base_model,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
            )

    def save_slot_weights(self, slot: int, path: str) -> None:
        self._save_slot_weights(slot, path)
        _sync_checkpoint_volume("commit")

    def _save_slot_weights(self, slot: int, path: str) -> None:
        weights = checkpoint._slot_weights_sharded_state_dict(self.model, slot)
        sharded = {checkpoint._WEIGHTS_KEY: weights}
        checkpoint._canonicalize_slot_keys(sharded, slot)
        checkpoint_io.write_checkpoint_dir(
            path, lambda temporary: dist_checkpointing.save(sharded, str(temporary))
        )
