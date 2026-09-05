"""Build, pack, execute, and collect Megatron training batches."""

from __future__ import annotations

import contextlib
import math
from typing import Any

import torch
from tinker import ForwardBackwardOutput, TensorData

from ...contract import ForwardBatch


def log_packing(
    batch: ForwardBatch,
    packed: list[dict[str, Any]],
    *,
    input_sequences: int,
    raw_tokens: int,
    token_capacity: int,
    rank: int,
) -> dict[str, float]:
    padded_tokens = sum(int(packed_bin["tokens"].numel()) for packed_bin in packed)
    metrics = {
        "packing_input_sequences:sum": float(input_sequences),
        "packing_raw_tokens:sum": float(raw_tokens),
        "packing_padded_tokens:sum": float(padded_tokens),
        "packing_bins:sum": float(len(packed)),
        "packing_token_capacity:max": float(token_capacity),
        "packing_padding_fraction:mean": (
            (padded_tokens - raw_tokens) / max(padded_tokens, 1)
        ),
        "packing_utilization:mean": (raw_tokens / max(len(packed) * token_capacity, 1)),
    }
    if rank == 0:
        print(
            "packed forward_backward "
            f"requests={len(batch.items)} items={len(batch.items)} "
            f"input_sequences={input_sequences} raw_tokens={raw_tokens} "
            f"padded_tokens={padded_tokens} packed_bins={len(packed)} "
            f"packed_microbatches={len(packed)} token_capacity={token_capacity} "
            f"padding_fraction={metrics['packing_padding_fraction:mean']:.6f} "
            f"packing_utilization={metrics['packing_utilization:mean']:.6f}",
            flush=True,
        )
    return metrics


def add_packing_metrics(
    outputs: tuple[ForwardBackwardOutput, ...],
    metrics: dict[str, float],
) -> None:
    for output in outputs:
        output.metrics.update(
            {
                key: value / len(outputs) if key.endswith(":sum") else value
                for key, value in metrics.items()
            }
        )


def _packed_seq_idx(
    token_indices: torch.Tensor,
    padded_cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    return (
        torch.bucketize(
            token_indices,
            padded_cu_seqlens[1:],
            right=True,
        )
        .to(torch.int32)
        .unsqueeze(0)
    )


def _loss_config(loss_name: str, config: dict[str, float]) -> dict[str, float]:
    allowed = {
        "cross_entropy": frozenset(),
        "importance_sampling": frozenset(),
        "ppo": frozenset({"clip_low_threshold", "clip_high_threshold"}),
        "cispo": frozenset({"clip_low_threshold", "clip_high_threshold"}),
        "dro": frozenset({"beta"}),
        "dppo": frozenset({"tv_threshold"}),
    }
    if loss_name not in allowed:
        raise ValueError(f"unsupported loss_fn: {loss_name}")
    unexpected = set(config) - allowed[loss_name]
    if unexpected:
        raise ValueError(
            f"unsupported {loss_name} loss_fn_config: {sorted(unexpected)}"
        )
    values = {key: float(value) for key, value in config.items()}
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("loss_fn_config values must be finite")
    if loss_name in {"ppo", "cispo"}:
        low = values.get(
            "clip_low_threshold",
            0.8 if loss_name == "ppo" else 0.0,
        )
        high = values.get(
            "clip_high_threshold",
            1.2 if loss_name == "ppo" else 4.0,
        )
        if low < 0 or high < low:
            raise ValueError("clip thresholds must satisfy 0 <= low <= high")
    if loss_name == "dro" and values.get("beta", 0.05) < 0:
        raise ValueError("beta must be non-negative")
    if loss_name == "dppo" and values.get("tv_threshold", 0.1) < 0:
        raise ValueError("tv_threshold must be non-negative")
    return values


RL_LOSSES = frozenset({"importance_sampling", "ppo", "cispo", "dro", "dppo"})


def build_sequence_batches(
    batch: ForwardBatch,
    adapter_slots: dict[str, int] | None,
    *,
    max_slots: int | None,
    max_seq_length: int,
) -> list[dict[str, Any]]:
    """Build one unpacked sequence batch per Tinker datum.

    FFT passes no adapter slots. LoRA adds slot token counts before the shared
    packing step.
    """

    loss_name = batch.loss_fn
    loss_config = _loss_config(loss_name, dict(batch.loss_fn_config))
    sequence_batches = []
    for job_id, item in enumerate(batch.items):
        slot = adapter_slots[item.model_id] if adapter_slots is not None else None
        if not item.data:
            raise ValueError(f"forward_backward has no data for job {item.model_id}")
        for output_index, datum in enumerate(item.data):
            input_ids = list(datum.model_input.to_ints())
            if not input_ids:
                raise ValueError("input_ids cannot be empty")
            if len(input_ids) > max_seq_length:
                raise ValueError(
                    f"sequence length {len(input_ids)} exceeds {max_seq_length}"
                )

            target = datum.loss_fn_inputs.get("target_tokens")
            labels = list(target.data) if target is not None else []
            if not labels:
                labels = input_ids[1:] + [-100]

            sampling_logprobs = [0.0] * len(input_ids)
            advantages = [0.0] * len(input_ids)
            weights_input = datum.loss_fn_inputs.get("weights")
            weights = list(weights_input.data) if weights_input is not None else []
            if loss_name == "cross_entropy":
                weights = weights or [1.0] * len(input_ids)
            elif loss_name in RL_LOSSES:
                logprobs = datum.loss_fn_inputs.get("logprobs")
                advantage_input = datum.loss_fn_inputs.get("advantages")
                sampling_logprobs = list(logprobs.data) if logprobs is not None else []
                advantages = (
                    list(advantage_input.data) if advantage_input is not None else []
                )
                if len(sampling_logprobs) != len(input_ids) or len(advantages) != len(
                    input_ids
                ):
                    raise ValueError(
                        "target_tokens, logprobs, and advantages must match "
                        "input_ids length"
                    )
                weights = weights or [
                    float(logprob != 0.0 or advantage != 0.0)
                    for logprob, advantage in zip(
                        sampling_logprobs,
                        advantages,
                        strict=True,
                    )
                ]

            if len(labels) != len(input_ids) or len(weights) != len(input_ids):
                raise ValueError(
                    "target_tokens and weights must match input_ids length"
                )

            sequence = {
                "job_id": job_id,
                "output_index": output_index,
                "original_length": len(input_ids),
                "is_dummy": False,
                "tokens": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "loss_mask": torch.tensor(weights, dtype=torch.float32),
                "sampling_logprobs": torch.tensor(
                    sampling_logprobs,
                    dtype=torch.float32,
                ),
                "advantages": torch.tensor(advantages, dtype=torch.float32),
                "loss_name": loss_name,
                "loss_config": loss_config,
            }
            if slot is not None:
                adapter_token_counts = [0] * max_slots
                adapter_token_counts[slot] = len(input_ids)
                sequence["adapter_token_counts"] = torch.tensor(
                    adapter_token_counts,
                    dtype=torch.int32,
                )
            sequence_batches.append(sequence)
    return sequence_batches


def pack_microbatches(
    microbatches: list[dict[str, Any]],
    *,
    max_tokens: int,
    pad_to_multiple: int,
    total_pad_to_multiple: int = 1,
) -> list[dict[str, Any]]:
    import torch.nn.functional as F

    bins: list[tuple[int, list[dict[str, Any]]]] = []
    for batch in sorted(
        microbatches,
        key=lambda item: int(item["tokens"].numel()),
        reverse=True,
    ):
        length = (
            math.ceil(int(batch["tokens"].numel()) / pad_to_multiple) * pad_to_multiple
        )
        destination = next(
            (
                index
                for index, (size, _) in enumerate(bins)
                if math.ceil((size + length) / total_pad_to_multiple)
                * total_pad_to_multiple
                <= max_tokens
            ),
            None,
        )
        if destination is None:
            bins.append((length, [batch]))
        else:
            size, items = bins[destination]
            bins[destination] = (size + length, [*items, batch])

    packed = []
    for _, items in bins:
        if "adapter_token_counts" in items[0]:
            items.sort(key=lambda item: int(item["adapter_token_counts"].argmax()))
        lengths = [int(item["tokens"].numel()) for item in items]
        padded_lengths = [
            math.ceil(length / pad_to_multiple) * pad_to_multiple for length in lengths
        ]
        padded_offsets = [0]
        tokens = []
        labels = []
        loss_masks = []
        sampling_logprobs = []
        advantages = []
        position_ids = []
        adapter_token_counts = (
            torch.zeros_like(items[0]["adapter_token_counts"])
            if "adapter_token_counts" in items[0]
            else None
        )
        for item, length, padded_length in zip(
            items,
            lengths,
            padded_lengths,
            strict=True,
        ):
            padding = padded_length - length
            tokens.append(F.pad(item["tokens"], (0, padding), value=0))
            labels.append(F.pad(item["labels"], (0, padding), value=-100))
            loss_masks.append(F.pad(item["loss_mask"], (0, padding), value=0.0))
            sampling_logprobs.append(
                F.pad(item["sampling_logprobs"], (0, padding), value=0.0)
            )
            advantages.append(F.pad(item["advantages"], (0, padding), value=0.0))
            position_ids.append(F.pad(torch.arange(length), (0, padding), value=0))
            padded_offsets.append(padded_offsets[-1] + padded_length)
            if adapter_token_counts is not None:
                active_slot = int(item["adapter_token_counts"].argmax())
                adapter_token_counts[active_slot] += padded_length

        starts = tuple(padded_offsets[:-1])
        total_padding = (
            math.ceil(padded_offsets[-1] / total_pad_to_multiple)
            * total_pad_to_multiple
            - padded_offsets[-1]
        )
        if total_padding:
            tokens.append(torch.zeros(total_padding, dtype=tokens[0].dtype))
            labels.append(torch.full((total_padding,), -100, dtype=labels[0].dtype))
            loss_masks.append(torch.zeros(total_padding, dtype=loss_masks[0].dtype))
            sampling_logprobs.append(
                torch.zeros(total_padding, dtype=sampling_logprobs[0].dtype)
            )
            advantages.append(torch.zeros(total_padding, dtype=advantages[0].dtype))
            position_ids.append(torch.arange(total_padding))
            padded_offsets.append(padded_offsets[-1] + total_padding)
            if adapter_token_counts is not None:
                adapter_token_counts[active_slot] += total_padding

        batch = {
            "job_ids": tuple(item["job_id"] for item in items),
            "output_indices": tuple(item["output_index"] for item in items),
            "starts": starts,
            "original_lengths": tuple(lengths),
            "is_dummy": tuple(item["is_dummy"] for item in items),
            "tokens": torch.cat(tokens),
            "labels": torch.cat(labels),
            "loss_mask": torch.cat(loss_masks),
            "sampling_logprobs": torch.cat(sampling_logprobs),
            "advantages": torch.cat(advantages),
            "position_ids": torch.cat(position_ids),
            "token_indices": torch.arange(padded_offsets[-1]),
            "loss_name": items[0]["loss_name"],
            "loss_config": items[0]["loss_config"],
            "cu_seqlens": torch.tensor(padded_offsets, dtype=torch.int32),
            "max_seqlen": max(*padded_lengths, total_padding),
        }
        if adapter_token_counts is not None:
            batch["adapter_token_counts"] = adapter_token_counts
        packed.append(batch)
    return packed


def shard_microbatches(
    microbatches: list[dict[str, Any]],
    *,
    data_parallel_rank: int,
    data_parallel_size: int,
) -> list[dict[str, Any]]:
    local = list(microbatches[data_parallel_rank::data_parallel_size])
    target_count = math.ceil(len(microbatches) / data_parallel_size)
    while len(local) < target_count:
        source = microbatches[0]
        dummy = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in source.items()
        }
        dummy["output_indices"] = tuple(-1 for _ in dummy["output_indices"])
        dummy["is_dummy"] = tuple(True for _ in dummy["is_dummy"])
        dummy["labels"].fill_(-100)
        dummy["loss_mask"].zero_()
        dummy["advantages"].zero_()
        local.append(dummy)
    return local


def _vocab_parallel_logprobs(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    from megatron.core import mpu
    from megatron.core.fusions.fused_cross_entropy import (
        fused_vocab_parallel_cross_entropy,
    )

    logits = logits.reshape(-1, logits.shape[-1])
    labels = labels.reshape(-1).to(logits.device)
    valid = labels != -100
    return (
        -fused_vocab_parallel_cross_entropy(
            logits.unsqueeze(1),
            labels.masked_fill(~valid, 0).unsqueeze(1),
            mpu.get_tensor_model_parallel_group(),
        )
        .squeeze(-1)
        .squeeze(-1)
        .masked_fill(~valid, 0)
        .float()
    )


def _loss(
    logprobs: torch.Tensor,
    batch: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = batch["loss_mask"].to(logprobs.device).reshape(-1)
    loss_name = batch["loss_name"]
    if loss_name == "cross_entropy":
        loss = -(logprobs * mask).sum()
    elif loss_name in RL_LOSSES:
        sampling_logprobs = batch["sampling_logprobs"].to(logprobs.device).reshape(-1)
        advantages = batch["advantages"].to(logprobs.device).reshape(-1)
        probability_ratio = torch.exp(logprobs - sampling_logprobs)
        if loss_name == "importance_sampling":
            objective = probability_ratio * advantages
        elif loss_name == "dppo":
            threshold = batch["loss_config"].get("tv_threshold", 0.1)
            ratio = probability_ratio.detach()
            divergence = (ratio * sampling_logprobs.exp() - sampling_logprobs.exp()).abs()
            leaving = ((advantages > 0) & (ratio > 1)) | ((advantages < 0) & (ratio < 1))
            blocked = leaving & (divergence > threshold)
            objective = probability_ratio * advantages * (~blocked).to(logprobs.dtype)
        elif loss_name == "ppo":
            low = batch["loss_config"].get("clip_low_threshold", 0.8)
            high = batch["loss_config"].get("clip_high_threshold", 1.2)
            clipped = torch.clamp(probability_ratio, low, high)
            objective = torch.minimum(
                probability_ratio * advantages,
                clipped * advantages,
            )
        elif loss_name == "cispo":
            low = batch["loss_config"].get("clip_low_threshold", 0.0)
            high = batch["loss_config"].get("clip_high_threshold", 4.0)
            objective = (
                torch.clamp(probability_ratio, low, high).detach()
                * logprobs
                * advantages
            )
        else:
            beta = batch["loss_config"].get("beta", 0.05)
            objective = (
                logprobs * advantages
                - 0.5 * beta * (logprobs - sampling_logprobs).square()
            )
        loss = -(objective * mask).sum()
    return loss, mask.count_nonzero()


def make_forward_step(
    output_collector: dict[str, list[dict[str, Any]]],
    metric_collector: dict[str, dict[str, torch.Tensor]],
    *,
    route_adapters: bool = True,
    defer_fp32_logits: bool = False,
):
    from megatron.bridge.training.utils.packed_seq_utils import (
        get_packed_seq_cp_partition_indices,
        get_packed_seq_params,
        get_packed_seq_q_cu_seqlens,
    )
    from megatron.core import parallel_state
    from megatron.core.utils import get_model_config

    if route_adapters:
        from megatron.bridge.peft.multi_lora_layers import (
            set_tokens_per_adapter_slot,
        )

    def forward_step(data_iterator, model):
        batch = next(data_iterator)
        device = torch.cuda.current_device()
        model_config = get_model_config(model)
        for key, value in tuple(batch.items()):
            if torch.is_tensor(value):
                batch[key] = value.to(device)

        total_tokens = int(batch["tokens"].numel())
        packed_metadata = {
            "cu_seqlens": batch["cu_seqlens"],
            "cu_seqlens_argmin": torch.tensor(
                batch["cu_seqlens"].numel(),
                device=device,
            ),
            "max_seqlen": torch.tensor(batch["max_seqlen"], device=device),
        }
        packed_seq_params = get_packed_seq_params(packed_metadata)

        context_parallel_size = parallel_state.get_context_parallel_world_size()
        if context_parallel_size > 1:
            index = get_packed_seq_cp_partition_indices(
                packed_seq_params,
                total_tokens=total_tokens,
                cp_size=context_parallel_size,
                cp_rank=parallel_state.get_context_parallel_rank(),
                device=batch["tokens"].device,
            )
            keys = (
                "tokens",
                "labels",
                "loss_mask",
                "sampling_logprobs",
                "advantages",
                "position_ids",
                "token_indices",
            )
            if hasattr(model_config, "mrope_section"):
                keys = tuple(
                    key for key in keys if key not in ("tokens", "position_ids")
                )
            for key in keys:
                batch[key] = batch[key].index_select(0, index)

        if getattr(model_config, "is_hybrid_model", False):
            _, padded_cu_seqlens = get_packed_seq_q_cu_seqlens(packed_seq_params)
            if padded_cu_seqlens is None:
                raise ValueError("hybrid packed input requires cu-seqlens")
            packed_seq_params.seq_idx = _packed_seq_idx(
                batch["token_indices"],
                padded_cu_seqlens,
            )

        if route_adapters:
            if context_parallel_size > 1:
                batch["adapter_token_counts"] //= context_parallel_size
            set_tokens_per_adapter_slot(model, batch["adapter_token_counts"])
        model_kwargs = {"fp32_output": False} if defer_fp32_logits else {}
        logits = model(
            input_ids=batch["tokens"].unsqueeze(0),
            position_ids=batch["position_ids"].unsqueeze(0),
            attention_mask=None,
            packed_seq_params=packed_seq_params,
            **model_kwargs,
        )

        def loss_func(output):
            logprobs = _vocab_parallel_logprobs(output, batch["labels"])
            loss, token_count = _loss(logprobs, batch)
            positions = batch["token_indices"].reshape(-1)
            for job_id, output_index, start, length, is_dummy in zip(
                batch["job_ids"],
                batch["output_indices"],
                batch["starts"],
                batch["original_lengths"],
                batch["is_dummy"],
                strict=True,
            ):
                if is_dummy:
                    continue
                selected = (positions >= start) & (positions < start + length)
                output_collector.setdefault(job_id, []).append(
                    {
                        "output_index": output_index,
                        "positions": (positions[selected] - start).detach().clone(),
                        "logprobs": logprobs[selected].detach().clone(),
                    }
                )
                sequence_batch = {
                    **batch,
                    "loss_mask": batch["loss_mask"][selected],
                    "sampling_logprobs": batch["sampling_logprobs"][selected],
                    "advantages": batch["advantages"][selected],
                }
                sequence_loss, sequence_tokens = _loss(
                    logprobs[selected],
                    sequence_batch,
                )
                metrics = metric_collector.setdefault(
                    job_id,
                    {
                        "loss": torch.zeros((), device=loss.device),
                        "tokens": torch.zeros((), device=loss.device),
                        "sequences": torch.zeros((), device=loss.device),
                    },
                )
                metrics["loss"] += sequence_loss.detach()
                metrics["tokens"] += sequence_tokens.detach()
                metrics["sequences"] += 1

            report = torch.stack([token_count.detach().float(), loss.detach().float()])
            return (
                loss,
                token_count.to(dtype=torch.long).clamp_min(1),
                {"keys": ["loss"], "values": report},
            )

        return logits, loss_func

    return forward_step


def _cpu_payload(output_collector, metric_collector) -> dict[str, Any]:
    return {
        "outputs": {
            job_id: [
                {
                    "output_index": output["output_index"],
                    "positions": output["positions"].cpu().tolist(),
                    "logprobs": output["logprobs"].float().cpu().tolist(),
                }
                for output in outputs
            ]
            for job_id, outputs in output_collector.items()
        },
        "metrics": {
            job_id: {key: float(value) for key, value in metrics.items()}
            for job_id, metrics in metric_collector.items()
        },
    }


def _merge_payloads(
    payloads: list[dict[str, Any]],
    *,
    context_parallel: bool,
) -> dict[str, Any]:
    merged: dict[str, Any] = {"outputs": {}, "metrics": {}}
    for payload in payloads:
        for job_id, outputs in payload["outputs"].items():
            merged["outputs"].setdefault(job_id, []).extend(outputs)
        for job_id, metrics in payload["metrics"].items():
            target = merged["metrics"].setdefault(
                job_id,
                {"loss": 0.0, "tokens": 0.0, "sequences": 0.0},
            )
            target["loss"] += metrics["loss"]
            target["tokens"] += metrics["tokens"]
            if context_parallel:
                target["sequences"] = max(
                    target["sequences"],
                    metrics["sequences"],
                )
            else:
                target["sequences"] += metrics["sequences"]
    return merged


def synchronize_collectors(
    output_collector,
    metric_collector,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, float]]]:
    import torch.distributed as dist
    from megatron.core import parallel_state

    if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
        payload = _cpu_payload(output_collector, metric_collector)
        context_parallel_size = parallel_state.get_context_parallel_world_size()
        if context_parallel_size > 1:
            context_payloads = [None] * context_parallel_size
            dist.all_gather_object(
                context_payloads,
                payload,
                group=parallel_state.get_context_parallel_group(),
            )
            payload = _merge_payloads(
                context_payloads,
                context_parallel=True,
            )
    else:
        payload = None

    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    if pipeline_parallel_size > 1:
        pipeline_payload = [payload]
        dist.broadcast_object_list(
            pipeline_payload,
            src=parallel_state.get_pipeline_model_parallel_last_rank(),
            group=parallel_state.get_pipeline_model_parallel_group(),
        )
        payload = pipeline_payload[0]

    data_parallel_size = parallel_state.get_data_parallel_world_size(
        with_context_parallel=False,
    )
    if data_parallel_size > 1:
        data_payloads = [None] * data_parallel_size
        dist.all_gather_object(
            data_payloads,
            payload,
            group=parallel_state.get_data_parallel_group(
                with_context_parallel=False,
            ),
        )
        payload = _merge_payloads(data_payloads, context_parallel=False)

    assert payload is not None
    return payload["outputs"], payload["metrics"]


def build_outputs(
    batch: ForwardBatch,
    output_collector: dict[int, list[dict[str, Any]]],
    metric_collector: dict[int, dict[str, float]],
) -> tuple[ForwardBackwardOutput, ...]:
    outputs = []
    for job_id in range(len(batch.items)):
        metrics = metric_collector[job_id]
        tokens = metrics["tokens"]
        sequences = metrics["sequences"]
        loss = metrics["loss"]

        by_output_index: dict[int, list[tuple[int, float]]] = {}
        for output in output_collector.get(job_id, []):
            values = by_output_index.setdefault(output["output_index"], [])
            values.extend(
                zip(
                    output["positions"],
                    output["logprobs"],
                    strict=True,
                )
            )
        loss_outputs = [
            {
                "logprobs": TensorData(
                    data=[
                        value
                        for _, value in sorted(
                            by_output_index[output_index],
                            key=lambda item: item[0],
                        )
                    ],
                    dtype="float32",
                    shape=[len(by_output_index[output_index])],
                )
            }
            for output_index in sorted(by_output_index)
        ]
        outputs.append(
            ForwardBackwardOutput(
                loss_fn_output_type={
                    "cross_entropy": "MegatronSFTLoss",
                    "importance_sampling": "MegatronImportanceSamplingLoss",
                    "ppo": "MegatronPPOLoss",
                    "cispo": "MegatronCISPOLoss",
                    "dro": "MegatronDROLoss",
                    "dppo": "MegatronDPPOLoss",
                }[str(batch.loss_fn)],
                loss_fn_outputs=loss_outputs,
                metrics={
                    "loss:sum": loss,
                    "loss:mean": loss / max(tokens, 1.0),
                    "tokens:sum": tokens,
                    "n_sequences:sum": sequences,
                    "response_length:mean": tokens / max(sequences, 1.0),
                },
            )
        )
    return tuple(outputs)


def prepare_microbatches(
    batch: ForwardBatch,
    adapter_slots: dict[str, int] | None,
    *,
    max_slots: int | None = None,
    config,
    rank: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    from megatron.core import parallel_state

    sequences = build_sequence_batches(
        batch,
        adapter_slots,
        max_slots=max_slots,
        max_seq_length=config.seq_length,
    )
    raw_tokens = sum(int(item["tokens"].numel()) for item in sequences)
    input_sequences = len(sequences)
    context_parallel_size = parallel_state.get_context_parallel_world_size()
    token_capacity = config.packed_token_capacity()
    packed = pack_microbatches(
        sequences,
        max_tokens=token_capacity,
        pad_to_multiple=(2 * context_parallel_size if context_parallel_size > 1 else 1),
        total_pad_to_multiple=config.sequence_padding_multiple(),
    )
    packing_metrics = log_packing(
        batch,
        packed,
        input_sequences=input_sequences,
        raw_tokens=raw_tokens,
        token_capacity=token_capacity,
        rank=rank,
    )
    return (
        shard_microbatches(
            packed,
            data_parallel_rank=parallel_state.get_data_parallel_rank(
                with_context_parallel=False,
            ),
            data_parallel_size=parallel_state.get_data_parallel_world_size(
                with_context_parallel=False,
            ),
        ),
        packing_metrics,
    )


def run_megatron_pipeline(
    model,
    optimizer,
    microbatches: list[dict[str, Any]],
    *,
    forward_only: bool,
    route_adapters: bool,
    defer_fp32_logits: bool = False,
) -> tuple[
    dict[str, list[dict[str, torch.Tensor]]],
    dict[str, dict[str, torch.Tensor]],
]:
    from megatron.core.pipeline_parallel import get_forward_backward_func
    from megatron.core.utils import get_model_config

    output_collector: dict[str, list[dict[str, torch.Tensor]]] = {}
    metric_collector: dict[str, dict[str, torch.Tensor]] = {}
    forward_step = make_forward_step(
        output_collector,
        metric_collector,
        route_adapters=route_adapters,
        defer_fp32_logits=defer_fp32_logits,
    )
    model_config = get_model_config(model[0])
    model_config.grad_scale_func = optimizer.scale_loss
    model_config.timers = None
    model_config.finalize_model_grads_func = lambda *args, **kwargs: None

    contexts = contextlib.ExitStack()
    for chunk in model:
        no_sync = getattr(chunk, "no_sync", None)
        if callable(no_sync):
            contexts.enter_context(no_sync())
    with contexts:
        get_forward_backward_func()(
            forward_step_func=forward_step,
            data_iterator=iter(microbatches),
            model=model,
            num_microbatches=len(microbatches),
            seq_length=max(int(item["tokens"].numel()) for item in microbatches),
            micro_batch_size=1,
            forward_only=forward_only,
        )
    return output_collector, metric_collector
