from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any

import torch
import torch.distributed as dist
from tinker import ForwardBackwardOutput, TensorData

from lilo.backends.contract import ForwardBatch


def run_forward_backward(
    engine,
    batch: ForwardBatch,
    *,
    rank: int,
    world_size: int,
    max_sequence_length: int,
) -> tuple[ForwardBackwardOutput, ...]:
    if len(batch.items) != 1:
        raise ValueError("DeepSpeed full backend accepts one model per batch")
    item = batch.items[0]
    if not item.data:
        raise ValueError(f"forward_backward has no data for model {item.model_id}")

    local_records = [
        (index, datum)
        for index, datum in enumerate(item.data)
        if index % world_size == rank
    ]
    is_dummy = not local_records
    if is_dummy:
        local_records = [(-1, item.data[0])]

    tensors = _prepare_batch(
        local_records,
        loss_name=batch.loss_fn,
        loss_config=dict(batch.loss_fn_config),
        max_sequence_length=max_sequence_length,
        pad_token_id=engine.module.config.pad_token_id
        or engine.module.config.eos_token_id
        or 0,
        dummy=is_dummy,
    )
    device = engine.device
    tensors = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in tensors.items()
    }

    context = torch.no_grad() if batch.forward_only else nullcontext()
    with context:
        logits = engine(
            input_ids=tensors["input_ids"],
            attention_mask=tensors["attention_mask"],
            use_cache=False,
        ).logits
        logprobs = _target_logprobs(logits, tensors["labels"])
        loss = _loss(logprobs, tensors)

    if not batch.forward_only:
        # DeepSpeed averages gradients across data-parallel ranks. Scaling the
        # local sum preserves Lilo's global summed-loss update.
        engine.backward(loss * world_size)

    local_output = {
        "records": [
            (
                output_index,
                logprobs[row, :length].detach().float().cpu().tolist(),
            )
            for row, (output_index, _datum) in enumerate(local_records)
            if output_index >= 0
            for length in (int(tensors["lengths"][row]),)
        ],
        "loss": float(loss.detach()),
        "tokens": int(tensors["loss_mask"].count_nonzero()),
        "response_tokens": [
            int(tensors["loss_mask"][row].count_nonzero())
            for row, (output_index, _datum) in enumerate(local_records)
            if output_index >= 0
        ],
    }
    gathered: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered, local_output)

    records = sorted(
        (record for payload in gathered if payload for record in payload["records"]),
        key=lambda record: record[0],
    )
    total_loss = sum(float(payload["loss"]) for payload in gathered if payload)
    total_tokens = sum(int(payload["tokens"]) for payload in gathered if payload)
    response_tokens = [
        count
        for payload in gathered
        if payload
        for count in payload["response_tokens"]
    ]
    loss_outputs = [
        {
            "logprobs": TensorData(
                data=values,
                dtype="float32",
                shape=[len(values)],
            )
        }
        for _index, values in records
    ]
    return (
        ForwardBackwardOutput(
            loss_fn_output_type={
                "cross_entropy": "DeepSpeedSFTLoss",
                "importance_sampling": "DeepSpeedImportanceSamplingLoss",
            }[batch.loss_fn],
            loss_fn_outputs=loss_outputs,
            metrics={
                "loss:sum": total_loss,
                "loss:mean": total_loss / max(total_tokens, 1),
                "tokens:sum": float(total_tokens),
                "n_sequences:sum": float(len(records)),
                "response_length:mean": (
                    sum(response_tokens) / max(len(response_tokens), 1)
                ),
            },
        ),
    )


def _prepare_batch(
    records,
    *,
    loss_name: str,
    loss_config: dict[str, float],
    max_sequence_length: int,
    pad_token_id: int,
    dummy: bool,
) -> dict[str, Any]:
    if loss_name not in {"cross_entropy", "importance_sampling"}:
        raise ValueError(
            "minimal DeepSpeed backend supports cross_entropy and "
            "importance_sampling"
        )
    if loss_config:
        raise ValueError(f"unsupported {loss_name} loss_fn_config")

    prepared = []
    for output_index, datum in records:
        input_ids = list(datum.model_input.to_ints())
        if not input_ids:
            raise ValueError("input_ids cannot be empty")
        if len(input_ids) > max_sequence_length:
            raise ValueError(
                f"sequence length {len(input_ids)} exceeds {max_sequence_length}"
            )
        labels_input = datum.loss_fn_inputs.get("target_tokens")
        labels = list(labels_input.data) if labels_input is not None else []
        labels = labels or input_ids[1:] + [-100]
        weights_input = datum.loss_fn_inputs.get("weights")
        weights = list(weights_input.data) if weights_input is not None else []
        sampling_input = datum.loss_fn_inputs.get("logprobs")
        advantages_input = datum.loss_fn_inputs.get("advantages")
        sampling_logprobs = (
            list(sampling_input.data) if sampling_input is not None else []
        )
        advantages = (
            list(advantages_input.data) if advantages_input is not None else []
        )
        if len(labels) != len(input_ids):
            raise ValueError("target_tokens must match input_ids length")
        if loss_name == "cross_entropy":
            weights = weights or [1.0] * len(input_ids)
        else:
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
        if len(weights) != len(input_ids):
            raise ValueError("weights must match input_ids length")
        prepared.append(
            {
                "output_index": output_index,
                "input_ids": input_ids,
                "labels": labels,
                "weights": [0.0] * len(weights) if dummy else weights,
                "sampling_logprobs": sampling_logprobs
                or [0.0] * len(input_ids),
                "advantages": advantages or [0.0] * len(input_ids),
            }
        )

    padded_length = max(len(record["input_ids"]) for record in prepared)
    batch_size = len(prepared)
    input_ids = torch.full(
        (batch_size, padded_length),
        pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros((batch_size, padded_length), dtype=torch.long)
    labels = torch.full((batch_size, padded_length), -100, dtype=torch.long)
    loss_mask = torch.zeros((batch_size, padded_length), dtype=torch.float32)
    sampling_logprobs = torch.zeros(
        (batch_size, padded_length),
        dtype=torch.float32,
    )
    advantages = torch.zeros((batch_size, padded_length), dtype=torch.float32)
    lengths = []
    for row, record in enumerate(prepared):
        length = len(record["input_ids"])
        lengths.append(length)
        input_ids[row, :length] = torch.tensor(record["input_ids"])
        attention_mask[row, :length] = 1
        labels[row, :length] = torch.tensor(record["labels"])
        loss_mask[row, :length] = torch.tensor(record["weights"])
        sampling_logprobs[row, :length] = torch.tensor(
            record["sampling_logprobs"]
        )
        advantages[row, :length] = torch.tensor(record["advantages"])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "loss_mask": loss_mask,
        "sampling_logprobs": sampling_logprobs,
        "advantages": advantages,
        "lengths": lengths,
        "loss_name": loss_name,
    }


def _target_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    valid = labels != -100
    safe_labels = labels.masked_fill(~valid, 0)
    return (
        torch.log_softmax(logits.float(), dim=-1)
        .gather(-1, safe_labels.unsqueeze(-1))
        .squeeze(-1)
        .masked_fill(~valid, 0.0)
    )


def _loss(logprobs: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
    mask = batch["loss_mask"]
    if batch["loss_name"] == "cross_entropy":
        return -(logprobs * mask).sum()
    ratio = torch.exp(logprobs - batch["sampling_logprobs"])
    return -(ratio * batch["advantages"] * mask).sum()


def apply_adam_params(engine, adam) -> None:
    optimizer = getattr(engine.optimizer, "optimizer", engine.optimizer)
    for group in optimizer.param_groups:
        group["lr"] = float(adam.learning_rate)
        group["betas"] = (float(adam.beta1), float(adam.beta2))
        group["eps"] = float(adam.eps)
        group["weight_decay"] = float(adam.weight_decay)
    engine.optimizer.clip_grad = float(adam.grad_clip_norm)


def optimizer_grad_norm(engine) -> float:
    value = getattr(engine.optimizer, "_global_grad_norm", 0.0)
    if torch.is_tensor(value):
        value = value.item()
    value = float(value or 0.0)
    return value if math.isfinite(value) else 0.0
