from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tinker import ForwardBackwardOutput, TensorData

from lilo.backends.contract import ForwardBatch

SUPPORTED_LOSSES = frozenset(
    {"cross_entropy", "importance_sampling", "ppo", "cispo", "dro"}
)
_RL_LOSSES = SUPPORTED_LOSSES - {"cross_entropy"}


@dataclass(frozen=True, slots=True)
class PreparedBatch:
    slot_rows: tuple[tuple[int, dict[str, Any]], ...]
    locations: tuple[tuple[int, int], ...]
    target_lengths: tuple[int, ...]


def pad_slot_rows(
    slot_rows: tuple[tuple[int, dict[str, Any]], ...],
    multiple: int,
) -> tuple[tuple[int, dict[str, Any]], ...]:
    """Pad to a multiple of multiple with zero-weight rows for Miles DP sharding."""
    if multiple <= 1 or len(slot_rows) % multiple == 0:
        return slot_rows
    slot, last = slot_rows[-1]
    pad: dict[str, Any] = {
        "tokens": last["tokens"][:2],
        "target_len": 1,
        "target_tokens": last["tokens"][1:2],
    }
    if "weights" in last:
        pad["weights"] = [0.0]
    if "advantages" in last:
        pad["advantages"] = [0.0]
    if "sampling_logprobs" in last:
        pad["sampling_logprobs"] = [0.0]
    n_pad = -len(slot_rows) % multiple
    return (*slot_rows, *((slot, dict(pad)) for _ in range(n_pad)))


def prepare_batch(
    batch: ForwardBatch,
    slots: dict[str, int],
    sequence_alignment: int = 1,
) -> PreparedBatch:
    if batch.loss_fn not in SUPPORTED_LOSSES:
        raise ValueError(f"Miles does not support loss {batch.loss_fn!r}")
    if sequence_alignment < 1:
        raise ValueError("sequence_alignment must be at least 1")
    entries: list[tuple[int, int, int, dict[str, Any], int]] = []
    for item_index, item in enumerate(batch.items):
        if item.model_id not in slots:
            raise ValueError(f"model {item.model_id} is not loaded")
        if not item.data:
            raise ValueError(f"forward_backward has no data for model {item.model_id}")
        for datum_index, datum in enumerate(item.data):
            row = _datum_row(datum, batch.loss_fn, datum_index)
            target_length = int(row["target_len"])
            _align_row(row, sequence_alignment)
            entries.append(
                (slots[item.model_id], item_index, datum_index, row, target_length)
            )

    entries.sort(key=lambda entry: entry[0])
    return PreparedBatch(
        slot_rows=tuple((slot, row) for slot, _, _, row, _ in entries),
        locations=tuple(
            (item_index, datum_index) for _, item_index, datum_index, _, _ in entries
        ),
        target_lengths=tuple(length for *_, length in entries),
    )


def _align_row(row: dict[str, Any], multiple: int) -> None:
    """Pad each sequence to a multiple of ``2 * cp * tp`` so context-parallel
    chunking and tensor-parallel sharding both divide it evenly."""

    pad = -len(row["tokens"]) % multiple
    if pad == 0:
        return
    filler = row["tokens"][-1]
    row["tokens"] = [*row["tokens"], *([filler] * pad)]
    row["target_tokens"] = [*row["target_tokens"], *([filler] * pad)]
    row["target_len"] = len(row["target_tokens"])
    for key in ("weights", "advantages", "sampling_logprobs"):
        values = row.get(key)
        if values is not None:
            row[key] = [*values, *([0.0] * pad)]


def build_outputs(
    batch: ForwardBatch,
    prepared: PreparedBatch,
    raw_outputs: list[dict[str, Any]],
) -> tuple[ForwardBackwardOutput, ...]:
    if len(raw_outputs) != len(prepared.locations):
        raise RuntimeError(
            f"Miles returned {len(raw_outputs)} datum outputs for "
            f"{len(prepared.locations)} inputs"
        )
    grouped: list[list[dict[str, Any] | None]] = [
        [None] * len(item.data) for item in batch.items
    ]
    for location, output, target_length in zip(
        prepared.locations, raw_outputs, prepared.target_lengths, strict=True
    ):
        item_index, datum_index = location
        logprobs = [float(value) for value in output["logprobs"]][:target_length]
        grouped[item_index][datum_index] = {
            "loss": float(output["loss"]),
            "logprobs": logprobs,
        }

    results = []
    for records in grouped:
        if any(record is None for record in records):
            raise RuntimeError("Miles omitted one or more datum outputs")
        complete = [record for record in records if record is not None]
        loss = sum(record["loss"] for record in complete)
        tokens = sum(len(record["logprobs"]) for record in complete)
        results.append(
            ForwardBackwardOutput(
                loss_fn_output_type="ArrayRecord",
                loss_fn_outputs=[
                    {
                        "loss:sum": TensorData(
                            data=[record["loss"]],
                            dtype="float32",
                            shape=[1],
                        ),
                        "logprobs": TensorData(
                            data=record["logprobs"],
                            dtype="float32",
                            shape=[len(record["logprobs"])],
                        ),
                    }
                    for record in complete
                ],
                metrics={
                    "loss:sum": loss,
                    "loss:mean": loss / max(tokens, 1),
                    "tokens:sum": float(tokens),
                    "n_sequences:sum": float(len(complete)),
                    "response_length:mean": tokens / max(len(complete), 1),
                },
            )
        )
    return tuple(results)


def _datum_row(datum, loss_fn: str, datum_index: int) -> dict[str, Any]:
    inputs = datum.loss_fn_inputs
    input_tokens = [int(token) for token in datum.model_input.to_ints()]
    if not input_tokens:
        raise ValueError(f"datum {datum_index}: model_input cannot be empty")
    target = inputs.get("target_tokens")
    if target is None:
        raise ValueError(f"datum {datum_index}: target_tokens is required")
    targets = [int(token) for token in _tensor_values(target)]
    if len(targets) != len(input_tokens):
        raise ValueError(
            f"datum {datum_index}: target_tokens length {len(targets)} "
            f"does not match model_input length {len(input_tokens)}"
        )
    if targets[:-1] != input_tokens[1:]:
        raise ValueError(
            f"datum {datum_index}: target_tokens must be model_input shifted by one"
        )

    row: dict[str, Any] = {
        "tokens": [*input_tokens, targets[-1]],
        "target_len": len(targets),
        "target_tokens": targets,
    }
    # Miles derives batch fields from its first datum. Forward only the fields
    # consumed by this loss so optional, unrelated inputs on one client cannot
    # cause missing-key failures when its rows are combined with another client.
    fields = (
        (("weights", "weights"),)
        if loss_fn == "cross_entropy"
        else (("advantages", "advantages"), ("logprobs", "sampling_logprobs"))
    )
    for source, destination in fields:
        value = inputs.get(source)
        if value is not None:
            values = [float(item) for item in _tensor_values(value)]
            if len(values) != len(targets):
                raise ValueError(
                    f"datum {datum_index}: {source} length {len(values)} "
                    f"does not match target_tokens length {len(targets)}"
                )
            row[destination] = values

    if loss_fn == "cross_entropy":
        row.setdefault("weights", [1.0] * len(targets))
    elif loss_fn in _RL_LOSSES:
        missing = [
            name for name in ("advantages", "sampling_logprobs") if name not in row
        ]
        if missing:
            raise ValueError(
                f"datum {datum_index}: {', '.join(missing)} required for {loss_fn}"
            )
    return row


def _tensor_values(tensor) -> list[Any]:
    values = list(tensor.data)
    crow = tensor.sparse_crow_indices
    if crow is None:
        return values
    shape = list(tensor.shape)
    columns = tensor.sparse_col_indices
    if len(shape) != 1 or columns is None or list(crow) != [0, len(values)]:
        raise ValueError("only one-dimensional CSR TensorData is supported")
    dense: list[Any] = [0] * shape[0]
    for column, value in zip(columns, values, strict=True):
        dense[int(column)] = value
    return dense
