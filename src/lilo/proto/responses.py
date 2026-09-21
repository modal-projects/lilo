"""Encode completed results for SDKs that request protobuf responses."""

from __future__ import annotations

from typing import Any

import numpy as np
from tinker.proto import tinker_public_pb2 as pb


def encode_result(result: dict[str, Any]) -> bytes | None:
    if "loss_fn_outputs" in result:
        message = pb.ForwardBackwardOutput(
            loss_fn_output_type=result["loss_fn_output_type"],
            metrics=result.get("metrics", {}),
        )
        # One record per datum supports different trailing shapes between rows.
        # num_datums also preserves rows whose output fields are all empty.
        for datum in result["loss_fn_outputs"]:
            record = message.loss_fn_outputs.add(num_datums=1)
            for name, tensor in datum.items():
                dtype, proto_dtype = {
                    "float32": ("<f4", pb.DTYPE_FLOAT32),
                    "int64": ("<i8", pb.DTYPE_INT64),
                }[tensor["dtype"]]
                values = np.asarray(tensor["data"], dtype=dtype)
                shape = tensor.get("shape") or [values.size]
                if tensor.get("sparse_crow_indices") is not None:
                    # BatchedTensor has no CSR representation.
                    dense = np.zeros(shape, dtype=dtype)
                    rows = tensor["sparse_crow_indices"]
                    cols = tensor["sparse_col_indices"]
                    for row, (start, end) in enumerate(zip(rows, rows[1:])):
                        np.add.at(dense[row], cols[start:end], values[start:end])
                    values = dense
                data = values.tobytes()
                record.fields[name].CopyFrom(
                    pb.BatchedTensor(
                        data=data,
                        offsets=np.asarray([0, len(data)], dtype="<i8").tobytes(),
                        dtype=proto_dtype,
                        trailing_shape=shape[1:],
                    )
                )
        return message.SerializeToString()
    if "sequences" in result:
        message = pb.SampleResponse(
            prompt_cache_hit_tokens=result.get("prompt_cache_hit_tokens", 0),
        )
        for sequence in result["sequences"]:
            item = message.sequences.add(
                stop_reason={
                    "stop": pb.STOP_REASON_STOP,
                    "length": pb.STOP_REASON_LENGTH,
                }[sequence["stop_reason"]],
                tokens=np.asarray(sequence["tokens"], dtype="<i4").tobytes(),
            )
            if sequence.get("logprobs") is not None:
                item.logprobs = np.asarray(sequence["logprobs"], dtype="<f4").tobytes()
        if result.get("prompt_logprobs") is not None:
            message.prompt_logprobs = np.asarray(
                result["prompt_logprobs"], dtype="<f4"
            ).tobytes()
        if result.get("topk_prompt_logprobs") is not None:
            rows = result["topk_prompt_logprobs"]
            k = max((len(row or ()) for row in rows), default=0)
            tokens = np.zeros((len(rows), k), dtype="<i4")
            logprobs = np.full((len(rows), k), -99999.0, dtype="<f4")
            for i, row in enumerate(rows):
                for j, (token, logprob) in enumerate(row or ()):
                    tokens[i, j], logprobs[i, j] = token, logprob
            message.topk_prompt_logprobs.CopyFrom(
                pb.TopkPromptLogprobs(
                    token_ids=tokens.tobytes(),
                    logprobs=logprobs.tobytes(),
                    prompt_length=len(rows),
                    k=k,
                )
            )
        return message.SerializeToString()
    return None
