from __future__ import annotations

import base64
import json
import struct
from collections.abc import Mapping

from google.protobuf.json_format import MessageToDict
from tinker import ModelInput
from tinker.types._pydantic_types.datum import Datum
from tinker.types._pydantic_types.tensor_data import TensorData
from tinker.types.forward_backward_input import ForwardBackwardInput

from lilo.proto import tinker_public_pb2

from .api import OperationKind
from .operations import OperationPayload, parse_operation_payload

_ENVELOPE_FIELDS = {"model_id", "seq_id", "type"}
_PROTO_DTYPES = {
    "DTYPE_FLOAT32": ("f", "float32"),
    "DTYPE_INT64": ("q", "int64"),
}


def decode_forward_backward(
    body: bytes,
    content_type: str,
) -> tuple[str, int, OperationKind, ForwardBackwardInput]:
    if content_type.startswith("application/x-protobuf"):
        message = tinker_public_pb2.ForwardBackwardRequest()
        message.ParseFromString(body)
        raw = MessageToDict(message, preserving_proto_field_name=True)
        kind = (
            OperationKind.FORWARD
            if message.forward_only
            else OperationKind.FORWARD_BACKWARD
        )
        payload = ForwardBackwardInput(
            data=[_decode_proto_datum(item) for item in raw.get("data", ())],
            loss_fn=raw.get("loss_fn"),
            loss_fn_config=raw.get("loss_fn_config") or None,
        )
        model_id, seq_id = _identity(raw, kind)
        return model_id, seq_id, kind, payload

    request = json.loads(body)
    nested = request.pop("forward_backward_input", None)
    if not isinstance(nested, Mapping):
        raise ValueError("forward_backward_input must be an object")
    request.update(nested)
    model_id, seq_id = _identity(request, OperationKind.FORWARD_BACKWARD)
    payload = parse_operation_payload(
        OperationKind.FORWARD_BACKWARD,
        _payload(request),
    )
    assert isinstance(payload, ForwardBackwardInput)
    return model_id, seq_id, OperationKind.FORWARD_BACKWARD, payload


def decode_json_operation(
    kind: OperationKind,
    request: Mapping[str, object],
) -> tuple[str, int, OperationPayload]:
    raw = dict(request)
    if kind == OperationKind.FORWARD:
        nested = raw.pop("forward_input", None)
        if not isinstance(nested, Mapping):
            raise ValueError("forward_input must be an object")
        raw.update(nested)
    model_id, seq_id = _identity(raw, kind)
    return model_id, seq_id, parse_operation_payload(kind, _payload(raw))


def _identity(
    request: Mapping[str, object],
    kind: OperationKind,
) -> tuple[str, int]:
    model_id = request.get("model_id")
    seq_id = request.get("seq_id")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError(f"{kind.value} requires model_id and seq_id")
    if isinstance(seq_id, bool) or not isinstance(seq_id, int) or seq_id <= 0:
        raise ValueError(f"{kind.value} requires model_id and seq_id")
    return model_id, seq_id


def _payload(request: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in request.items() if key not in _ENVELOPE_FIELDS}


def _decode_proto_datum(value: object) -> Datum:
    if not isinstance(value, Mapping):
        raise ValueError("protobuf datum must be an object")
    raw_inputs = value.get("loss_fn_inputs") or {}
    if not isinstance(raw_inputs, Mapping):
        raise ValueError("protobuf loss_fn_inputs must be an object")
    return Datum(
        model_input=_decode_proto_model_input(value.get("model_input")),
        loss_fn_inputs={
            str(key): _decode_proto_tensor(tensor) for key, tensor in raw_inputs.items()
        },
    )


def _decode_proto_model_input(value: object) -> ModelInput:
    if not isinstance(value, list):
        raise ValueError("protobuf model_input must be a list")
    chunks = []
    for chunk in value:
        if not isinstance(chunk, Mapping):
            raise ValueError("protobuf model input chunk must be an object")
        if "encoded_text" in chunk:
            encoded = chunk["encoded_text"]
            if not isinstance(encoded, Mapping):
                raise ValueError("invalid encoded text chunk")
            raw = base64.b64decode(encoded["tokens"])
            if len(raw) % 4:
                raise ValueError("encoded text tokens have invalid byte length")
            chunks.append(
                {
                    "type": "encoded_text",
                    "tokens": [token for (token,) in struct.iter_unpack("<i", raw)],
                }
            )
        elif "image" in chunk:
            chunks.append({"type": "image", **dict(chunk["image"])})
        elif "dmel" in chunk:
            chunks.append({"type": "dmel", **dict(chunk["dmel"])})
        else:
            raise ValueError("unsupported protobuf model input chunk")
    return ModelInput.model_validate({"chunks": chunks})


def _decode_proto_tensor(value: object) -> TensorData:
    if not isinstance(value, Mapping):
        raise ValueError("protobuf tensor must be an object")
    dtype_value = str(value.get("dtype"))
    try:
        item_format, dtype = _PROTO_DTYPES[dtype_value]
    except KeyError:
        raise ValueError(f"unsupported protobuf tensor dtype: {dtype_value}") from None
    shape = [int(item) for item in value.get("shape") or ()]
    sparse = value.get("sparse_csr")
    if sparse is not None:
        if not isinstance(sparse, Mapping):
            raise ValueError("invalid protobuf sparse tensor")
        return TensorData(
            data=_decode_values(sparse.get("values"), item_format),
            dtype=dtype,
            shape=shape,
            sparse_crow_indices=_decode_values(
                sparse.get("crow_indices"),
                "q",
            ),
            sparse_col_indices=_decode_values(
                sparse.get("col_indices"),
                "q",
            ),
        )
    return TensorData(
        data=_decode_values(value.get("dense"), item_format),
        dtype=dtype,
        shape=shape,
    )


def _decode_values(value: object, item_format: str) -> list[int | float]:
    if not isinstance(value, str):
        raise ValueError("protobuf tensor data must be base64")
    raw = base64.b64decode(value)
    item_size = struct.calcsize(f"<{item_format}")
    if len(raw) % item_size:
        raise ValueError("protobuf tensor data has invalid byte length")
    return [item for (item,) in struct.iter_unpack(f"<{item_format}", raw)]
