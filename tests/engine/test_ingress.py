import json

from tinker import types
from tinker.proto.request_conv import forward_backward_request_to_proto
from tinker.types.forward_backward_input import ForwardBackwardInput
from tinker.types.forward_backward_request import ForwardBackwardRequest

from lilo.engine import OperationKind
from lilo.engine.ingress import decode_forward_backward
from lilo.engine.operations import (
    parse_operation_payload,
    serialize_operation_payload,
)


def test_json_and_protobuf_normalize_to_the_same_forward_payload() -> None:
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={"target_tokens": [2, 3, 4]},
    )
    request = ForwardBackwardRequest(
        model_id="model-a",
        seq_id=7,
        forward_backward_input=ForwardBackwardInput(
            data=[datum],
            loss_fn="cross_entropy",
        ),
    )
    protobuf = forward_backward_request_to_proto(request).SerializeToString()
    json_body = json.dumps(
        {
            "model_id": "model-a",
            "seq_id": 7,
            "forward_backward_input": {
                "data": [
                    {
                        "model_input": {"chunks": [{"tokens": [1, 2, 3]}]},
                        "loss_fn_inputs": {"target_tokens": [2, 3, 4]},
                    }
                ],
                "loss_fn": "cross_entropy",
            },
        }
    ).encode()

    json_request = decode_forward_backward(json_body, "application/json")
    proto_request = decode_forward_backward(protobuf, "application/x-protobuf")

    assert (
        json_request[:3]
        == proto_request[:3]
        == (
            "model-a",
            7,
            OperationKind.FORWARD_BACKWARD,
        )
    )
    assert serialize_operation_payload(json_request[3]) == serialize_operation_payload(
        proto_request[3]
    )


def test_unknown_payload_fields_are_preserved_as_extensions() -> None:
    payload = parse_operation_payload(
        OperationKind.SAVE_WEIGHTS,
        {
            "path": "checkpoint",
            "future_option": {"enabled": True},
        },
    )

    assert payload.extensions == {"future_option": {"enabled": True}}
    assert serialize_operation_payload(payload)["extensions"] == payload.extensions
