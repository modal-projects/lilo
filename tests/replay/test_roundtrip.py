import asyncio
import base64
import json
import struct

import httpx
import pytest
from tinker import types
from tinker.lib._pydantic_conv import deserialize_json_response, to_pydantic_request
from tinker.proto.request_conv import forward_backward_request_to_proto

from lilo.backends.miles_runtime.data import _datum_row, pad_slot_rows
from lilo.backends.miles_runtime.replay_data import add_replay_to_train_data
from lilo.engine.ingress import decode_forward_backward
from lilo.inference.sampling import sample_task
from lilo.replay import ReplaySampleResponse, SamplingReplay, capture_replay


def captured():
    routes = list(range(8))  # four input positions, two layers, top-1
    return {
        "finish_reason": "length",
        "output_token_logprobs": [[-2.0, 5], [-3.0, 6]],
        "output_token_sampling_logprobs": [-0.2, -0.3],
        "output_token_sampling_mask": [[5, 7], [6, 8, 9]],
        "routed_experts": base64.b64encode(struct.pack("<8i", *routes)).decode(),
    }


def sample_response():
    sent = []

    def handle(req):
        sent.append(json.loads(req.content))
        return httpx.Response(200, json={"meta_info": captured()})

    result = asyncio.run(
        sample_task(
            {
                "request_id": "r",
                "payload": {
                    "prompt": {
                        "chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]
                    },
                    "sampling_params": {
                        "max_tokens": 2,
                        "temperature": 0.7,
                        "top_p": 0.9,
                    },
                    "num_samples": 2,
                    "return_routed_experts": True,
                    "return_sampling_mask": True,
                },
            },
            "http://sampler",
            transport=httpx.MockTransport(handle),
        )
    )
    assert len(sent) == 2
    assert all(
        r["return_routed_experts"]
        and r["return_sampling_mask"]
        and r["routed_experts_start_len"] == 0
        for r in sent
    )
    return deserialize_json_response(result, ReplaySampleResponse)


@pytest.mark.parametrize("encoding", ["json", "protobuf"])
def test_sampler_sdk_training_roundtrip(encoding):
    response = sample_response()
    sequence = response.sequences[0]
    assert sequence.logprobs == [-2.0, -3.0]  # Preserve the ordinary API's meaning.
    replay = sequence.replay
    inputs = replay.training_inputs(2, num_layers=2, experts_per_token=1)
    assert inputs["logprobs"].data == pytest.approx([0.0, 0.0, -0.2, -0.3])
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2, 3, 5]),
        loss_fn_inputs={
            "target_tokens": [2, 3, 5, 6],
            "advantages": [0.0, 0.0, 1.0, 1.0],
            **inputs,
        },
    )
    request = types.ForwardBackwardRequest(
        model_id="m",
        seq_id=1,
        forward_backward_input=types.ForwardBackwardInput(
            data=[datum], loss_fn="ppo", loss_fn_config=replay.loss_fn_config()
        ),
    )
    if encoding == "json":
        body = to_pydantic_request(request).model_dump_json().encode()
        content_type = "application/json"
    else:
        body = forward_backward_request_to_proto(request).SerializeToString()
        content_type = "application/x-protobuf"
    _, _, _, payload = decode_forward_backward(body, content_type)
    assert payload.loss_fn_config["sampling_temperature"] == 0.7
    row = _datum_row(payload.data[0], "ppo", 0)
    assert row["sampling_mask_offsets"] == [0, 0, 0, 2, 5]
    assert row["routed_experts"]["shape"] == [4, 2, 1]
    # Exercise zero-loss DP padding, not only a single-rank transport.
    rows = pad_slot_rows(((0, row),), 2)
    batch = {}
    add_replay_to_train_data(batch, [r for _, r in rows])
    assert batch["rollout_routed_experts"][0].tolist() == [
        [[0], [1]],
        [[2], [3]],
        [[4], [5]],
        [[6], [7]],
    ]
    assert batch["rollout_routed_experts"][1].shape == (1, 2, 1)
    assert batch["rollout_sampling_mask_offsets"][0].tolist() == [0, 0, 0, 2, 5]
    assert batch["rollout_sampling_mask_offsets"][1].tolist() == [0, 0]


@pytest.mark.parametrize(
    "missing",
    ["routed_experts", "output_token_sampling_mask", "output_token_sampling_logprobs"],
)
def test_missing_requested_replay_is_an_error(missing):
    meta = captured()
    del meta[missing]
    with pytest.raises(ValueError):
        capture_replay(
            meta, [5, 6], prompt_tokens=3, temperature=1.0, routes=True, mask=True
        )


@pytest.mark.parametrize("support", [[], [7], [5, 5], [5, -1], [5, True]])
def test_invalid_support_is_rejected(support):
    meta = captured()
    meta["output_token_sampling_mask"][0] = support
    with pytest.raises(ValueError):
        capture_replay(
            meta, [5, 6], prompt_tokens=3, temperature=1.0, routes=False, mask=True
        )


def test_stop_boundary_extra_router_row_is_trimmed():
    raw = base64.b64encode(struct.pack("<10i", *range(10))).decode()
    replay = SamplingReplay(prompt_tokens=3, temperature=1.0, routed_experts=raw)
    assert (
        len(
            replay.training_inputs(2, num_layers=2, experts_per_token=1)[
                "routed_experts"
            ].data
        )
        == 8
    )
    with pytest.raises(ValueError, match="shape"):
        replay.training_inputs(2, num_layers=3, experts_per_token=1)


def test_mixed_mask_and_ordinary_datums_keep_unmasked_rows():
    replay = SamplingReplay(
        **capture_replay(
            captured(),
            [5, 6],
            prompt_tokens=3,
            temperature=1.0,
            routes=False,
            mask=True,
        )
    )
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2, 3, 5]),
        loss_fn_inputs={
            "target_tokens": [2, 3, 5, 6],
            "advantages": [0.0, 0.0, 1.0, 1.0],
            **replay.training_inputs(2),
        },
    )
    row = _datum_row(datum, "ppo", 0)
    plain = {"target_len": 2}
    batch = {}
    add_replay_to_train_data(batch, [row, plain])
    assert batch["rollout_sampling_mask_offsets"][1].tolist() == [0, 0, 0]


@pytest.mark.parametrize(
    "offsets", [[0, 0, 0, 3, 2], [1, 1, 1, 2, 5], [0, 0, 2, 5], [0, 0, 0, 2, 4]]
)
def test_invalid_training_offsets_rejected(offsets):
    replay = SamplingReplay(
        **capture_replay(
            captured(),
            [5, 6],
            prompt_tokens=3,
            temperature=1.0,
            routes=False,
            mask=True,
        )
    )
    inputs = replay.training_inputs(2)
    inputs["sampling_mask_offsets"] = types.TensorData(
        data=offsets, dtype="int64", shape=[len(offsets)]
    )
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2, 3, 5]),
        loss_fn_inputs={
            "target_tokens": [2, 3, 5, 6],
            "advantages": [0.0, 0.0, 1.0, 1.0],
            **inputs,
        },
    )
    with pytest.raises(ValueError, match="offsets"):
        _datum_row(datum, "ppo", 0)
