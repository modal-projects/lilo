import numpy as np
import pytest
from tinker.proto.response_conv import (
    deserialize_forward_backward_output,
    deserialize_sample_response,
)

from lilo.proto.responses import encode_result


def test_forward_results_preserve_ragged_shapes_dtypes_and_metrics():
    result = {
        "loss_fn_output_type": "ArrayRecord",
        "metrics": {"loss:sum": 2.5},
        "loss_fn_outputs": [
            {
                "logprobs": {
                    "dtype": "float32",
                    "shape": [2, 2],
                    "data": [-1.0, -2.0, -3.0, -4.0],
                },
                "tokens": {"dtype": "int64", "shape": [2], "data": [1, 2**40]},
            },
            {
                "logprobs": {
                    "dtype": "float32",
                    "shape": [1, 3],
                    "data": [-5.0, -6.0, -7.0],
                },
                "tokens": {"dtype": "int64", "shape": [1], "data": [3]},
            },
        ],
    }
    decoded = deserialize_forward_backward_output(encode_result(result))
    assert decoded.metrics == result["metrics"]
    for expected, actual in zip(result["loss_fn_outputs"], decoded.loss_fn_outputs):
        for key, tensor in expected.items():
            assert actual[key].shape == tensor["shape"]
            assert actual[key].dtype == tensor["dtype"]
            assert actual[key].data == tensor["data"]


@pytest.mark.parametrize("rows", [[], [{}, {}, {}]])
def test_empty_forward_outputs_preserve_row_count(rows):
    decoded = deserialize_forward_backward_output(
        encode_result(
            {
                "loss_fn_output_type": "ArrayRecord",
                "loss_fn_outputs": rows,
                "metrics": {},
            }
        )
    )
    assert decoded.loss_fn_outputs == rows


def test_sampling_preserves_logprobs_cache_hits_and_topk():
    decoded = deserialize_sample_response(
        encode_result(
            {
                "sequences": [
                    {
                        "tokens": [12, 13],
                        "logprobs": [-0.25, -0.5],
                        "stop_reason": "stop",
                    },
                    {"tokens": [14], "stop_reason": "length"},
                ],
                "prompt_logprobs": [None, -0.5],
                "topk_prompt_logprobs": [None, [(12, -0.25), (13, -0.5)]],
                "prompt_cache_hit_tokens": 8,
            }
        )
    )
    assert decoded.sequences[0].tokens == [12, 13]
    assert decoded.sequences[0].logprobs == [-0.25, -0.5]
    assert decoded.sequences[0].stop_reason == "stop"
    assert decoded.sequences[1].logprobs is None
    assert decoded.sequences[1].stop_reason == "length"
    assert decoded.prompt_logprobs == [None, -0.5]
    assert decoded.prompt_cache_hit_tokens == 8
    np.testing.assert_array_equal(
        decoded.topk_prompt_logprobs_np.token_ids, [[0, 0], [12, 13]]
    )
    np.testing.assert_array_equal(
        decoded.topk_prompt_logprobs_np.logprobs, [[-99999.0, -99999.0], [-0.25, -0.5]]
    )


def test_sparse_forward_output_is_encoded_as_dense():
    decoded = deserialize_forward_backward_output(
        encode_result(
            {
                "loss_fn_output_type": "ArrayRecord",
                "metrics": {},
                "loss_fn_outputs": [
                    {
                        "values": {
                            "dtype": "float32",
                            "shape": [2, 3],
                            "data": [1.0, 2.0],
                            "sparse_crow_indices": [0, 1, 2],
                            "sparse_col_indices": [1, 2],
                        }
                    }
                ],
            }
        )
    )
    assert decoded.loss_fn_outputs[0]["values"].data == [0.0, 1.0, 0.0, 0.0, 0.0, 2.0]
    assert decoded.loss_fn_outputs[0]["values"].shape == [2, 3]


def test_non_tensor_results_remain_json():
    assert encode_result({"metrics": {"lr": 0.01}}) is None
    assert encode_result({"path": "tinker://model/weights/checkpoint"}) is None
