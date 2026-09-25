"""Exercise the public helper through the real SDK and HTTP control plane."""

from types import SimpleNamespace

import tinker
from tinker import types

from lilo import sample_with_replay
from lilo.control_plane import ControlPlane, create_control_plane_app
from lilo.engine import OperationKind
from lilo.providers.local import (
    InMemoryKeyValueStore,
    LocalEnginePlatform,
    LocalSamplingTaskPlatform,
)
from lilo.replay import capture_replay
from tests.support import TinkerStubExecutor, serve


def test_sdk_replay_roundtrip():
    seen = []

    class Executor(TinkerStubExecutor):
        async def execute(self, model_id, kind, payload):
            if kind == OperationKind.FORWARD_BACKWARD:
                seen.append(payload)
            return await super().execute(model_id, kind, payload)

    async def sampler(task):
        assert task.payload["return_sampling_mask"] is True
        assert task.payload["return_routed_experts"] is False
        return {
            "type": "sample",
            "sequences": [
                {
                    "tokens": [3, 4],
                    "stop_reason": "length",
                    "logprobs": [-2.0, -3.0],
                    "replay": capture_replay(
                        {
                            "output_token_sampling_mask": [[3, 5], [4]],
                            "output_token_sampling_logprobs": [-0.3, 0.0],
                        },
                        [3, 4],
                        prompt_tokens=2,
                        temperature=0.7,
                        routes=False,
                        mask=True,
                    ),
                }
            ],
        }

    definition = SimpleNamespace(
        DEFINITION_ID="replay",
        MODEL_NAME="Qwen/Qwen3-8B",
        PARAMETERIZATION="lora",
        CATALOG_VISIBLE=True,
        MAX_CONTEXT_LENGTH=32768,
    )
    plane = ControlPlane(
        InMemoryKeyValueStore(),
        LocalEnginePlatform("replay", Executor),
        sampling_tasks=LocalSamplingTaskPlatform(sampler),
    )
    app = create_control_plane_app(
        plane, (definition,), api_key="tml-test-key", retrieve_window=5.0
    )
    with serve(app) as url:
        service = tinker.ServiceClient(base_url=url, api_key="tml-test-key")
        sampling = service.create_sampling_client(base_model=definition.MODEL_NAME)
        sequence = (
            sample_with_replay(
                sampling,
                types.ModelInput.from_ints([1, 2]),
                sampling_params=types.SamplingParams(
                    max_tokens=2, temperature=0.7, top_p=0.9
                ),
                return_sampling_mask=True,
            )
            .result(timeout=30)
            .sequences[0]
        )
        assert sequence.logprobs == [-2.0, -3.0]
        assert sequence.replay.sampling_logprobs == [-0.3, 0.0]
        training = service.create_lora_training_client(
            base_model=definition.MODEL_NAME, rank=16
        )
        datum = types.Datum(
            model_input=types.ModelInput.from_ints([1, 2, 3]),
            loss_fn_inputs={
                "target_tokens": [2, 3, 4],
                "advantages": [0.0, 1.0, 1.0],
                **sequence.replay.training_inputs(2),
            },
        )
        training.forward_backward(
            [datum],
            "ppo",
            loss_fn_config=sequence.replay.loss_fn_config(),
        ).result(timeout=30)
    assert len(seen) == 1
    inputs = seen[0].data[0].loss_fn_inputs
    assert list(inputs["sampling_mask_offsets"].data) == [0, 0, 2, 3]
    assert list(inputs["sampling_mask_ids"].data) == [3, 5, 4]
    assert seen[0].loss_fn_config["sampling_temperature"] == 0.7
