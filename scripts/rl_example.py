from __future__ import annotations

import os

import tinker
from tinker import types

from lilo.client import create_full_training_client

BASE_MODEL = "Qwen/Qwen3.5-4B"
TIMEOUT = 60 * 60


def main() -> None:
    service = tinker.ServiceClient(
        base_url=os.environ["TINKER_BASE_URL"],
        api_key=os.environ["TINKER_API_KEY"],
    )
    training = create_full_training_client(service, BASE_MODEL)
    tokenizer = training.get_tokenizer()

    prompt = tokenizer.encode(
        "What is 2 + 2? Answer with only the number.",
        add_special_tokens=True,
    )
    sampling = training.save_weights_and_get_sampling_client()
    sampled = sampling.sample(
        prompt=types.ModelInput.from_ints(prompt),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=16, temperature=1.0),
    ).result(timeout=TIMEOUT)
    response = sampled.sequences[0]
    response_tokens = list(response.tokens)
    response_logprobs = list(response.logprobs or ())
    if not response_tokens or len(response_logprobs) != len(response_tokens):
        raise RuntimeError("sampling returned invalid tokens or logprobs")
    response_text = tokenizer.decode(response_tokens)
    reward = 1.0 if "4" in response_text else -1.0

    prompt_targets = len(prompt) - 1
    datum = types.Datum(
        model_input=types.ModelInput.from_ints(prompt + response_tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": [0] * prompt_targets + response_tokens,
            "logprobs": [0.0] * prompt_targets + response_logprobs,
            "advantages": [0.0] * prompt_targets
            + [reward] * len(response_tokens),
        },
    )
    forward = training.forward_backward([datum], "importance_sampling")
    optimizer = training.optim_step(types.AdamParams(learning_rate=1e-5))

    print(f"response: {response_text!r}")
    print(f"reward: {reward}")
    print(f"training metrics: {forward.result(timeout=TIMEOUT).metrics}")
    print(f"optimizer metrics: {optimizer.result(timeout=TIMEOUT).metrics}")


if __name__ == "__main__":
    main()
