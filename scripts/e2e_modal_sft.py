import os

import modal
import tinker
from tinker import types

from lilo.providers.modal.app import app, server

BASE_MODEL = "Qwen/Qwen3-4B"
LORA_RANK = 32


def main() -> None:
    api_key = os.environ["TINKER_API_KEY"]

    with modal.enable_output():
        with app.run():
            base_url = server.get_web_url()
            if base_url is None:
                raise RuntimeError("Modal did not provide a control-plane URL")

            print(f"Tinker server: {base_url}")
            service = tinker.ServiceClient(base_url=base_url, api_key=api_key)

            capabilities = service.get_server_capabilities()
            supported_models = [
                model.model_name for model in capabilities.supported_models
            ]
            if supported_models != [BASE_MODEL]:
                raise RuntimeError(f"unexpected model catalog: {supported_models}")

            training = service.create_lora_training_client(
                base_model=BASE_MODEL,
                rank=LORA_RANK,
            )
            tokenizer = training.get_tokenizer()
            prompt_tokens = tokenizer.apply_chat_template(
                [{"role": "user", "content": "What is the capital of France?"}],
                tokenize=True,
                add_generation_prompt=True,
            )
            tokens = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": "What is the capital of France?"},
                    {
                        "role": "assistant",
                        "content": "The capital of France is Paris.",
                    },
                ],
                tokenize=True,
            )
            datum = types.Datum(
                model_input=types.ModelInput.from_ints(tokens[:-1]),
                loss_fn_inputs={
                    "target_tokens": tokens[1:],
                    "weights": [0.0] * (len(prompt_tokens) - 1)
                    + [1.0] * (len(tokens) - len(prompt_tokens)),
                },
            )

            forward = training.forward_backward([datum], "cross_entropy")
            optim = training.optim_step(types.AdamParams(learning_rate=1e-4))
            print("forward_backward:", forward.result(timeout=600).metrics)
            print("optim_step:", optim.result(timeout=600).metrics)


if __name__ == "__main__":
    main()
