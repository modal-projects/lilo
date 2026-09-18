# Multi-LoRA RL

Train six independent LoRA adapters on GSM8K using the Tinker SDK. Each client
samples answers, scores them, and updates its adapter. The clients run
concurrently and share the deployment's training and sampling resources.

## Run

Connect to a Lilo deployment with a Miles multi-LoRA definition for
`Qwen/Qwen3.5-9B-Base`. The bundled
[`qwen3_5_9b_base_miles_lora_16k` definition](../../src/lilo/providers/modal/definitions/qwen3_5_9b_base_miles_lora_16k.py)
supports six adapters and a 16,384-token context window. See the
[deployment quick start](../../README.md#shared-deployment-quick-start) and
[multi-LoRA guide](../../docs/multi-lora.md) for setup. Custom definitions need
rank 16 support and enough context for the prompt plus the generation limit.
Whether all six clients share one training engine depends on its available slots.

From the repository root:

```bash
export TINKER_BASE_URL=https://your-modal-server-url
export TINKER_API_KEY=...
uv run examples/multi-lora/train.py
```

`uv` installs the script's dependencies. The script downloads GSM8K and the model
tokenizer on its first run. It uses ordinary Tinker clients and needs no Lilo
imports or Modal credentials.

Defaults are six clients, three steps per client, eight answers per prompt,
a 4,096-token generation limit, rank 16, and learning rate `1e-5`.
To change the run size:

```bash
uv run examples/multi-lora/train.py --clients 4 --steps 10 --group-size 16
```

Use `--base-model` to select another model supported by your deployment and
`--max-tokens` to change the generation limit.

## Training loop

Each client uses a separate slice of the training split and publishes its current
adapter before sampling. The reward is 1 when the last number in the response
matches the GSM8K answer, and 0 otherwise. This simple grader expects the model
to put its final answer last.

The loop subtracts the group's mean reward to get each response's advantage,
then applies an `importance_sampling` update using the sampled logprobs.
Prompt tokens have zero advantage. A group with identical rewards produces
zero advantages and contributes no policy gradient.

Each completed step prints the client ID, mean reward, mean generated token
count, and elapsed time for publication, sampling, and training. The example
keeps adapters in the live session; it does not save resumable checkpoints.
