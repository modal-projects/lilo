# Lilo

Lilo is a Tinker SDK-compatible training and sampling infrastructure built on
Modal. Individual engine containers run `forward_backward` and `optim_step`
jobs and publish weights to autoscaling sampling infrastructure based on
[Stitch](https://github.com/modal-projects/stitch).

## Quick start

Prerequisites:

- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- A configured Modal account

Install dependencies and configure the Modal secrets:

```bash
uv sync

export TINKER_API_KEY=$(python -c "import secrets; print(f'tml-lilo-{secrets.token_urlsafe(42)}')")
uv run modal secret create lilo-api \
  TINKER_API_KEY="$TINKER_API_KEY"
```

Sampler pools are served behind proxy auth. Create a proxy token,
allow it for the deployment environment, and store it in the `lilo-proxy`
secret so the control plane and trainers can reach the pools:

```bash
export MODAL_ENVIRONMENT=your-environment
read -r MODAL_PROXY_TOKEN_ID MODAL_PROXY_TOKEN_SECRET < <(uv run python -c '
import os
import modal
tokens = modal.Workspace.from_context().proxy_tokens
token = tokens.create()
tokens.allow(token.token_id, os.environ["MODAL_ENVIRONMENT"])
print(token.token_id, token.token_secret)
')
uv run modal secret create lilo-proxy \
  MODAL_PROXY_TOKEN_ID="$MODAL_PROXY_TOKEN_ID" \
  MODAL_PROXY_TOKEN_SECRET="$MODAL_PROXY_TOKEN_SECRET"
```

Deploy the control plane and model definitions:

```bash
LILO_TRAINER_MAX_CONTAINERS=5 \
  uv run modal deploy -m lilo.providers.modal.app
```

Copy the server URL printed by Modal, then run an FFT Tinker script:

```bash
export TINKER_BASE_URL=https://your-modal-server-url
uv run python scripts/rl_example.py
```

The example uses raw Tinker calls to sample one response, assign a toy reward,
and perform one policy update. It does not use Tinker Cookbook.

In a self-serve scenario, the containers lazily cold-start upon the first calls
to `sample()` and `forward_backward()`, so the first step in your training loop
may take unusually long.

After creating an FFT client with Lilo's helper, training uses the regular
Tinker API:

```python
import os

import tinker
from tinker import types
from lilo.client import create_full_training_client

service = tinker.ServiceClient(
    base_url=os.environ["TINKER_BASE_URL"],
    api_key=os.environ["TINKER_API_KEY"],
)
training = create_full_training_client(
    service,
    "Qwen/Qwen3.5-4B",
)

datum = types.Datum(
    model_input=types.ModelInput.from_ints([151644, 872, 198]),
    loss_fn_inputs={
        "target_tokens": [872, 198, 151643],
        "weights": [1.0, 1.0, 1.0],
    },
)
forward = training.forward_backward([datum], "cross_entropy")
optimizer = training.optim_step(types.AdamParams(learning_rate=1e-4))

print(forward.result(timeout=600).metrics)
print(optimizer.result(timeout=600).metrics)
```

See [Validation](docs/validation.md) for end-to-end Lilo and Miles training
runs. Read [Working with Full Fine-Tunes](docs/full-fine-tunes.md) before
running a job. See [Design](docs/design.md) for the control-plane,
training-engine, and sampling architecture.
