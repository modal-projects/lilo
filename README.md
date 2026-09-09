# Lilo

Lilo is a Tinker SDK-compatible training and sampling infrastructure built on
Modal. Individual engine containers run `forward_backward` and `optim_step`
jobs and publish weights to autoscaling sampling infrastructure based on
[Stitch](https://github.com/modal-projects/stitch).

## Quick start

Install Lilo into your own Python project, deploy it once to Modal, then call
its API from your training scripts. The commands below work in Bash or Zsh.

If someone has already deployed Lilo for you, install the package in step 1,
then skip to step 4 with the server URL and Lilo API key they provide. API
clients do not need Modal deployment credentials or sampler proxy tokens.

### 1. Install into your project

With [uv](https://docs.astral.sh/uv/) installed:

```bash
uv init my-lilo-project
cd my-lilo-project
uv add 'lilo @ git+https://github.com/modal-projects/lilo.git'
```

### 2. Configure Modal and secrets once

Use a [Modal account](https://modal.com/docs/guide/workspaces) with permission to
deploy apps and create secrets in your chosen environment. Authenticate if you
have not already configured credentials for that workspace:

```bash
uv run modal token new
export MODAL_ENVIRONMENT=your-environment
uv run modal environment list
```

Replace `your-environment` with an existing environment name. Set it **before**
creating secrets so secrets and deployments all use the same environment.
For automation, existing `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` credentials can
be supplied instead of the interactive login.

There are three separate credentials:

| Credential | Purpose | Who needs it |
| --- | --- | --- |
| Modal API token / local profile | Manage Modal resources | Deployer |
| `TINKER_API_KEY` in the `lilo-api` secret | Authenticate calls to the Lilo API | Deployer and API clients |
| Proxy token in the `lilo-proxy` secret | Let Lilo reach protected sampler pools | Deployed control plane and trainers |

For a new deployment, generate a Lilo API key and store it in Modal:

```bash
export TINKER_API_KEY=$(uv run python -c 'import secrets; print(f"tml-lilo-{secrets.token_urlsafe(42)}")')
uv run modal secret create lilo-api \
  TINKER_API_KEY="$TINKER_API_KEY"
```

Keep this key in your secret manager for subsequent client sessions. Reuse the
existing key and secret when returning to an existing deployment.

Sampler pools use [Modal proxy authentication](https://modal.com/docs/guide/webhook-proxy-auth).
Create a proxy token and allow it in the deployment environment. If you deploy
with service-user credentials or lack permission to create workspace proxy
tokens, have a workspace owner or manager provision an allowed token first;
set `MODAL_PROXY_TOKEN_ID` and `MODAL_PROXY_TOKEN_SECRET` to that pair and skip
the token-creation block below.

```bash
read -r MODAL_PROXY_TOKEN_ID MODAL_PROXY_TOKEN_SECRET < <(uv run python -c '
import os
import modal
tokens = modal.Workspace.from_context().proxy_tokens
token = tokens.create()
tokens.allow(token.token_id, os.environ["MODAL_ENVIRONMENT"])
print(token.token_id, token.token_secret)
')
```

Store the token in the same environment:

```bash
uv run modal secret create lilo-proxy \
  MODAL_PROXY_TOKEN_ID="$MODAL_PROXY_TOKEN_ID" \
  MODAL_PROXY_TOKEN_SECRET="$MODAL_PROXY_TOKEN_SECRET"
```

### 3. Deploy the installed package

```bash
uv run modal deploy -m lilo.providers.modal.app
```

This deploys the control plane and bundled model definitions from the installed
package. A successful deployment prints a URL for the `server` web function.
Keep that URL for step 4. Redeploy when you intentionally update Lilo; deployment
is not required for every training run.

Training and sampling allocate GPUs on demand. The bundled `Qwen/Qwen3.5-4B`
definition used below has a four-H100 trainer and one H100 per sampling replica;
a tiny training batch still uses that trainer topology. Model assets may need
to download on first use. See [cold starts and capacity configuration](docs/full-fine-tunes.md#performance-and-behavior-considerations)
before running a larger workload.

### 4. Connect and run one SFT update

Set the server URL printed by deployment. In a new shell, also load the same
`TINKER_API_KEY` stored in `lilo-api`:

```bash
export TINKER_BASE_URL=https://your-modal-server-url
```

Save the following as `sft_smoke.py` in **your project**. It checks API access,
creates a full-training client, and performs one supervised next-token update.
It uses regular Tinker training calls after Lilo's client-creation helper.

```python
import os

import tinker
from tinker import types
from lilo.client import create_full_training_client

service = tinker.ServiceClient(
    base_url=os.environ["TINKER_BASE_URL"],
    api_key=os.environ["TINKER_API_KEY"],
)
print("Supported models:", service.get_server_capabilities().supported_models)
training = create_full_training_client(service, "Qwen/Qwen3.5-4B")
tokenizer = training.get_tokenizer()
tokens = tokenizer.encode("The capital of France is Paris.", add_special_tokens=True)

datum = types.Datum(
    model_input=types.ModelInput.from_ints(tokens[:-1]),
    loss_fn_inputs={
        "target_tokens": tokens[1:],
        "weights": [1.0] * (len(tokens) - 1),
    },
)
forward = training.forward_backward([datum], "cross_entropy")
optimizer = training.optim_step(types.AdamParams(learning_rate=1e-6))

print("Training metrics:", forward.result(timeout=3600).metrics)
print("Optimizer metrics:", optimizer.result(timeout=3600).metrics)
```

Run it with:

```bash
uv run python sft_smoke.py
```

Expect a supported-model list followed by training and optimizer metrics.
The first update can take several minutes for GPU allocation, model loading,
and compilation; it is not representative of steady-state step time. This is
a connectivity and training smoke test, not a model-quality evaluation. It does
not save a checkpoint; read [Working with Full Fine-Tunes](docs/full-fine-tunes.md)
for checkpointing, sampling, and longer runs.

### 5. Clean up

After the script exits, session heartbeats stop. Lilo's periodic cleaner reclaims
idle training models and their latest sampler pools; cleanup is not immediate.
Check the apps and running containers in your Modal dashboard or list apps with:

```bash
uv run modal app list
```

For immediate teardown of a disposable deployment, stop its separately deployed
`lilo-fft-...` sampler apps and then the `lilo` app. Use the exact app IDs shown
by the list command with `uv run modal app stop <app-id>`. Stopping
`lilo` alone does not stop separately deployed sampler apps. Do not stop a shared
deployment that other users are using. App shutdown leaves persisted Volumes
and secrets in place.

## Next steps

Read [Working with Full Fine-Tunes](docs/full-fine-tunes.md) before running a job.
See [Validation](docs/validation.md) for end-to-end Lilo and Miles training runs,
and the [raw Tinker RL example](scripts/rl_example.py) for sampling and a toy
policy update. Copy examples you want to run into your project; repository
`scripts/` are not installed with the package.

See [Design](docs/design.md) for the control-plane, training-engine, and sampling
architecture.
