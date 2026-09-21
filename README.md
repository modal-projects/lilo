# Lilo

Lilo is a Tinker SDK-compatible backend run on Modal. Trainers run `forward_backward` and `optim_step` calls, then publish updated weights to autoscaling sampling replicas managed by the [Stitch](https://github.com/modal-projects/stitch) protocol (hence the name!). Currently, Lilo supports single-tenant full-parameter training as well as multi-tenant LoRA training.

# Getting Started 

## Full-parameter training runs

For a dedicated full-parameter fine-tuning (FFT) run, use Python 3.12 and configure your Modal
credentials and `lilo-proxy` secret as described below. Then:

```python
import lilo
import tinker
from lilo.engines import qwen3_5_4b_full_64k

engine = qwen3_5_4b_full_64k()
with lilo.run(engine=engine) as (url, api_key):
    service = tinker.ServiceClient(base_url=url, api_key=api_key)
    training = lilo.create_full_training_client(service, engine.model)
    # Train and sample through the Tinker SDK here.
```

Our FFT path is *not* Tinker compatible, but roughly obeys the same abstractions. 

See [scoped runs](docs/scoped-runs.md) for recovery and custom engines,
and the [Codeforces example](examples/codeforces-codegolf/README.md) for a complete
training loop with sandbox judging and checkpoints.

## LoRA training runs

Supported Tinker SDK versions: `>=0.24.1,<0.26`. SDK 0.25 requires the updated
Lilo server to be deployed because it requires protobuf training and sampling responses.

The traditional Tinker path uses LoRA training, which is implemented via a multi-tenant Miles/Megatron backend in our system. Our LoRA path *is* Tinker-compatible out of the box on any of our supported models: 

```python
import os
import tinker

service = tinker.ServiceClient(
    base_url=os.environ["TINKER_BASE_URL"],
    api_key=os.environ["TINKER_API_KEY"],
)
training = service.create_lora_training_client(
    base_model="Qwen/Qwen3.5-9B-Base",
    rank=16,
)
# Train and sample through the Tinker SDK here.
```

## Shared deployment quick start

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

Set `your-environment` to an existing environment **before** creating secrets
so the secrets and deployment use the same environment.
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

Deploying the entire Tinker server can be done with a single modal deploy command: 

```bash
uv run modal deploy -m lilo.providers.modal.app
```

This deploys the control plane and bundled model definitions, then prints the
`server` URL to use in step 4. Reuse the deployment across training runs and
redeploy after updating Lilo.

Deploying the server doesn't allocate any GPUs; rather, this allocation for both the training and sampling sides are done on demand. See [cold starts and capacity configuration](docs/full-fine-tunes.md#performance-and-behavior-considerations)
before running a larger workload.

### 4. Clean up

After the script exits, session heartbeats stop and Lilo's periodic cleaner
reclaims idle training models and their latest sampler pools. Check that cleanup
has finished in the Modal dashboard or list apps with:

```bash
uv run modal app list
```

To tear down the deployment, stop its `lilo-fft-...` sampler apps, then `lilo`,
using `uv run modal app stop <app-id>`. Stopping `lilo` does not stop sampler apps.

## Next steps

Refer to the docs for design and for more advanced features when working with either the full-parameter or LoRA paths: 

Read [Working with Full Fine-Tunes](docs/full-fine-tunes.md) for full training,
or [Working with Multi-LoRA](docs/multi-lora.md) for shared Miles adapters, batch
submission, scheduling, and sampling.

and the [raw Tinker RL example](scripts/rl_example.py) for sampling and a toy
policy update. Copy examples you want to run into your project; repository
`scripts/` are not installed with the package.

The [W&B RL example](scripts/wandb_rl_example.py) extends it to a multi-step
loop that logs reward, response length, and Lilo's training metrics to Weights
& Biases from the client side; tinker-cookbook users can instead set
`wandb_project`/`wandb_name` on the cookbook `Config`.

See [Design](docs/design.md) for the control-plane, training-engine, and sampling
architecture.

See [Profiling](docs/profiling.md) for how to enable the `torch.profiler` trace of
a training step and read it in Perfetto.

See [Observability](docs/observability.md) for OTLP export to Datadog or a custom
destination, experiment labels, and the complete span/metric inventory.

## Validation

See [FFT validation](docs/validation.md) and [LoRA validation](docs/lora_validation.md)
for end-to-end training runs we've done with both parameterizations. The [Codeforces codegolf](examples/codeforces-codegolf/README.md) example provides a larger-scale e2e code-RL training run, which trains Qwen3.5-9B
with GRPO or TailRL advantages for correctness and short solutions. It includes
a sandboxed judge, checkpoint recovery, and commands to continue a checkpoint
with a different reward or advantage estimator, as well as pass@k and best-of-k evaluation. 
