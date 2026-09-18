# Lilo

Lilo runs model training and sampling on Modal through the Tinker SDK. Trainers
run `forward_backward` and `optim_step`, then publish updated weights to sampling
replicas managed by [Stitch](https://github.com/modal-projects/stitch).

## Scoped training runs

For a dedicated full-training run, use Python 3.12 and configure your Modal
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

`lilo.run` creates an API endpoint, API key, trainer, and sampling apps. Leaving
the context stops those resources but keeps completed checkpoints. It does not
save a checkpoint automatically. Each scope allows one active training model;
other processes can connect while the process holding the context stays alive.
This path does not require deploying the shared API or creating a `lilo-api`
secret. See [scoped runs](docs/scoped-runs.md) for recovery and custom engines,
and the [Codeforces example](examples/codeforces-codegolf/README.md) for a complete
training loop with sandbox judging and checkpoints.

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

```bash
uv run modal deploy -m lilo.providers.modal.app
```

This deploys the control plane and bundled model definitions, then prints the
`server` URL to use in step 4. Reuse the deployment across training runs and
redeploy after updating Lilo.

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
and compilation. This smoke test checks connectivity and one training update.
To save checkpoints, sample, or run longer jobs, see
[Working with Full Fine-Tunes](docs/full-fine-tunes.md).

### 5. Clean up

After the script exits, session heartbeats stop and Lilo's periodic cleaner
reclaims idle training models and their latest sampler pools. Check that cleanup
has finished in the Modal dashboard or list apps with:

```bash
uv run modal app list
```

To tear down the deployment, stop its `lilo-fft-...` sampler apps, then `lilo`,
using `uv run modal app stop <app-id>`. Stopping `lilo` does not stop sampler apps.

## Profiling a training step

The Miles LoRA trainer path supports an opt-in `torch.profiler` capture plus
always-on per-phase wall-clock timing.

Set these environment variables in the deploying shell before `modal deploy`;
`trainer_deployment_env()` forwards them onto the trainer container:

- `LILO_TORCH_PROFILE_STEP`: 0-based optimizer step index to trace. When the
  first `forward_backward` of that step arrives, the selected rank-local Ray
  actors start a CPU+CUDA profiler and the backend process starts a CPU-only
  controller trace. The capture stops just before the first forward of the
  following step. Rank traces cover the actors' forward/backward, optimizer and
  slot export; the controller trace also covers the post-step sampler-weight
  save and publish, which run only in the backend process.
- `LILO_TORCH_PROFILE_RANKS`: comma-separated trainer ranks to trace, or
  `all`. Defaults to `0`; traces are large (~250 MB per rank per step).
- `LILO_TORCH_PROFILE_DIR`: output directory. Defaults to
  `<checkpoint volume mount>/torch-profile/<LILO_DEFINITION_ID>`. Each rank
  writes `rank{N}.trace.json.gz` (Chrome trace, loadable in
  `chrome://tracing` or Perfetto) and `rank{N}.key_averages.txt`; the backend
  writes `controller.trace.json.gz` / `controller.key_averages.txt`.

Deploying a separate app for a profiling run:

```bash
LILO_TORCH_PROFILE_STEP=2 \
  uv run modal deploy -m lilo.providers.modal.app
```

Always-on timing: every backend op prints one `lilo_step_timing` JSON line to
the trainer log per phase, and `optim_step` responses carry `timing/*` metrics
(`timing/forward_backward_s`, `timing/optim_step_s`, `timing/trainer_step_s`,
`timing/idle_wait_s`, `timing/save_sampler_weights_s`,
`timing/publish_weights_s`, per-phase `_calls` counts, and
`timing/optimizer_step`). These ride the existing `optim_step` response metrics
that the Tinker client/cookbook merge into per-step logged metrics, rather than
the trainer logging to W&B directly. Note that sampler save/publish for step k
runs after `optim_step` k returns, so `timing/save_sampler_weights_s` and
`timing/publish_weights_s` reported at step k+1 refer to the checkpoint taken
after step k.

To copy the traces off the checkpoint volume into the W&B run's **Files** tab
(under `<trace dir basename>/rank0.trace.json.gz` etc., downloadable straight into
Perfetto):

```bash
uv run scripts/upload_torch_profile.py \
  --dir <trace dir> --entity modal-labs --project miles-lora-longcontext \
  --run-id <wandb run id>
```

`WANDB_API_KEY` is read from the environment.

## Next steps

Read [Working with Full Fine-Tunes](docs/full-fine-tunes.md) for full training,
or [Working with Multi-LoRA](docs/multi-lora.md) for shared Miles adapters, batch
submission, scheduling, and sampling.
See [Validation](docs/validation.md) for end-to-end Lilo and Miles training runs,
and the [raw Tinker RL example](scripts/rl_example.py) for sampling and a toy
policy update. Copy examples you want to run into your project; repository
`scripts/` are not installed with the package.

The [W&B RL example](scripts/wandb_rl_example.py) extends it to a multi-step
loop that logs reward, response length, and Lilo's training metrics to Weights
& Biases from the client side; tinker-cookbook users can instead set
`wandb_project`/`wandb_name` on the cookbook `Config`.

See [Design](docs/design.md) for the control-plane, training-engine, and sampling
architecture.

See [Observability](docs/observability.md) for OTLP export to Datadog or a custom
destination, experiment labels, and the complete span/metric inventory.

## Examples

[Codeforces codegolf](examples/codeforces-codegolf/README.md) trains Qwen3.5-9B
with GRPO or TailRL advantages for correctness and short solutions. It includes
a sandboxed judge, checkpoint recovery, and commands to continue a checkpoint
with a different reward or advantage estimator. It also includes held-out
Pass@k and Best-of-k evaluation, plotting tools, and recorded learning curves.
