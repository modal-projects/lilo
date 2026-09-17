---
name: lilo
description: Set up and run Tinker-compatible training and sampling with Lilo on Modal. Use for deployment, engine configuration, weight publication, checkpoint recovery, and diagnosing startup or throughput.
---

# Working with Lilo

Lilo runs training and sampling on Modal through the Tinker SDK. Below details lilo-specific infra considerations that go beyond the Tinker SDK. 

## Connect or deploy

For an existing endpoint, create `tinker.ServiceClient(base_url=url, api_key=key)`.
Clients need only that URL and key, not Modal credentials.

For a new deployment, install Lilo in the user's project:

```bash
uv add 'lilo @ git+https://github.com/modal-projects/lilo.git'
```

Use the Python version supported by the installed revision; scoped runs currently
require Python 3.12. Select the Modal workspace and environment before creating
secrets. There are three separate credentials:

- Modal credentials allow deployment and resource management.
- The `lilo-proxy` secret holds `MODAL_PROXY_TOKEN_ID` and
  `MODAL_PROXY_TOKEN_SECRET` for access to sampler pools.
- The Lilo API key authenticates Tinker clients. A scoped run generates one;
  a shared deployment reads it from the `lilo-api` secret.

See the [README](../../README.md) for authentication and shared deployment commands.

## Start a full fine-tuning run

Use the scoped `lilo.run` for a new, dedicated full fine-tuning (FFT) run:

```python
import lilo
import tinker
from lilo.engines import qwen3_5_4b_full_64k

engine = qwen3_5_4b_full_64k()
with lilo.run(engine=engine) as (url, api_key):
    service = tinker.ServiceClient(base_url=url, api_key=api_key)
    training = lilo.create_full_training_client(service, engine.model)
    # Train and sample here using the Tinker SDK.
```

A scope allows one active training model. Creating another client does not attach
it to the existing model. Other processes can connect while the scope's owner is
alive. For multiple training jobs behind one API, use a shared deployment (ie. multi-lora) -- each FFT model still gets its own dedicated trainer. 

The process holding `lilo.run` owns its trainer and sampler apps within the scope. If the owner dies, Modal stops them after detecting the disconnect. Exiting does not automatically save a checkpoint, and retrying the controller creates a new scope, so the application must restore an explicitly saved past checkpoint upon retry. 

See [scoped runs](../../docs/scoped-runs.md) for more details on setup and recovery.

## Configure the engine and sampler capacity

An engine recipe sets the model, context limits, GPU layout, and backend options.
Built-in recipes in `lilo.engines` return frozen dataclasses. Use
`dataclasses.replace`, including for nested settings, to customize one in the
user's project:

```python
from dataclasses import replace

engine = replace(
    engine,
    name="my-engine",
    sampling=replace(engine.sampling, memory_fraction=0.90),
)
```

Changing the GPU type alone may not work: GPU count, parallelism, context length,
and memory requirements must fit together.

Set scoped sampler capacity with
`lilo.run(engine=engine, latest=lilo.Pool(min_containers=..., max_containers=...,
scaledown_window=...))`. For shared deployments, set `rollout` when creating the
full training client. Scoped clients reject `rollout`; `user_metadata` never
configures resources.

## Publish weights before sampling

Training and inference keep separate copies of the weights. After an update,
publish the weights that the next rollouts need:

```python
latest = training.save_weights_and_get_sampling_client()
```

This reuses the latest-policy pool (ie. monotonically increasing weight version). Thus, this handle gives a "min-version" guarantee, that given a sample request with a particular weight version, it will be served with *at least* that version. 

For evaluation/any requests that needs a fixed version, instead used the *named* sampler path: 

```python
publication = training.save_weights_for_sampler(publication_name).result()
pinned = training.create_sampling_client(publication.path)
```

Because each pinned version can need its own inference allocation and cold start, the latest pool should be used for standard RL rollout sampling when the algorithm allows newer weight versions. FFT publications store incremental weight deltas, so larger deltas take longer to write and apply (can be roughly thought as linear scaling in the difference in weight versions). Neither publication API saves optimizer state or replaces training checkpoints, adhering with the Tinker semantics. 


## Allow for cold starts

Trainer and sampler startup occur separately in our decoupled design. A sampling handle can exist before
its replicas are ready, and requests will receive HTTP 408 until the inference replicas are ready (sglang engine on each one has spun up and is ready to receive requests). 

| Operation | Startup work to expect |
| --- | --- |
| Enter `lilo.run(..., warm=True)` | Prepare assets, allocate the trainer, and load the model. `warm=False` defers trainer warmup. |
| Create a full training client | Wait for trainer readiness, including loading and distributed initialization if cold. |
| First forward/backward | Compile kernels; the delay appears while waiting for the result. |
| First sample on a cold pool | Allocate inference GPUs, load and compile the model, and apply published weights. |

Measure steady-state calls separately. Samplers can go cold after scaling down,
and a new pinned version may start a separate pool even if the latest pool is warm.

## Checkpoint and recover

`save_state(name)` saves both model parameters and optimizer state, from which these can be loaded by 
`load_state_with_optimizer(path)` to resume the training run perfectly.

After trainer loss, a replacement trainign client can be created from a prior checkpoint. Within the same scope, old sampling_client/training_client handles will return HTTP 410 after this reassignment. 

An optimizer timeout can mean the update succeeded but its response was lost.
Do not blindly repeat it. A sampling failure alone does not mean the trainer was
lost. See [FFT recovery and checkpointing](../../docs/full-fine-tunes.md) for the
supported recovery paths and an example of overlapping saves with training/using the asynchronous capture capabilities to maximize trainer utilization and not block future GPU work unnecesarily on disk writes.

## Diagnose cost and throughput

Costs depend on both allocated GPUs and time. Training and inference have separate
allocations, and a small batch still uses the engine's configured GPU layout.
A trainer waiting for rollouts or the controller can remain allocated and billed.

Use Modal container logs and GPU metrics, plus Lilo's optional traces and trainer
operation metric to diagnose response latency/timing. The
[observability guide](../../docs/observability.md) explains OTLP configuration,
experiment labels, and what each measurement includes.
