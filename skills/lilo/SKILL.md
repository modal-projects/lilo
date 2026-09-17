---
name: lilo
description: Set up and run Tinker-compatible training and sampling with Lilo on Modal. Use for deployment, engine configuration, weight publication, checkpoint recovery, and diagnosing startup or throughput.
---

# Working with Lilo

Lilo runs training and sampling on Modal through the Tinker SDK. Use the user's
chosen model and training algorithm. The guidance below covers resource ownership,
weight versions, and recovery; it does not prescribe an RL algorithm.

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

Prefer `lilo.run` for a new, dedicated full fine-tuning (FFT) run:

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
alive. For multiple training jobs behind one API, use a shared deployment; each
FFT model still gets a dedicated trainer.

The process holding `lilo.run` owns its trainer and sampler apps. Normal exit
stops them. If the owner dies, Modal stops them after detecting the disconnect;
cleanup is not immediate. Exiting does not save a checkpoint. Completed
checkpoints remain in the Modal Volume. Retrying the controller creates a new
scope, so the application must restore its checkpoint.

See [scoped runs](../../docs/scoped-runs.md) for setup and recovery.

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

This reuses the latest-policy pool. The handle requires weights at least as new
as this publication; later requests may use newer weights.

For evaluation that needs a fixed version:

```python
publication = training.save_weights_for_sampler(publication_name).result()
pinned = training.create_sampling_client(publication.path)
```

Each pinned version can need its own inference allocation and cold start. Prefer
the latest pool when the algorithm allows newer weights. FFT publications store
incremental weight deltas; larger deltas take longer to write and apply. Neither
publication API saves optimizer state or replaces a training checkpoint.

## Allow for cold starts

Trainer and sampler startup happen separately. A sampling handle can exist before
its replicas are ready.

| Operation | Startup work to expect |
| --- | --- |
| Enter `lilo.run(..., warm=True)` | Prepare assets, allocate the trainer, and load the model. `warm=False` defers trainer warmup. |
| Create a full training client | Wait for trainer readiness, including loading and distributed initialization if cold. |
| First forward/backward | Compile kernels; the delay appears while waiting for the result. |
| First sample on a cold pool | Allocate inference GPUs, load and compile the model, and apply published weights. |

Measure steady-state calls separately. Samplers can go cold after scaling down,
and a new pinned version may start a separate pool even if the latest pool is warm.

## Checkpoint and recover

`save_state(name)` saves model and optimizer state.
`load_state_with_optimizer(path)` restores both. Record the returned checkpoint
path and its captured step only after the save succeeds. Also persist the data
iterator position and other application state needed to resume; a model
checkpoint does not save the controller's Python memory.

Choose a checkpoint interval from measured save time and the amount of work the
user can afford to repeat. Full FFT checkpoints can take minutes. A reproducible
base model usually does not need a step-zero save.

Capture is ordered with training; writing the captured state can overlap later
GPU work. Keep at most one checkpoint save pending. Awaiting every write before
submitting more work can leave the trainer idle; queuing saves faster than storage
can finish them can block later operations. Wait for the last save before exit.
A checkpoint captured at step 20 restores step 20 even if its write finishes at
step 22. Until that write succeeds, use the previous completed checkpoint.

After trainer loss, create a replacement, restore a completed checkpoint, and
publish weights again. In a surviving scope, old latest-policy handles return
HTTP 410 after reassignment; obtain a new handle from the replacement trainer.
A controller retry alone does not restore state.

An optimizer timeout can mean the update succeeded but its response was lost.
Do not blindly repeat it. A sampling failure alone does not mean the trainer was
lost. See [FFT recovery and checkpointing](../../docs/full-fine-tunes.md) for the
supported recovery paths and an example of overlapping saves with training.

## Diagnose cost and throughput

Costs depend on allocated GPUs and time. Training and inference have separate
allocations, and a small batch still uses the engine's configured GPU layout.
A trainer waiting for rollouts or the controller can remain allocated and billed.

Check where time goes before adding GPUs: startup, generation, trainer execution,
weight publication, checkpoint writes, or waiting between operations. More
inference replicas may reduce trainer waiting but increase total cost. Keeping
replicas warm trades idle cost for fewer cold starts. Overlap rollouts and
training only if the user's algorithm permits the resulting policy lag.

Use Modal container logs and GPU metrics, plus Lilo's optional traces and trainer
operation metric. Operation time is not GPU utilization. The
[observability guide](../../docs/observability.md) explains OTLP configuration,
experiment labels, and what each measurement includes.
