# Lilo design

Lilo presents the Tinker API while separating control-plane orchestration,
GPU execution, and sampling. Here, we describe the high-level design and interfaces
between these components. 

## System overview

A request moves through three layers: 

1. The stateless control plane owns the Tinker-facing API, session lifecycle,
   model placement, and sampling orchestration.
2. A training engine owns one or more model states as well as the actual distributed training backend (Megatron) that holds per-rank GPU workers. A frontend server in each engine turns ordered API calls into commands for the GPU workers, similar to the "broadcast" orchestration of RL libraries. 
3. The sampling plane loads published weights from the trainer via a shared Modal volume (the stitch "bulletin")and serves `sample` requests independent from the training engine. 

Both the training and sampling parts of the system are independently autoscalable: the control plane will provision new training engines for newly registered training clients, and the sampling layer will autoscale replicas based on incoming load through the Flash proxy gateway. 

## Control plane

The control plane is a stateless, autoscaling Modal web server. Its replicas are responsible for: 

- creating and expiring client sessions
- handling model creation, assigning models to engine containers, scaling the set of engines based on model demand, and routing training operations to each engine. 
- recording and resolving Tinker API futures 
- handling sampling (`asample` submission and futures) 

The control plane stores sessions, models, engine assignments, and sampling
state in Modal Dicts so every replica shares the same durable state.

### Sampling retention

Sampling-session records, creation anchors, and sampler-artifact records are
point-lookup metadata, not a resource inventory. The cleaner never enumerates
them. Modal reclaims inactive entries using its
[seven-day read/write inactivity expiry](https://modal.com/docs/guide/dicts).
Reading or writing an entry can extend that retention; this is not a fixed
seven-day lifetime or an indefinite idempotency guarantee.

Application `expires_at` deadlines are enforced on reads and retries, including
before ensuring a sampling pool or recovering an export from its receipt.
Sampling-session lookup also requires an open parent session. Late lookups reject
a closed or removed parent before accessing its task Dict. This does not add
transactional exclusion between requests already in flight and parent cleanup.
Expired records report `expired` while retained, rather than becoming `not found`
on the next cleaner pass. Named artifacts remain reserved against conflicting
exports while their records exist, even after application expiry; callers should
use unique checkpoint names rather than relying on the cleaner to free a name.

Parent-session cleanup still closes idle sessions, unloads models, removes their
placement/demand state, and deletes each parent's entire task Dict. Individual
expired sampling tasks remain until that deletion or storage expiry. Idle-engine
and sampling-pool shutdown remain unchanged. Dict expiry does not delete weight
files or stop running resources. The local in-memory provider has no automatic
entry expiry; it retains this metadata for the lifetime of the store.

## Training engines

Client sessions are handled by the control plane. Training engines receive only
model-scoped operations corresponding to Tinker API calls, including model
creation, forward and backward passes, optimizer steps, checkpoints, sampler
publication, and model unload.

Each accepted operation is assigned an ordered model sequence ID and a future.
The engine buffers later requests while the current command runs, allowing the
next command's inputs to accumulate before the accelerator is available. The
engine scheduler then selects the next ready command. Compatible
`forward_backward` operations can be grouped and their variable-length
sequences packed under the backend's token budget.

The engine exposes the command protocol in
[`engine/api.py`](../src/lilo/engine/api.py). Its scheduler and future handling
live in [`engine/server.py`](../src/lilo/engine/server.py).

### Execution lanes

To maximize concurrency through our system, we separate "GPU" operations from "non-GPU" operations, 
such that disk writing of checkpoints can be issued asynchronously and does not block the GPU. GPU-requiring commands are serialized, which preserves ordering of training commands and avoids concurrent mutation of weights/other states. 

Disk- and network-heavy persistence runs outside the GPU lane. Full checkpoints
and sampler publication have independent persistence lanes, so a long durable
state write cannot delay publishing weights for the next rollout. Each operation
is split into:

1. `capture_checkpoint`, which captures model state from the GPU onto immutable CPU buffers
2. `persist_checkpoint`, which writes that state to persistent storage *independent* of the GPU lane. 

Persistence can therefore overlap a later forward/backward or optimizer command
without reading partially updated model state.

## Weight publication

Sampler weights are persisted to Modal volumes shared with the sampling plane. With this setup, 
the training engine containers and entire sampling plane (and individual sampling replicas) need not be RDMA connected to each other, as p2p weight transfer is never used anywhere in the system. 

To minimize transfer latency, publication format depends on the parameterization:

- **LoRA:** each publication contains the full adapter weights. The adapter is
  written to the shared Stitch bulletin Volume and can be loaded independently
  by a sampler replica.
- **Full fine-tuning (FFT):** publications after the base version contain sparse
  weight deltas. Replicas discover versions through Stitch and walk the parent
  chain of deltas, applying updates in place until they reach the requested
  version.

Publications are immutable. A version becomes visible to samplers only after
its persistence phase completes.

## Sampling

An `asample` request creates one Modal Function call. That function owns the
sample attempt synchronously: it selects a compatible sampling replica, sends
the generation request, reroutes retryable failures, and returns the completed
Tinker sample response. The control plane stores the Modal Function call ID and
uses it as the durable future for polling and retries.

The sampling topology differs by parameterization:

- **LoRA (target topology):** one autoscaling Modal server hosts the base
  model. Adapters are loaded lazily from the bulletin Volume and retained in an
  LRU cache, allowing many adapters to share one sampling fleet. LoRA training
  is available today, but the Modal sample dispatcher is currently connected only
  to FFT definitions; the shared LoRA sampler still needs to be connected.
- **FFT:** each trained model has its own autoscaling Modal server. Stitch
  tracks exact and latest weight versions, while replicas update their local
  weights from the shared delta chain.

This separation lets training engines scale according to active models while
sampling scales according to rollout traffic. We best-effort sticky-route groups (GRPO) to the same container to maximize KV cache reuse. 

## Relevant implementation

- [`control_plane/service.py`](../src/lilo/control_plane/service.py): handles
  sessions, places models on engines, routes operations, and submits sampling
  requests
- [`providers/modal/kv.py`](../src/lilo/providers/modal/kv.py): stores
  control-plane state in Modal Dicts
- [`providers/modal/trainer_reconciler.py`](../src/lilo/providers/modal/trainer_reconciler.py):
  starts and stops training engines to match model demand
- [`engine/server.py`](../src/lilo/engine/server.py): orders model operations,
  batches compatible training requests, and runs persistence alongside later
  GPU work
- [`engine/spmd.py`](../src/lilo/engine/spmd.py): broadcasts backend commands to
  every rank and collects results
- [`inference/bulletin.py`](../src/lilo/inference/bulletin.py): stores immutable
  LoRA adapter snapshots and tracks the latest version (LoRA sampler publication path)
- [`inference/fft_bulletin.py`](../src/lilo/inference/fft_bulletin.py): stores
  and resolves versioned FFT weight updates (FFT sampler publication path)
- [`providers/modal/fft_pool.py`](../src/lilo/providers/modal/fft_pool.py):
  creates, finds, wakes, and stops each FFT model's sampling service using Modal flash proxy

## Adding a new model deployment 

"Model deployment" in this context refers to a particular training configuration for a base model, defined by its parameterization (full parameter, LoRA, etc.), desired context/sampling length, parallelism, quantization, and so forth. To add a new deployment that can be spun up by the control plane: 

1. Existing model definitions are in [`providers/modal/definitions`](../src/lilo/providers/modal/definitions) (one per file). Create a new file with the desired configuration details (model, checkpoint, context-length, GPU, and parallelism settings). 
2. Keep the module filename, `DEFINITION_ID`, and engine function name the same.
3. Import the module in
   [`providers/modal/app.py`](../src/lilo/providers/modal/app.py) and append it
   to `DEFINITIONS`.

Every definition exports:

- `DEFINITION_ID`, `MODEL_NAME`, `PARAMETERIZATION`, and `CATALOG_VISIBLE`;
- `TRAINER_MODELS_PER_INSTANCE`;
- the model asset Volume and backend configuration;
- a Modal `app` and engine function that calls
  [`run_engine_with_backend`](../src/lilo/providers/modal/serve.py); and
- `ENGINE_FUNCTION`, referencing that engine function.

The existing model definitions show the complete template for parameterization-specific settings. FOr example, FFT definitions include rollout resources + Stitch bulletin volume, whereas LoRA definitions include adapter rank, slot capacity, and adapter storage. Our FFT engines are currently only capable of hosting one model (but if multiple FFT experiments are submitted to the control plane, it will spin up as many engine replicas as necessary to support these concurrently). LoRA engines are multi-lora and so can host multiple adapters. 

Adding the module to `DEFINITIONS` automatically includes its Modal
sub-application and makes it available to model lookup, trainer provisioning,
parameterization lookup, and the public catalog. The registry tests also cover
the new definition automatically. To run the tests before deploying: 

```bash
uv run pytest tests/providers/test_definition_registry.py
uv run modal deploy -m lilo.providers.modal.app
```

Trainer container limits are deployment-specific. Set
`LILO_TRAINER_MAX_CONTAINERS` to a positive integer when deploying to apply the
same limit to every definition. Leaving it unset makes trainer containers
unlimited.

The following script tests e2e deployment for one model definition, launching the Modal app, creating the particular model, running forward_backward + optim_step operations, and sampler weight publication/rollouts: 

```bash
uv run python scripts/e2e_engine_definition.py \
  --definition-id <definition_id>
```

This validates all steps of the training cycle for the particular model definition. For an FFT checkpoint round trip, add
`--checkpoint-only`.

## Adding a new backend

Each backend implements the synchronous per-rank interface in
[`backends/contract.py`](../src/lilo/backends/contract.py). `EngineServer`
handles asynchronous scheduling and calls the backend in rank lockstep.

Create a backend module that provides:

- a `CommandBackend` implementation for model acceptance, forward/backward,
  optimizer steps, checkpoints, sampler publication, unload, and shutdown;
- `build_executor()`, which initializes distributed communication, constructs
  the backend, and returns its `DistributedExecutor`.

Use [`backends/megatron_lora.py`](../src/lilo/backends/megatron_lora.py) and
[`backends/megatron_fft.py`](../src/lilo/backends/megatron_fft.py) as the LoRA
and FFT references. A model definition passes its `module:build_executor` path
to `run_engine_with_backend`:

```
run_engine_with_backend(
    ...,
    "lilo.backends.megatron_fft:build_executor",
    ...
)
```

**Distributed Executor**: All ranks participate in commands through
[`engine/spmd.py`](../src/lilo/engine/spmd.py). Rank zero exposes the backend
HTTP bridge while the other ranks follow its broadcasts. To maximize GPU utilization, we separate "GPU ops" (ie. forward_backward, optim_step) from "non-GPU ops" (ie. CPU -> disk checkpoint writing). For this reason, checkpoint and sampler publication operations are split into a GPU and non-GPU component: 

1. `capture_*` writes out an immutable snapshot to CPU (in general can be any intermediate destination).
2. `persist_*` writes that snapshot on to persistent storage. 

`capture_*` operations hence are "GPU ops," and `persist_*` are "non-GPU ops" that can be run asynchronously from the GPU lane. 

**IMPORTANT!!** The EngineServer handles the asynchronous scheduling of the GPU and persistence lanes. However, it is on the backend writer to ensure that their `persist_*` methods and GPU methods do not race with each other (ie. dont have conflicting access to the same state). We recommend against using locks for this purpose and have our LoRA and FFT backends written as references of how to implement lock-free backends. In our implementation, the main pattern is that `capture_*` creates detached CPU state that `persist_*` exclusively owns.
