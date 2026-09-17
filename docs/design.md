# Lilo design

Lilo implements the Tinker API with three components: an API service that routes
requests, training engines that run GPU work, and sampling replicas that serve
published weights.

## System overview

1. The **control plane** handles the Tinker API, client sessions, model placement,
   and request routing.
2. A **training engine** holds model and optimizer state. Its server orders API
   operations and dispatches them to the distributed backend's GPU workers.
3. **Sampling replicas** load published weights from a shared Modal Volume,
   called the Stitch bulletin, and serve generation requests independently.

The control plane starts trainers as models are created. Sampling replicas scale
separately with incoming requests through the Flash proxy gateway.

## Control plane

The control plane is a stateless Modal web server that scales with API traffic.
It creates sessions and models, assigns models to trainers, routes training and
sampling requests, and records results for Tinker futures.

The control plane stores sessions, models, engine assignments, and sampling
state in Modal Dicts so every replica shares the same durable state.

## Training engines

Training engines receive operations for individual models: creation, forward
and backward passes, optimizer steps, checkpoints, sampler publication, and
unloading. The control plane handles client sessions.

Each accepted operation is assigned an ordered model sequence ID and a future.
The engine queues later requests while the current command runs, then selects
the next ready command. Compatible
`forward_backward` operations can be grouped and their variable-length
sequences packed under the backend's token budget.

The engine exposes the command protocol in
[`engine/api.py`](../src/lilo/engine/api.py). Its scheduler and future handling
live in [`engine/server.py`](../src/lilo/engine/server.py).

### Execution lanes

The engine runs one GPU operation at a time, preserving command order and
preventing concurrent changes to model state. Checkpoint writing runs separately
so it can overlap later training.

Disk- and network-heavy persistence runs outside the GPU lane. Full checkpoints
and sampler publication use separate persistence lanes. Each operation has two
phases:

1. `capture_checkpoint` copies model state into immutable CPU buffers on the GPU
   lane.
2. `persist_checkpoint` writes those buffers to storage outside the GPU lane.

Persistence can therefore overlap a later forward/backward or optimizer command
without reading partially updated model state.

## Weight publication

Trainers publish sampler weights through shared Modal Volumes. They do not need
an RDMA connection to sampling replicas. The publication format depends on which
parameters are trained:

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

An `asample` request starts a Modal Function call that selects a sampling replica,
sends the generation request, retries recoverable failures on another replica,
and returns the Tinker response. The control plane stores the Modal Function call ID and
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

Training capacity follows the number of active models; sampling capacity follows
rollout traffic. When possible, samples from the same group (for example, a GRPO
group) go to one replica to reuse the prompt's KV cache.

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

A model definition specifies a base model, which parameters to train, context
limits, GPU layout, and backend settings. Scoped runs can use a custom engine
recipe from the user's project; see [scoped runs](scoped-runs.md#custom-engines).
To add a definition to the shared deployment:

1. Add a file in [`providers/modal/definitions`](../src/lilo/providers/modal/definitions)
   with the model, checkpoint, context, GPU, and parallelism settings.
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

Use an existing definition as a template. FFT definitions include sampling
resources and the Stitch bulletin Volume; LoRA definitions include adapter rank,
slot capacity, and adapter storage. Each FFT engine holds one training model.
The control plane starts more engines for concurrent FFT models, subject to
container limits. LoRA engines can hold several adapters.

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

This script deploys one definition, creates a model, runs forward/backward and an
optimizer update, publishes weights, and samples:

```bash
uv run python scripts/e2e_engine_definition.py \
  --definition-id <definition_id>
```

For an FFT checkpoint save-and-restore test, add `--checkpoint-only`.

## Adding a new backend

Each backend implements the synchronous per-rank interface in
[`backends/contract.py`](../src/lilo/backends/contract.py). `Engine`
handles asynchronous scheduling and calls the backend in rank lockstep.

Create a backend module that provides:

- a `Backend` implementation for model acceptance, forward/backward,
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
HTTP bridge while the other ranks follow its broadcasts through `run_follower_loop`.
The `Engine` schedules operations and passes `Command` values to its
executor. `HttpBackendClient` in
[`engine/backend_http.py`](../src/lilo/engine/backend_http.py) implements that
interface across the local training subprocess boundary.

Implement checkpoint and publication work in two phases so writing to storage
can overlap GPU work:

1. `capture_*` creates an immutable snapshot, typically in CPU memory.
2. `persist_*` writes that snapshot to storage.

The executor names these phases `capture_snapshot` and `persist_snapshot`.
The per-rank backend implements `persist_checkpoint` for checkpoints and
`publish_sampler_snapshot` for sampler weights. The latter persists the snapshot
and makes it available to samplers; checkpoint persistence does not imply sampler
publication.

The engine schedules capture on the GPU lane and persistence on a separate
lane. The backend must make this overlap safe: persistence must not read live
weights while an optimizer update changes them. The existing backends solve
this by giving persistence exclusive ownership of detached CPU snapshots,
rather than holding a lock that blocks training during the write.
