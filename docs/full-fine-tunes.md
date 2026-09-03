# Working with Full Fine-Tunes

Lilo's bundled full fine-tuning stack runs as a shared deployment that
orchestrates single-tenant training containers in your Modal workspace. It
exposes a Tinker-compatible API, but usage is not infrastructure-agnostic like
the hosted Tinker service. Because pricing is compute-based rather than
token-based, trainer topology, rollout capacity, and cold starts directly
determine performance and cost. The Tinker-level abstractions can still be used
for training experiments, but good performance requires matching deployment
capacity to the workload.

## Read this before starting a run

- **Take regular training checkpoints.** Trainer GPUs can be preempted, and
  containers have a hard 24-hour lifetime. Use `save_state()` throughout a run.
- **Use the reusable latest-version pool for iterative RL.** Creating sampling
  clients from named publications starts exact-version pools that can cold
  start separately.
- **Tune rollout capacity for target throughput.** Use `min_containers` to keep
  replicas warm between bursts and `max_containers` to control peak
  parallelism, keeping the trainer supplied without excessive queueing.
- **Higher learning rates can increase sampler publication time.** Our FFT
  weight syncs use sparse deltas, and in our Qwen
  FFT RL runs, `1e-5` and above produced larger delta payloads, while `1e-6`
  converged without unwieldy deltas. All our validation runs use a learning
  rate of `1e-6` across model sizes and context lengths.

## Concrete limitations

### Checkpointing and recovery

The trainer holds model and optimizer state in GPU memory. A GPU preemption,
container failure, or the 24-hour Modal function timeout loses that in-memory
state. Take checkpoints regularly and recover manually from the latest completed one:

```python
saved = training.save_state("step-000100").result(timeout=60 * 60)

resumed = create_full_training_client(service, base_model)
resumed.load_state_with_optimizer(saved.path).result(timeout=60 * 60)
```

### Checkpoint archives are not served

Lilo does not provide a presigned URL from Tinker's checkpoint archive
endpoint. Instead, checkpoints can be read directly from the deployment's
Modal Volume:

```bash
modal volume get <checkpoint-volume> /<checkpoint-path> <local-destination>
```

## Differences from the Tinker SDK

### Full-client creation uses a Lilo helper

Create FFT clients with `create_full_training_client()` or its async variant:

```python
from lilo.client import create_full_training_client

training = create_full_training_client(
    service,
    "Qwen/Qwen3.5-35B-A3B",
)
```

After creation, the client exposes the normal Tinker training methods. Lilo
currently supports `tinker>=0.24.1,<0.25`; newer SDK versions are not guaranteed
to be compatible.

### Training checkpoints and sampler publications are different

`save_state()` persists full model and optimizer state for recovery.
`save_weights_for_sampler()` publishes inference weights to the sampling plane.
FFT sampler publications are incremental deltas that depend on the base model
and preceding deltas, so their paths cannot be passed to `load_state()`.

### Latest and exact sampling clients use different pools

`save_weights_and_get_sampling_client()` publishes the current weights and
returns a client for the model's reusable latest-version pool. Requests through
that client are constrained to a version at least as new as the publication
that created it, but later requests may use newer publications. This is the
preferred path for async RL where bounded policy drift is acceptable and
delta-based weight updates are fast.

`save_weights_for_sampler(name)` creates a named, immutable publication.
Creating a sampling client from the returned path uses a separate pool pinned
to that exact publication version. Use exact-version clients for evaluation,
reproducibility, or other work that must not advance to newer weights. Note that each
exact-version pool may pay its own deployment and model cold-start cost.

Both sampler publication workflows are separate from `save_state()`, which
checkpoints parameters and optimizer state. Publishing weights makes them
available to inference but does not save full model or optimizer state.

### Shared Deployment Usage Limits

For concurrent FFT training jobs, each new training run spins up its own
training engine due to the single-tenancy of full-parameter training. The
deployment lets you limit the number of training engines that can be active
per model.

## Performance and behavior considerations

### Configure rollout capacity for the workload

Elastic inference autoscales based on time-averaged incoming rollout requests.
This works for fully async RL in steady state, but not for bursty synchronous
RL patterns. Synchronous RL often benefits from keeping rollout replicas warm
between batches:

```python
training = create_full_training_client(
    service,
    "Qwen/Qwen3.5-35B-A3B",
    rollout={
        "min_containers": 2,
        "max_containers": 8,
        "scaledown_window": 600,
    },
)
```

Set `min_containers` to capacity the job can keep busy and the workspace can
run continuously. Fully asynchronous workloads can usually leave the minimum
at zero and rely on incoming demand to scale up. Always choose
`max_containers` from the desired concurrency and budget. Modal workspace
limits still apply.

`scaledown_window` controls how many seconds an idle rollout replica remains
warm before scaling down, with a default of five minutes. Increase it when
synchronous batches have longer gaps and would otherwise repeatedly cold
start; decrease it when traffic is continuous or replicas should be released
more quickly.

The `rollout` configuration applies only to full-parameter clients created
through `create_full_training_client()` and must be
provided during client creation. `user_metadata` does not configure trainer or
sampler topology.

### Expect cold starts

The first model creation may need to download model assets, initialize the
distributed trainer, and compile kernels. Rollout replicas separately load and
compile the sampling model. Model creation starts the latest pool
asynchronously, but does not guarantee that rollout replicas are ready.

The first training step and sample can therefore take much longer than steady
state. In a shared deployment, the first experiment may encounter cold starts,
while later experiments can reuse warm containers when they use the same model
deployment and jobs arrive continuously.

### Bound asynchronous rollout work

Rollout queues should be large enough to keep the trainer saturated even with
long generation times. Async RL can benefit from overprovisioning rollout
requests and discarding extras, but Lilo does not offer sampling-side abort
semantics. Cancelling local workers therefore does not guarantee that their
Modal calls were cancelled, which can leave inference compute occupied by
discarded rollouts.

For a complete Cookbook-backed LongRLVR workload, see
[`scripts/e2e_longrlvr_qwen3_5_35b_a3b_full.py`](../scripts/e2e_longrlvr_qwen3_5_35b_a3b_full.py).

### Keep sampler deltas small

FFT publications capture incremental sparse weight deltas on the trainer's
serialized GPU lane and persist them separately. Larger parameter updates,
including those from higher learning rates, can change more bytes and increase
publication and sampler update time. The validated `1e-6` learning rate is a
starting point; the scaling of sparse delta payload sizes with different
training dynamics requires further study.

### Persist checkpoints without blocking every training step

Checkpoint capture runs on the serialized GPU lane, while Volume persistence
can overlap later GPU work. Keep at most one checkpoint future pending because
the persistence writer processes one checkpoint at a time. The engine
serializes captures, so submitting another checkpoint before the prior one
persists can block later GPU operations.

The following example waits on async checkpointing and sampling clients
separately. The pinned Tinker Cookbook RL loop handles periodic checkpoints
with `save_checkpoint_async(kind="both")`. It submits the state checkpoint and
sampler publication together, but waits for both before returning the new
sampling client. A slow full-state write can therefore delay rollout of the
updated policy even though Lilo persists state checkpoints and sampler
publications on separate lanes.

To use Lilo's split persistence behavior, keep the state-checkpoint future
pending while waiting only for the sampler client required by new rollout work:

```python

pending_checkpoint = None

for step, batch in enumerate(batches, start=1):
    forward = training.forward_backward(batch, loss_fn)
    update = training.optim_step(adam)

    await deadline(forward.result_async())
    await deadline(update.result_async())

    if step % save_every == 0:
        if pending_checkpoint is not None:
            await deadline(pending_checkpoint.result_async())

        # Submit, but let Volume persistence overlap later work.
        pending_checkpoint = training.save_state(f"step-{step:06d}")

    # Await this because new rollouts need the updated sampling client.
    sampling_client = await deadline(
        training.save_weights_and_get_sampling_client_async()
    )
    start_rollouts(sampling_client)

# Ensure the final checkpoint is durable before exiting.
if pending_checkpoint is not None:
    saved = await deadline(pending_checkpoint.result_async())
    print(saved.path)
```

Wait for the final future before exiting and record `saved.path`.
