# Working with Full Fine-Tunes

Full fine-tuning (FFT) updates all model parameters on a dedicated trainer in
your Modal workspace. Use a [scoped run](scoped-runs.md) for one training job or
a shared deployment for multiple jobs behind one API. Both use the Tinker SDK.

You pay for allocated GPUs and time. The trainer's GPU layout, number of sampling
replicas, and cold starts determine cost and throughput, even when the training
code stays the same.

## Read this before starting a run

- **Take regular training checkpoints.** Trainer GPUs can be preempted, and
  containers have a hard 24-hour lifetime. Use `save_state()` throughout a run.
- **Use the reusable latest-version pool for iterative RL.** Creating sampling
  clients from named publications starts exact-version pools that can cold
  start separately.
- **Size the sampling pool for your workload.** `min_containers` keeps replicas
  warm between batches; `max_containers` limits how far the pool can scale.
  Too few replicas leave the trainer waiting for rollouts.
- **Higher learning rates can increase sampler publication time.** Our FFT
  weight syncs use sparse deltas, and in our Qwen
  FFT RL runs, `1e-5` and above produced larger delta payloads, while `1e-6`
  kept the deltas smaller in the recorded runs. The FFT reasoning runs in the
  [validation guide](validation.md) use `1e-6`; this is a tested setting, not a
  learning-rate rule for every workload.

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

Transport failures on the engine's local connection to its GPU worker also
terminate the backend process group and exit the engine. A lost response can
leave a gradient accumulation or optimizer update applied without confirmation;
the command is not retried, and the failed connection rejects further commands.
Recover from a completed checkpoint rather than replaying the uncertain update.
This applies to backend transport failures, not errors polling the public API.

### Checkpoint archives are not served

Lilo does not provide a presigned URL from Tinker's checkpoint archive
endpoint. Instead, checkpoints can be read directly from the deployment's
Modal Volume:

```bash
modal volume get <checkpoint-volume> /<checkpoint-path> <local-destination>
```

New training checkpoints are stored as `<checkpoint-name>/<model-id>/` in the
volume, so you can browse by the name passed to `save_state`. The model ID keeps
identically named checkpoints from different runs separate. Public
`tinker://<model-id>/weights/<checkpoint-name>` paths use this volume layout.

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

### Shared deployment limits

Each FFT training model needs its own engine. The deployment can limit the number
of engines for each model definition. Multiple clients using the same API do not
share one FFT model's GPU or optimizer state.

## Performance and behavior considerations

### Configure rollout capacity for the workload

Elastic inference autoscales based on time-averaged incoming rollout requests.
Bursty synchronous RL can leave the pool idle between batches, then wait for it
to scale up again. Keep replicas warm when that delay is significant. In a
shared deployment, set their capacity when creating the client:

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

For scoped runs, pass `latest=lilo.Pool(...)` to `lilo.run` instead; scoped
clients reject the `rollout` argument. In shared deployments, `rollout` applies
to full-parameter clients and must be set at creation. `user_metadata` does not
configure trainer or sampler resources.

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

Keep enough rollouts in flight to feed the trainer, but bound the queue. Queued
samples can become too stale for the next update, and cancelling a local worker
does not guarantee that its remote Modal call stops. Discarded rollouts may
therefore continue using inference GPUs.

For a complete Cookbook-backed LongRLVR workload, see
[`scripts/e2e_longrlvr_qwen3_5_35b_a3b_full.py`](../scripts/e2e_longrlvr_qwen3_5_35b_a3b_full.py).

### Keep sampler deltas small

FFT publications capture incremental sparse weight deltas on the trainer's
serialized GPU lane and persist them separately. Larger parameter updates,
including those from higher learning rates, can change more bytes and increase
publication and sampler update time. Measure these times for your workload;
`1e-6` is the setting used in the recorded FFT reasoning runs.

### Persist checkpoints without blocking every training step

Checkpoint capture runs on the serialized GPU lane, while Volume persistence
can overlap later GPU work. Keep at most one checkpoint future pending because
the persistence writer processes one checkpoint at a time. The engine
serializes captures, so submitting another checkpoint before the prior one
persists can block later GPU operations.

The pinned Tinker Cookbook loop uses `save_checkpoint_async(kind="both")`, which
waits for both the state checkpoint and sampler publication before returning a
sampling client. A slow checkpoint write can therefore delay the next rollouts.

You can overlap that write with later work: keep the checkpoint future pending
and wait only for the sampler publication needed by the next rollouts. In this
sketch, `deadline` is the application's timeout wrapper and `start_rollouts`
submits work using the new sampling client:

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
