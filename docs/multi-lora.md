# Working with Multi-LoRA

The Miles backend keeps several independent LoRA adapters on one shared base
model and trainer. Each Tinker training client owns its adapter, gradients, and
optimizer state. Clients share GPU execution time and a rollout pool; creating
another client does not reserve a private trainer.

## Read this before starting a run

- **Submit work from clients concurrently.** Waiting for one client's whole
  update before submitting the next client's batch prevents useful batching.
- **Keep each client's update order explicit.** Submit all gradient-accumulation
  batches before its `optim_step()`. Other clients can advance independently.
- **Bound outstanding work.** Large batches and long queues increase latency
  for clients sharing the trainer. There is no per-client latency guarantee.
- **Checkpoint each adapter.** A shared trainer failure loses all resident
  adapters' unsaved model and optimizer state. Sampler publication is not a
  recovery checkpoint.

## Create clients

Use the ordinary Tinker LoRA constructor against a deployment with the Miles
LoRA definition enabled:

```python
import tinker

service = tinker.ServiceClient()  # TINKER_BASE_URL and TINKER_API_KEY
clients = [
    service.create_lora_training_client(
        base_model="Qwen/Qwen3.5-9B-Base", rank=16,
    )
    for _ in range(6)
]
```

The bundled `qwen3_5_9b_base_miles_lora_16k` definition has six adapter slots
per four-H100 trainer, TP=4, and a 16,384-token context. Clients using the same
definition can share an engine while it has capacity. Placement is automatic;
the SDK does not guarantee that particular clients are colocated. Additional
clients depend on available trainer capacity and deployment limits.

Ranks can differ across clients, up to the deployment's maximum of 32.
LoRA alpha and target modules are deployment-wide. For this definition,
`train_attn`, `train_mlp`, and `train_unembed` must all be true (the SDK defaults).
Per-client LoRA initialization `seed` is unsupported; leave it unset. Keep
clients alive across updates rather than recreating them each step.

## How training is scheduled

Each engine has one GPU command lane. It preserves operation order within each
client, but there is no barrier requiring every client to reach the same step.

At each dispatch, the engine checks the next queued operation for each client.
If it selects `forward_backward`, it combines ready requests with the **same
`loss_fn` and `loss_fn_config`**, including consecutive compatible requests from
one client. Different ranks, batch sizes, and sequence lengths can participate.
It does not wait for missing clients to fill a batch. Different losses or loss
configurations execute separately. Optimizer steps, forward-only calls, and
snapshot captures are not combined by this scheduling path.

Miles splits the combined work into microbatches using the deployment's token
budget (16,384 tokens per GPU in this definition). That is a packing budget,
not a limit on total tokens in a submitted batch. A large submission can occupy
multiple microbatches; it is not preempted to run a later client's request.
Client ordering is not strict round-robin, so avoid keeping an unbounded backlog
on one client.

Submit each client's forward/backward and optimizer operations without waiting
for the forward result first, then check both futures:

```python
import asyncio
from tinker import types

adam = types.AdamParams(learning_rate=1e-5)

async def update(client, batch):
    forward = await client.forward_backward_async(
        batch, loss_fn="importance_sampling",
    )
    optimizer = await client.optim_step_async(adam)
    result = await forward.result_async()
    await optimizer.result_async()
    return result

# Each batch contains data for its corresponding adapter only.
results = await asyncio.gather(*(
    update(client, batch)
    for client, batch in zip(clients, batches, strict=True)
))
```

This example waits for all clients for convenience; an async RL controller can
start each client's next rollout when that client's publication completes.
Keep one ordered submission loop per client. For gradient accumulation, submit
several `forward_backward` calls before the single `optim_step`; inserting an
optimizer step changes the update, even if requests happen to be batched together.

## Batch requirements

- Each `Datum` belongs to the training client receiving it. Do not concatenate
  different adapters' data into one client's batch; the scheduler combines
  clients' separate requests.
- Inputs must be text tokens. `target_tokens` must have the same length as
  `model_input` and be shifted by one token. Per-token weights, advantages, and
  sampling logprobs must align with those targets. Mask prompt positions when
  training only on completions.
- Supported losses are `cross_entropy`, `importance_sampling`, `ppo`, `cispo`,
  and `dro`. RL losses require caller-supplied advantages and sampling logprobs;
  the backend does not compute reward groups for you.
- Keep each prompt plus completion within the context limit. Reducing the
  number of examples does not make an overlong individual sequence fit.

## Publication, rollouts, and checkpoints

`save_weights_and_get_sampling_client()` returns a client that may use that
publication or a newer one. For an exact policy version, call
`save_weights_for_sampler(name)` and create a sampling client from its returned
path. Both use the same shared multi-LoRA rollout pool; an exact version does
not get its own pool. A replica may still need to load that adapter version on
first use. The bundled 16K definition retains at most 64 adapter versions per
replica, shared across clients. Older versions are evicted and reloaded on demand;
this is a cache limit, not a limit on training steps or saved publications.

Publish after `optim_step()` and await publication before requesting rollouts
from the updated policy. If prefetching rollouts, bound policy lag in your
controller: the backend does not reject a batch merely because it was generated
by an older policy. Retain the logprobs returned with those rollouts.

Adapter capture uses the shared GPU command lane. Persistence can overlap later
training and other adapters' publications, but only one publication per adapter
can be in flight. Queueing another publication for that adapter blocks its
later operations until the prior publication persists. Keep at most one
`save_state()` future pending per controller; repeated checkpoint captures can
wait on the engine-wide checkpoint writer and delay other clients. Model
creation, loading, and unloading also wait for outstanding persistence.

Use `save_state()` for adapter and optimizer recovery, then
`load_state_with_optimizer()` on a new matching LoRA client. Restore requires
the same base model, LoRA configuration, Miles revision, and trainer topology.
Published PEFT adapters are inference artifacts and cannot replace these Miles
training checkpoints. Checkpoints from the earlier Miles revision are not
compatible with the current distributed checkpoint format; start a fresh run
when upgrading. Wait for checkpoint completion before relying on a checkpoint.

Rollout capacity is configured by the deployment, not per LoRA client. The
bundled definition keeps eight one-H200 replicas warm and allows up to eight
LoRAs per inference batch. Trainer slots and rollout replicas are separate
limits. More clients can fill gaps while others generate rollouts, but share
that fixed capacity. Cold starts and adapter loading still affect latency.

The normal path does not promise bitwise equality across different packing or
request schedules. Matching sampling seeds alone does not provide deterministic
training or eliminate policy-version differences.

New deployment processes resolve `radixark/miles` `main` once and build that exact
commit. The SHA is part of the image cache key and is stored in training
checkpoints; running deployments do not update underneath active jobs. Deployment
requires Git/network access to resolve the branch. For a reproducible rebuild,
set `LILO_MILES_COMMIT` to a full commit SHA before deployment. Megatron and
Megatron-Bridge retain their separately configured revisions.
