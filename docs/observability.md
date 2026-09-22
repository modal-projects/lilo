# Observability

For Tinker-SDK level metrics logging, the easiest solution is to use a library like Wandb -- we have an example of this in `scripts/wandb_rl_example.py`


Lilo also comes with much more fine-grained tracking for each subsystem, ie. how long training commands, checkpoint writes, and sampling requests
take as well as when the trainer is waiting between operations. It exports two kinds
of measurements:

- **Traces** record individual requests and operations. Each timed operation is
  a span, with a start time, end time, and attributes such as token count.
- **A trainer-state metric** reports the current operation every five seconds.
  It shows what the trainer is doing, not GPU utilization.

Lilo sends these measurements via OTLP (export off by default), from which Datadog or another compatible service can be used to receive and display these traces.

## Critical path without a tracing backend

Every trainer keeps a running breakdown of its critical path in memory, with no
exporter and no configuration. It separates the two things a step time conflates:
`*.queue_wait`, the interval between a request arriving and the trainer starting it
(fair sharing across LoRA slots, serialized persistence, other tenants), and
`*.execute`, the GPU work itself. Persistence adds `*.capture` and `*.persist`, and
model admission adds `accept.queue_wait` / `accept.execute`.

Read it from a training loop and log it next to the rest of your step metrics:

```python
from lilo.timing import log_critical_path

for step in range(steps):
    ...  # forward_backward, optim_step, publish weights
    log_critical_path(model_id, step=step)
```

`log_critical_path` reads `TINKER_BASE_URL` / `TINKER_API_KEY`, logs to the active
W&B run when there is one, prints a JSON line otherwise, and never raises into the
step. It resets the counters by default, so each call reports the interval since the
last one. Use `critical_path_metrics` to get the same flat `lilo/...` scalars without
logging them.

The snapshot is also available directly at `GET /api/v1/timing?model_id=...`, and a
trainer nobody is polling prints one `lilo_critical_path` JSON line every five
minutes, so container logs still show where the time went.

Gauges cover startup, measured from trainer-process start:
`trainer.backend_ready_s` (Megatron up and answering), `trainer.serving_ready_s`
(engine serving requests), and `trainer.first_model_ready_s` (first model admitted).

| Metric | Meaning |
| --- | --- |
| `lilo/forward_backward.queue_wait.mean_s` | Wait between submission and execution |
| `lilo/forward_backward.execute.mean_s` | Backend execution of the batch |
| `lilo/forward_backward.execute.mean_batch` | Commands coalesced into one execution |
| `lilo/save_weights_for_sampler.persist.total_s` | Publishing weights for sampling |
| `lilo/trainer.first_model_ready_s` | Cold start to first admitted model |

Phase totals are per model when a `model_id` is given and across every model on the
trainer otherwise, which is how a slot's own execution time is separated from the
time the trainer spends on its neighbors.

## Setup

### Modal to Datadog

Add the following variables to the `lilo-api` Modal secret in your deployment's
workspace and environment, preserving its existing authentication settings.
Replace `YOUR_DATADOG_API_KEY` with your Datadog API key and `your-environment`
with your deployment environment. Lilo sends telemetry directly to Datadog's
Modal intake endpoint.

```dotenv
OTEL_SERVICE_NAME=lilo
OTEL_RESOURCE_ATTRIBUTES=deployment.environment.name=your-environment
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=https://modal.integrations.otlp.datadoghq.com/v1/traces
OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf
OTEL_EXPORTER_OTLP_TRACES_HEADERS=dd-api-key=YOUR_DATADOG_API_KEY
OTEL_EXPORTER_OTLP_TRACES_TIMEOUT=5
OTEL_EXPORTER_OTLP_METRICS_ENDPOINT=https://modal.integrations.otlp.datadoghq.com/v1/metrics
OTEL_EXPORTER_OTLP_METRICS_PROTOCOL=http/protobuf
OTEL_EXPORTER_OTLP_METRICS_HEADERS=dd-api-key=YOUR_DATADOG_API_KEY
OTEL_EXPORTER_OTLP_METRICS_TIMEOUT=4
OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE=DELTA
```

These URLs are for Datadog US1. For another Datadog site, use the corresponding
[Modal managed-platform intake endpoints](https://docs.datadoghq.com/opentelemetry/setup/otlp_ingest/managed_platforms/).

Deploy Lilo after updating the secret:

```bash
MODAL_PROFILE=your-workspace MODAL_ENVIRONMENT=your-environment \
  uv run modal deploy -m lilo.providers.modal.app
```

The configuration applies to the control plane, sampling workers, and new
trainer containers. Existing trainer containers retain their configuration until
they are replaced.

### Custom OTLP destination

For a collector or compatible observability backend, set the standard OTLP base
endpoint and optional headers:

```dotenv
OTEL_SERVICE_NAME=lilo
OTEL_RESOURCE_ATTRIBUTES=deployment.environment.name=staging
OTEL_EXPORTER_OTLP_ENDPOINT=https://telemetry.example.com
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Bearer%20YOUR_TOKEN
```

The HTTP exporters append `/v1/traces` and `/v1/metrics` to the base endpoint.
Alternatively, use `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` and
`OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` for explicit complete URLs and configure
their headers independently. Signal-specific settings override general settings.
Header values use standard OTLP percent-encoding, as shown by `Bearer%20` above.
The destination must accept OTLP HTTP/protobuf and be reachable from the server
processes.

A trace endpoint alone enables only traces. A metric endpoint alone enables only
the trainer-state metric. A general endpoint enables both. With no endpoints,
export is disabled. Set `OTEL_SDK_DISABLED=true` to disable both explicitly.

## Experiment labels

Set `run_id` and `attempt_id` in `user_metadata` when creating a training client:

```python
training = create_full_training_client(
    service,
    "Qwen/Qwen3.5-4B",
    user_metadata={
        "run_id": "training-experiment-001",
        "attempt_id": "attempt-001",
    },
)
```

Keep `run_id` unchanged when recovering an experiment into a replacement model;
set a new `attempt_id` for the replacement. Only these two metadata keys are
exported, and only nonempty strings of at most 256 characters are accepted.

| Metadata key | Exported attribute | Meaning |
| --- | --- | --- |
| `run_id` | `lilo.run_id` | Experiment identity shared across replacement models |
| `attempt_id` | `lilo.run_attempt_id` | Experiment attempt; distinct from individual sampling HTTP attempts |

The labels follow training commands and their results. Publishing sampler weights
copies the labels into the publication, so sampling requests and retries can be
traced back to the same experiment. Publications and sampling sessions created
without labels remain untagged. Base-model sampling has no experiment label.

A trainer batch gets a label only when **all** commands in it have the same
value. A batch containing several experiments links to their commands instead.
The trainer-state metric describes the shared trainer, so it does not copy
experiment labels from individual models.

If a scoped deployment belongs to just one experiment, set `lilo.run_id` in
`OTEL_RESOURCE_ATTRIBUTES` to label the whole deployment. Lilo also copies this
value onto trainer metric datapoints so it is searchable through Datadog's direct
OTLP intake. The value stays the same when a model is replaced; these metrics
have no attempt label. Do not assign one experiment's run ID to a shared
deployment.

## Trace structure and lifecycle

Each accepted command gets a `lilo.command.<operation>` span from API submission
to trainer result. Child spans record submission, execution, snapshot capture,
persistence, and result availability. Waiting appears as gaps between those
spans; there is no separate queue span. The trainer closes the command span when
the result is ready, even if the client has stopped polling.

A command's execution span measures its participation in a backend operation.
Several commands can share that operation, so use `lilo.trainer.*` spans to count
batches or measure total trainer work without counting the same work twice.

Deduplicated submissions attach to the original command trace. Rejected
submissions have standalone control-plane spans. Model creation and unloading
have separate control-plane submission and trainer lifecycle spans.

Each backend execution gets a trace with the batch's total workload and links
to its commands. Separate spans show snapshot capture, persistence, and waiting
for an earlier write, so overlapping persistence and training remain visible.
Model acceptance and unloading also have their own spans.

Counts refer to logical input examples and their supplied text tokens, before
backend packing/padding. Input-token count is omitted if any input chunk has no
known text-token length. Executor spans measure elapsed time on the host,
including backend transport and synchronization.

Megatron adds child spans for preparation, forward or combined forward/backward,
result collection, and optimizer work on rank zero. Interleaved forward/backward
microbatches appear as one interval. These measure host wall time without adding
CUDA synchronization. 

Workload attributes have distinct scopes:

- `lilo.loss_tokens` counts positions with a nonzero resolved loss weight and a
  target other than `-100`. It is a position count, not a sum of weights or a
  guarantee of a nonzero gradient. It appears on each command and is summed on
  the physical batch after preparation.
- `lilo.padded_tokens` and `lilo.packed_microbatch_count` describe the whole packed
  batch before data-parallel sharding, including packing padding. They exclude
  dummy microbatches added for rank balancing and are not divided among commands.
- `lilo.checkpoint_bytes` is the logical size of files in a completed training
  checkpoint directory, including all rank shards, metadata, and any model export.
  It measures stored file bytes rather than upload traffic or in-memory tensors.
  Size is omitted if it cannot be read. Sampler publications do not report this
  training-checkpoint attribute.

Within checkpoint persistence, `lilo.backend.checkpoint_write` measures rank-zero
serialization and file writes, while `lilo.backend.checkpoint_commit` measures
the wait for all writers and the Volume commit. The enclosing command and trainer
spans show where this work overlaps later training.

## Viewing telemetry in Datadog

After running a training or sampling operation, search APM for `service:lilo`
(or your configured `OTEL_SERVICE_NAME`). Filter by resource name to select a
span family:

- `lilo.command.forward_backward`: full command lifetime and its execution children.
- `lilo.trainer.forward_backward`: physical trainer batches and aggregate workload.
- `lilo.sample`: sampling requests and their HTTP attempts.

Use `@lilo.run_id` to filter by experiment and `@lilo.run_attempt_id` to select an
attempt. Follow span links between a command and its shared trainer batch. Native
notebook span searches support `@duration`, `@lilo.example_count`, and
`@lilo.input_tokens` as columns.

For trainer activity, graph the state metric by operation:

```text
avg:lilo.trainer.state{lilo.trainer_instance_id:ENGINE_ID,lilo.lane:execution} by {lilo.operation}.fill(null)
```

Replace `ENGINE_ID` with the trainer instance ID. Use a stacked area display and
separate charts for the `execution`, `checkpoint`, and `sampler` lanes. Each sample
identifies the active operation in that lane. Missing reports appear as gaps;
rollups can average samples into fractional values. This metric represents
operation state, not GPU utilization.

Configure Datadog APM retention for spans you need to search later. Child spans
can arrive while the parent command is still running.

## Export inventory

All spans include IDs, parent or linked span IDs, start and end times, status,
and resource attributes such as `service.name` and `OTEL_RESOURCE_ATTRIBUTES`.
Lilo does not export logs through OTLP.

| Signal | Name | Boundary / purpose |
| --- | --- | --- |
| Span | `lilo.command.<operation>` | Submission receipt → trainer result ready; `forward`, `forward_backward`, `optim_step`, `save_weights`, `load_weights`, `save_weights_for_sampler`, internal `skip` |
| Span | `lilo.control.submit` | HTTP submission work, attached to the canonical command root |
| Span | `lilo.control.<operation>` | Submission without a command root, including rejected submissions and model creation/unloading |
| Span | `lilo.command.execute`, `lilo.command.capture`, `lilo.command.persist` | Active executor/capture/persistence interval for this command; child of its root, linked to the physical batch; excludes waiting |
| Span | `lilo.trainer.result_ready` | Terminal command marker, including failure |
| Span | `lilo.trainer.forward`, `lilo.trainer.forward_backward`, `lilo.trainer.optim_step`, `lilo.trainer.load_weights` | One actual executor invocation/batch; links to participating commands |
| Span | `lilo.trainer.accept`, `lilo.trainer.unload` | Engine model lifecycle work |
| Span | `lilo.trainer.wait_persistence.save_weights`, `lilo.trainer.wait_persistence.save_weights_for_sampler` | Wait for preceding work in the same persistence lane |
| Span | `lilo.trainer.capture.save_weights`, `lilo.trainer.capture.save_weights_for_sampler` | Capture state for persistence/publication |
| Span | `lilo.trainer.persist.save_weights`, `lilo.trainer.persist.save_weights_for_sampler` | Background persistence/publication |
| Span | `lilo.backend.prepare`, `lilo.backend.forward`, `lilo.backend.forward_backward`, `lilo.backend.collect`, `lilo.backend.outputs`, `lilo.backend.optimizer` | Rank-zero backend phases; children of the physical trainer operation |
| Span | `lilo.backend.checkpoint_write`, `lilo.backend.checkpoint_commit` | Rank-zero file writing, then writer synchronization and volume commit |
| Span | `lilo.sample` | Sampling acceptance → worker completion; worker start if acceptance timestamp unavailable |
| Span | `lilo.sample.attempt` | One upstream sampling HTTP attempt, including retries; child of sampling root |
| Gauge | `lilo.trainer.state` | One-hot operation state, observed/exported every five seconds |

| Span family | Additional exported attributes |
| --- | --- |
| Trainer and command identity | `lilo.trainer_instance_id`, `lilo.definition_id`, `lilo.boot_id`, `lilo.component`; `lilo.model_id`, `lilo.request_id` where there is one owner |
| Command | `lilo.seq_id`, `lilo.operation`, `lilo.example_count`, `lilo.input_tokens`, `lilo.loss_tokens`, `lilo.checkpoint_bytes` where applicable; `lilo.incomplete=true` on graceful shutdown with unfinished work |
| Trainer phase/batch | `lilo.lane`, `lilo.operation`, `lilo.command_count`; aggregate `lilo.example_count`, `lilo.input_tokens`, `lilo.loss_tokens` when known for all participants; `lilo.padded_tokens`, `lilo.packed_microbatch_count`, `lilo.checkpoint_bytes` when supplied by the backend |
| Backend phase | Physical operation attributes plus `lilo.rank=0` and `lilo.component=backend` |
| Control | `lilo.operation`, `lilo.component`, `http.response.status_code`, `error.type` on exceptions; model/request identity after successful handoff |
| Experiment-aware spans | `lilo.run_id`, `lilo.run_attempt_id` under the rules above |
| Sampling root | `lilo.request_id`, `lilo.model_id`, `lilo.base_model`, `lilo.num_samples`, `lilo.version_requested`, `lilo.latest`, `lilo.start_boundary`, `lilo.input_tokens`, `lilo.output_tokens`, `lilo.attempt_count`, `lilo.retry_count`, `error.type`; `lilo.version_served_start`, `lilo.version_served_end` for single-sequence requests |
| Sampling attempt | `lilo.request_id`, `lilo.attempt_id`, `lilo.sequence_index`, `lilo.attempt_number`, `lilo.input_tokens`, `lilo.output_tokens`, `http.response.status_code`, `error.type`, `lilo.version_served_start`, `lilo.version_served_end` |
| SGLang timing and cache | `sglang.request_id`, `sglang.queue_s`, `sglang.prefill_s`, `sglang.post_prefill_to_finish_s`, `sglang.cached_tokens`, `sglang.prompt_tokens`, `sglang.completion_tokens` |

SGLang fields are present when supplied by the backend.
`sglang.post_prefill_to_finish_s` includes decoding and final processing. Missing
fields are omitted. Raw backend timing payloads and training results are not
exported.

| Gauge | Values and labels |
| --- | --- |
| `lilo.trainer.state` | `1` for current operation, explicit `0` for other operations in that lane. Labels: `lilo.trainer_instance_id`, `lilo.definition_id`, `lilo.boot_id`, `lilo.component`, `lilo.lane`, `lilo.operation` |
| Execution lane | `idle`, `accept`, `unload`, `forward`, `forward_backward`, `optim_step`, `save_weights`, `load_weights`, `save_weights_for_sampler`, `skip` |
| Checkpoint lane | `idle`, `save_weights` |
| Sampler lane | `idle`, `save_weights_for_sampler` |
