# Observability

Lilo exports traces for training commands, trainer operations, and sampling
requests, plus live queue, stage, and GPU activity metrics. Traces include
workload counts and optional experiment labels for correlating activity across
requests and models.

Export is disabled by default. To enable it, configure an OTLP HTTP/protobuf
destination in the server environment.

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

A trace endpoint alone enables only traces. A metric endpoint alone enables metrics without tracing. A general endpoint enables both. With no endpoints,
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
Arbitrary user metadata is not exported.

| Metadata key | Exported attribute | Meaning |
| --- | --- | --- |
| `run_id` | `lilo.run_id` | Experiment identity shared across replacement models |
| `attempt_id` | `lilo.run_attempt_id` | Experiment attempt; distinct from individual sampling HTTP attempts |

Labels attach to command roots, command execution/result spans, control submissions,
and trainer execution/lifecycle spans. Sampler artifacts snapshot these labels;
sessions and submitted sampling tasks carry them to sampling roots and HTTP
attempt spans. Artifacts and sessions created without labels remain untagged.
Base-model sampling has no model experiment identity.

A trainer execution receives a label only if **all** its participating commands
have that same label. A batch crossing experiments has links to each command and
is not attributed to a single experiment. Trainer-state metrics describe the
physical trainer and do not copy experiment labels from models.

For a single-tenant scoped deployment, the owner can set `lilo.run_id` in
`OTEL_RESOURCE_ATTRIBUTES`. Scoped trainers also emit that deployment identity
as a metric datapoint tag: direct Datadog OTLP intake does not necessarily promote
custom resource attributes to searchable metric tags. The tag stays constant
across model replacements, and physical metrics do not gain an attempt label.
Shared deployments retain their original physical-only datapoint labels.

## Trace structure and lifecycle

Each accepted engine command has a root `lilo.command.<operation>` span, beginning
at control-plane submission receipt and ending when its trainer result is ready.
The trainer owns completion, so client polling is not necessary to close it.
Control submission, active execution/capture/persistence, and result-ready are
children of that root. Waiting appears as gaps; no command queue span is emitted.
Execution children link to the physical backend span and carry only their own
command’s workload and experiment labels. These show participation latency; use
`lilo.trainer.*` execution spans to count physical batches without duplication.

Deduplicated submissions attach to the original command trace. Rejected
submissions have standalone control-plane spans. Model creation and unloading
have separate control-plane submission and trainer lifecycle spans.

Backend executions are separate traces, with span links to every
participating command, including when there is only one command. Thus a merged
batch appears once, with aggregate workload counts, rather than being duplicated
under several command roots. Capture, persistence, and waits for previous
persistence are distinct physical intervals linked to their command. Persistence
can overlap a subsequent execution. Lifecycle accept/unload spans are independent.

Counts refer to logical input examples and their supplied text tokens, before
backend packing/padding. Input-token count is omitted if any input chunk has no
known text-token length. Executor spans measure wall-clock time, including
backend transport and synchronization. They do not measure GPU kernel time.

Megatron backend phases are children of the physical trainer span. They measure
preparation, forward or combined forward/backward scheduling, result collection,
and optimizer work on rank zero. Forward/backward scheduling may interleave
microbatches; it is recorded as one interval. These are host wall-clock intervals
and do not introduce CUDA synchronization or measure individual GPU kernels.

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

Checkpoint capture and persistence retain their existing command and trainer
spans. Within persistence, `lilo.backend.checkpoint_write` measures rank-zero
file serialization and writes; `lilo.backend.checkpoint_commit` measures the
existing wait for all writers and the volume commit. Persistence can overlap
later training operations.

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

Configure Datadog APM retention for the spans that need to remain searchable.
Finished child spans may arrive before their command root has finished.

## Export inventory

All spans include standard OpenTelemetry identity, parent/link context, start/end
timestamps, status, and resource attributes (`service.name`, SDK metadata, and
configured `OTEL_RESOURCE_ATTRIBUTES`). No log exporter is installed.

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


## Multi-LoRA performance

With an OTLP metrics endpoint configured, new rollout containers start SGLang
with `--enable-metrics`. The sidecar reads its local `/metrics` endpoint and sends
an approved set of counters, gauges, and histograms over OTLP HTTP/protobuf.
It does not expose that endpoint through the public gateway. No separate
Collector is required; the same OTLP destination can be Datadog or your Collector.

The engine snapshots its queue once a second. Metrics export every five seconds;
native serving and GPU samples run about every five seconds plus scrape/export
time. These background tasks do not refresh volumes or synchronize CUDA.

| Signal | What it measures |
| --- | --- |
| `lilo.trainer.queue.depth` | Buffered client commands, excluding lifecycle operations |
| `lilo.trainer.queue.eligible` | Model-head commands that the current scheduler can select; a compatible batch may also consume subsequent forward/backward commands |
| `lilo.trainer.queue.blocked` | Buffered commands grouped by sequence, model readiness, lifecycle barrier, persistence capacity, or same-adapter publication |
| `lilo.trainer.queue.inflight` | Dispatched commands that have not finished, including capture and persistence |
| `lilo.trainer.queue.oldest_age`, `last_progress_age` | Seconds since the oldest buffered command arrived, or since the last dispatch/completion |
| `lilo.trainer.queue.lifecycle_depth` | Pending model registration/unload operations |
| `lilo.stage.inflight`, `oldest_age`, `last_progress_age` | Live calls, oldest unfinished call age, and time since a call completed, by component and stage |
| `lilo.stage.duration`, `completed` | Completed host-stage wall times and counts |
| `lilo.trainer.batch.input_tokens`, `example_count`, `adapter_count` | Distributions of actual Miles batch sizes before internal packing |
| `sglang.num_running_reqs`, `num_queue_reqs` | Native scheduler running and waiting counts, per scheduler rank |
| `sglang.token_usage`, `cache_hit_rate`, `kv_*`, `mamba_usage` | Native cache occupancy, available/evictable tokens, and cache hits |
| `sglang.lora_pool_slots_used`, `lora_pool_slots_total`, `lora_pool_utilization` | GPU adapter-slot occupancy |
| `sglang.generation_tokens`, `prompt_tokens`, `cached_tokens` | Native token counters, exported as deltas |
| `sglang.time_to_first_token_seconds`, `inter_token_latency_seconds`, `e2e_request_latency_seconds`, `queue_time_seconds` | Native latency histograms, when SGLang supplies observations |
| `sglang.num_requests`, `num_aborted_requests`, `num_retracted_requests` | Native completion, abort, and retraction counters |
| `lilo.gpu.activity`, `memory_activity`, `memory_used`, `memory_total`, `power` | Periodic `nvidia-smi` samples per device; activity is a percentage |
| `lilo.gpu.capacity` | Visible GPU-seconds per container, exported as deltas |

Stage spans use `lilo.<component>.<stage>`. Sampling distinguishes pool readiness,
pool-lock acquisition, route readiness, retry backoff, and each inference HTTP
attempt. The inference side distinguishes adapter-resolution waiting, lock
acquisition, volume refresh, validation, adapter registration, and the SGLang
request. The HTTP attempt propagates its trace context into the sidecar.

Training adds a `lilo.command.queue` child from engine acceptance to dispatch.
Separate stages cover backend HTTP, Miles dispatch, actor method execution,
checkpoint-lock acquisition, and volume commit/refresh. Miles dispatch and actor
spans carry the same `lilo.batch_id`; correlate them within a trainer container.
Actor spans include data access, collectives, and execution. They do not isolate
individual kernels. Backend phase spans also separate input preparation and
output construction from the Miles call.

The `lilo.sample.provider_handoff` span covers the time immediately before Modal
spawn through sampling-worker entry, including transport and scheduling. It does
not establish how much of that interval was spent in Modal's queue. Admission
and placement have their own control-plane stages. Client work before submission
requires instrumentation in the client script.

To investigate a stall, compare live ages and queue state with GPU activity:

- An idle trainer with no eligible work is waiting for its clients or a recorded
  dependency. Check `queue.blocked` before blaming the dispatcher.
- An idle trainer with eligible work aging needs a dispatch/backend investigation.
- Inference requests aging in `adapter_lock_wait` identify registration contention.
  Requests aging in `sglang_request` need the native queue/activity metrics to
  distinguish serving backlog from transport or execution.
- Growing native queues alongside high GPU activity indicate serving pressure.

Use the rate of `sglang.generation_tokens` for generated tokens/sec. Divide its
sum over an interval by `lilo.gpu.capacity` over the same containers and interval
for tokens/GPU-second. Sum tokenizer token counters once per replica; do not
multiply them by tensor-parallel ranks. Inter-token latency observations and
request-average time per output token are different statistics.

The first native counter/histogram scrape establishes a baseline; later scrapes
export exact delta counts and histogram buckets. Resets establish a new baseline.
Unknown metric labels are skipped, so adding custom SGLang labels requires
reviewing the forwarding allowlist. Request IDs and adapter versions appear only
in traces, not metric labels. GPU telemetry failure produces missing samples.

Remaining limits: SGLang's adapter occupancy does not count GPU adapter swaps;
registration frequency alone cannot prove thrashing. Native SGLang queues lack
oldest-request age, and requests stuck before a Modal worker starts have no live
worker gauge. Ray admission and actor data preparation are not individually
measured. Trainer padded/loss tokens and MFU are omitted until measured at the
packing/kernel boundary. Host stage times can overlap and must not be added as
if they were GPU utilization. Export failures can lose telemetry; they do not
retry training or sampling operations.
