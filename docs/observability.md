# Observability

Tune exports traces for training commands, trainer operations, and sampling
requests, plus a metric showing the trainer's current operation. Traces include
workload counts and optional experiment labels for correlating activity across
requests and models.

Export is disabled by default. To enable it, configure an OTLP HTTP/protobuf
destination in the server environment.

## Setup

### Modal to Datadog

Add the following variables to the `tune-api` Modal secret in your deployment's
workspace and environment, preserving its existing authentication settings.
Replace `YOUR_DATADOG_API_KEY` with your Datadog API key and `your-environment`
with your deployment environment. Tune sends telemetry directly to Datadog's
Modal intake endpoint.

```dotenv
OTEL_SERVICE_NAME=tune
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

Deploy Tune after updating the secret:

```bash
MODAL_PROFILE=your-workspace MODAL_ENVIRONMENT=your-environment \
  uv run modal deploy -m tune.providers.modal.app
```

The configuration applies to the control plane, sampling workers, and new
trainer containers. Existing trainer containers retain their configuration until
they are replaced.

### Custom OTLP destination

For a collector or compatible observability backend, set the standard OTLP base
endpoint and optional headers:

```dotenv
OTEL_SERVICE_NAME=tune
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
Arbitrary user metadata is not exported.

| Metadata key | Exported attribute | Meaning |
| --- | --- | --- |
| `run_id` | `tune.run_id` | Experiment identity shared across replacement models |
| `attempt_id` | `tune.run_attempt_id` | Experiment attempt; distinct from individual sampling HTTP attempts |

Labels attach to command roots, command execution/result spans, control submissions,
and trainer execution/lifecycle spans. Sampler artifacts snapshot these labels;
sessions and submitted sampling tasks carry them to sampling roots and HTTP
attempt spans. Artifacts and sessions created without labels remain untagged.
Base-model sampling has no model experiment identity.

A trainer execution receives a label only if **all** its participating commands
have that same label. A batch crossing experiments has links to each command and
is not attributed to a single experiment. Trainer-state metrics describe the
physical trainer and do not copy experiment labels from models.

For a single-tenant scoped deployment, the owner can set `tune.run_id` in
`OTEL_RESOURCE_ATTRIBUTES`. Scoped trainers also emit that deployment identity
as a metric datapoint tag: direct Datadog OTLP intake does not necessarily promote
custom resource attributes to searchable metric tags. The tag stays constant
across model replacements, and physical metrics do not gain an attempt label.
Shared deployments retain their original physical-only datapoint labels.

## Trace structure and lifecycle

Each accepted engine command has a root `tune.command.<operation>` span, beginning
at control-plane submission receipt and ending when its trainer result is ready.
The trainer owns completion, so client polling is not necessary to close it.
Control submission, active execution/capture/persistence, and result-ready are
children of that root. Waiting appears as gaps; no command queue span is emitted.
Execution children link to the physical backend span and carry only their own
command’s workload and experiment labels. These show participation latency; use
`tune.trainer.*` execution spans to count physical batches without duplication.

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

- `tune.loss_tokens` counts positions with a nonzero resolved loss weight and a
  target other than `-100`. It is a position count, not a sum of weights or a
  guarantee of a nonzero gradient. It appears on each command and is summed on
  the physical batch after preparation.
- `tune.padded_tokens` and `tune.packed_microbatch_count` describe the whole packed
  batch before data-parallel sharding, including packing padding. They exclude
  dummy microbatches added for rank balancing and are not divided among commands.
- `tune.checkpoint_bytes` is the logical size of files in a completed training
  checkpoint directory, including all rank shards, metadata, and any model export.
  It measures stored file bytes rather than upload traffic or in-memory tensors.
  Size is omitted if it cannot be read. Sampler publications do not report this
  training-checkpoint attribute.

Checkpoint capture and persistence retain their existing command and trainer
spans. Within persistence, `tune.backend.checkpoint_write` measures rank-zero
file serialization and writes; `tune.backend.checkpoint_commit` measures the
existing wait for all writers and the volume commit. Persistence can overlap
later training operations.

## Viewing telemetry in Datadog

After running a training or sampling operation, search APM for `service:tune`
(or your configured `OTEL_SERVICE_NAME`). Filter by resource name to select a
span family:

- `tune.command.forward_backward`: full command lifetime and its execution children.
- `tune.trainer.forward_backward`: physical trainer batches and aggregate workload.
- `tune.sample`: sampling requests and their HTTP attempts.

Use `@tune.run_id` to filter by experiment and `@tune.run_attempt_id` to select an
attempt. Follow span links between a command and its shared trainer batch. Native
notebook span searches support `@duration`, `@tune.example_count`, and
`@tune.input_tokens` as columns.

For trainer activity, graph the state metric by operation:

```text
avg:tune.trainer.state{tune.trainer_instance_id:ENGINE_ID,tune.lane:execution} by {tune.operation}.fill(null)
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
| Span | `tune.command.<operation>` | Submission receipt → trainer result ready; `forward`, `forward_backward`, `optim_step`, `save_weights`, `load_weights`, `save_weights_for_sampler`, internal `skip` |
| Span | `tune.control.submit` | HTTP submission work, attached to the canonical command root |
| Span | `tune.control.<operation>` | Submission without a command root, including rejected submissions and model creation/unloading |
| Span | `tune.command.execute`, `tune.command.capture`, `tune.command.persist` | Active executor/capture/persistence interval for this command; child of its root, linked to the physical batch; excludes waiting |
| Span | `tune.trainer.result_ready` | Terminal command marker, including failure |
| Span | `tune.trainer.forward`, `tune.trainer.forward_backward`, `tune.trainer.optim_step`, `tune.trainer.load_weights` | One actual executor invocation/batch; links to participating commands |
| Span | `tune.trainer.accept`, `tune.trainer.unload` | Engine model lifecycle work |
| Span | `tune.trainer.wait_persistence.save_weights`, `tune.trainer.wait_persistence.save_weights_for_sampler` | Wait for preceding work in the same persistence lane |
| Span | `tune.trainer.capture.save_weights`, `tune.trainer.capture.save_weights_for_sampler` | Capture state for persistence/publication |
| Span | `tune.trainer.persist.save_weights`, `tune.trainer.persist.save_weights_for_sampler` | Background persistence/publication |
| Span | `tune.backend.prepare`, `tune.backend.forward`, `tune.backend.forward_backward`, `tune.backend.collect`, `tune.backend.outputs`, `tune.backend.optimizer` | Rank-zero backend phases; children of the physical trainer operation |
| Span | `tune.backend.checkpoint_write`, `tune.backend.checkpoint_commit` | Rank-zero file writing, then writer synchronization and volume commit |
| Span | `tune.sample` | Sampling acceptance → worker completion; worker start if acceptance timestamp unavailable |
| Span | `tune.sample.attempt` | One upstream sampling HTTP attempt, including retries; child of sampling root |
| Gauge | `tune.trainer.state` | One-hot operation state, observed/exported every five seconds |

| Span family | Additional exported attributes |
| --- | --- |
| Trainer and command identity | `tune.trainer_instance_id`, `tune.definition_id`, `tune.boot_id`, `tune.component`; `tune.model_id`, `tune.request_id` where there is one owner |
| Command | `tune.seq_id`, `tune.operation`, `tune.example_count`, `tune.input_tokens`, `tune.loss_tokens`, `tune.checkpoint_bytes` where applicable; `tune.incomplete=true` on graceful shutdown with unfinished work |
| Trainer phase/batch | `tune.lane`, `tune.operation`, `tune.command_count`; aggregate `tune.example_count`, `tune.input_tokens`, `tune.loss_tokens` when known for all participants; `tune.padded_tokens`, `tune.packed_microbatch_count`, `tune.checkpoint_bytes` when supplied by the backend |
| Backend phase | Physical operation attributes plus `tune.rank=0` and `tune.component=backend` |
| Control | `tune.operation`, `tune.component`, `http.response.status_code`, `error.type` on exceptions; model/request identity after successful handoff |
| Experiment-aware spans | `tune.run_id`, `tune.run_attempt_id` under the rules above |
| Sampling root | `tune.request_id`, `tune.model_id`, `tune.base_model`, `tune.num_samples`, `tune.version_requested`, `tune.latest`, `tune.start_boundary`, `tune.input_tokens`, `tune.output_tokens`, `tune.attempt_count`, `tune.retry_count`, `error.type`; `tune.version_served_start`, `tune.version_served_end` for single-sequence requests |
| Sampling attempt | `tune.request_id`, `tune.attempt_id`, `tune.sequence_index`, `tune.attempt_number`, `tune.input_tokens`, `tune.output_tokens`, `http.response.status_code`, `error.type`, `tune.version_served_start`, `tune.version_served_end` |
| SGLang timing and cache | `sglang.request_id`, `sglang.queue_s`, `sglang.prefill_s`, `sglang.post_prefill_to_finish_s`, `sglang.cached_tokens`, `sglang.prompt_tokens`, `sglang.completion_tokens` |

SGLang fields are present when supplied by the backend.
`sglang.post_prefill_to_finish_s` includes decoding and final processing. Missing
fields are omitted. Raw backend timing payloads and training results are not
exported.

| Gauge | Values and labels |
| --- | --- |
| `tune.trainer.state` | `1` for current operation, explicit `0` for other operations in that lane. Labels: `tune.trainer_instance_id`, `tune.definition_id`, `tune.boot_id`, `tune.component`, `tune.lane`, `tune.operation` |
| Execution lane | `idle`, `accept`, `unload`, `forward`, `forward_backward`, `optim_step`, `save_weights`, `load_weights`, `save_weights_for_sampler`, `skip` |
| Checkpoint lane | `idle`, `save_weights` |
| Sampler lane | `idle`, `save_weights_for_sampler` |
