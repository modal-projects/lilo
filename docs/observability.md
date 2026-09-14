# Observability

Lilo exports traces for training commands, trainer operations, and sampling
requests, plus a metric showing the trainer's current operation. Traces include
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
physical trainer and do not carry experiment labels.

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
| Span | `lilo.sample` | Sampling acceptance → worker completion; worker start if acceptance timestamp unavailable |
| Span | `lilo.sample.attempt` | One upstream sampling HTTP attempt, including retries; child of sampling root |
| Gauge | `lilo.trainer.state` | One-hot operation state, observed/exported every five seconds |

| Span family | Additional exported attributes |
| --- | --- |
| Trainer and command identity | `lilo.trainer_instance_id`, `lilo.definition_id`, `lilo.boot_id`, `lilo.component`; `lilo.model_id`, `lilo.request_id` where there is one owner |
| Command | `lilo.seq_id`, `lilo.operation`, `lilo.example_count`, `lilo.input_tokens` where applicable; `lilo.incomplete=true` on graceful shutdown with unfinished work |
| Trainer phase/batch | `lilo.lane`, `lilo.operation`, `lilo.command_count`; aggregate `lilo.example_count`, `lilo.input_tokens` when known for all participants |
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

## Delivery and data handling

Spans are exported in background batches. Trainer-state reporting begins when
the engine server is constructed, after process initialization. Operations shorter
than the sampling interval may not appear in the state metric; their spans retain
the operation's start and end times.

Abrupt process termination can lose open or buffered spans. Graceful shutdown
marks unfinished command roots with `lilo.incomplete=true` and error status.
Buffered commands discarded during unloading also end with error status.
Deduplicated submissions reuse the original trace context while it remains in
the engine's cache of the most recent 2,048 accepted commands; older submissions
may have standalone control-plane spans.

Export failures do not change operation results. Lilo uses private OpenTelemetry
providers, leaving application-wide providers and instrumentation unchanged.
Credentials are configured on the server; API clients do not need access to the
telemetry destination.

Exported data excludes prompts, generated text, token arrays, gradients,
checkpoint contents, exception messages and stacks, API credentials, and metadata
other than the documented experiment labels. Lilo exports traces and metrics;
dashboards, notebooks, and retention policies are managed in the destination.
