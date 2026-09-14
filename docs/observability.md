# Observability

Lilo exports OpenTelemetry traces and a sampled trainer-state metric over OTLP
HTTP/protobuf. Export is opt-in. No collector, Datadog Agent, telemetry volume, or
additional long-running service is required. Credentials stay in the server
processes; clients do not need access to the observability destination.

## Modal → Datadog

Add the following variables to the existing `lilo-api` Modal secret in the same
workspace and environment as the deployment. Merge these values into the secret;
retain its existing API authentication settings. Use a **Datadog API key** for
intake. A Datadog application key is not needed by Lilo.

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
Configure header values using the standard OTLP header encoding (percent-encode
values when necessary). Never commit the populated environment file or credentials.

Deploy Lilo after updating the secret:

```bash
MODAL_PROFILE=your-workspace MODAL_ENVIRONMENT=your-environment \
  uv run modal deploy -m lilo.providers.modal.app
```

The control plane, sampling workers, and new trainer containers receive the
configuration. An already-running trainer must finish and be replaced before it
uses new code or environment values. Existing commands are not retroactively
instrumented.

Run a training operation, then search Datadog APM for `service:lilo`. Use resource
names such as `lilo.command.forward_backward` and `lilo.trainer.forward_backward`;
Datadog may derive its displayed operation name from span kind. In a native span
search block, the duration column is **`@duration`**, not `duration`.

For trainer activity, use a timeseries such as:

```text
avg:lilo.trainer.state{lilo.trainer_instance_id:ENGINE_ID,lilo.lane:execution} by {lilo.operation}.fill(null)
```

Choose a stacked area display. Add separate charts for `checkpoint` and `sampler`.
Missing reports remain gaps. Datadog rollups may average several samples into
fractional values; those values are not GPU utilization. Configure APM retention
for the diagnostic spans you want searchable historically. Exported spans can
arrive before other spans from the same trace have finished or been indexed.

## Custom OTLP destination

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
The destination must be reachable from the exporting processes. Only
HTTP/protobuf is supported; this integration does not run a gRPC exporter.

A trace endpoint alone enables only traces. A metric endpoint alone enables only
the trainer-state metric. A general endpoint enables both. With no endpoints,
export is disabled. Set `OTEL_SDK_DISABLED=true` to disable both explicitly.
Lilo uses its own providers and does not replace the application's global
OpenTelemetry providers or enable automatic instrumentation.

## Experiment labels

Supply experiment identity through the existing model `user_metadata` API:

```python
training = create_full_training_client(
    service,
    "Qwen/Qwen3.5-4B",
    user_metadata={
        "run_id": "codegolf-qwen4b-v1",
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

Labels attach to command roots, command queue/result spans, control submissions,
and trainer execution/lifecycle spans. Sampler artifacts snapshot these labels;
sessions and submitted sampling tasks carry them to sampling roots and HTTP
attempt spans. No extra model lookup is required per sample. Older artifacts and
sessions without labels remain untagged; base-model-only sampling has no model
experiment identity.

A trainer execution receives a label only if **all** its participating commands
have that same label. A batch crossing experiments has links to each command and
is not attributed to a single experiment. Trainer-state metrics describe the
physical trainer and intentionally do not carry experiment labels, which could
misattribute shared work and increase metric cardinality.

## Trace structure and lifecycle

Each accepted engine command has a root `lilo.command.<operation>` span, beginning
at control-plane submission receipt and ending when its trainer result is ready.
The trainer owns completion, so client polling is not necessary to close it.
Control submission, queue wait, and result-ready are children of that root.

The engine returns the canonical root context in its authenticated HTTP response.
The control plane records its submission span with explicit start/end times once
that context is known. Retries accepted by engine deduplication attach to the
original command rather than creating another execution. Rejected submissions
are standalone control spans. A successful model-create/unload HTTP response is
also a submission boundary, not a trainer-lifetime span.

Actual backend executions are **separate traces**, with span links to every
participating command, including when there is only one command. Thus a merged
batch appears once, with aggregate workload counts, rather than being duplicated
under several command roots. Capture, persistence, and waits for previous
persistence are distinct physical intervals linked to their command. Persistence
can overlap a subsequent execution. Lifecycle accept/unload spans are independent.

Counts refer to logical input examples and their supplied text tokens, before
backend packing/padding. Input-token count is omitted if any input chunk has no
known text-token length. Loss-token count, packed-microbatch count, and pure GPU
forward/backward timing are not inferred. Executor spans include backend transport
and synchronization; they are wall-clock timings, not a GPU profiler.

## Export inventory

All spans include standard OpenTelemetry identity, parent/link context, start/end
timestamps, status, and resource attributes (`service.name`, SDK metadata, and
configured `OTEL_RESOURCE_ATTRIBUTES`). No log exporter is installed.

| Signal | Name | Boundary / purpose |
| --- | --- | --- |
| Span | `lilo.command.<operation>` | Submission receipt → trainer result ready; `forward`, `forward_backward`, `optim_step`, `save_weights`, `load_weights`, `save_weights_for_sampler`, internal `skip` |
| Span | `lilo.control.submit` | HTTP submission work, attached to the canonical command root |
| Span | `lilo.control.<operation>` | Standalone rejected/unhanded-off submission or model-create/unload request |
| Span | `lilo.trainer.queue` | Accepted into engine buffer → dequeued |
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
| SGLang attempt evidence | `sglang.request_id`, `sglang.queue_s`, `sglang.prefill_s`, `sglang.post_prefill_to_finish_s`, `sglang.cached_tokens`, `sglang.prompt_tokens`, `sglang.completion_tokens` |

SGLang fields are present only when the backend returns the corresponding evidence.
Post-prefill-to-finish includes decode and final processing; it is deliberately not
called pure decode time. Missing fields remain absent, not zero. Raw backend timing
payloads and arbitrary numeric training results are not exported.

| Gauge | Values and labels |
| --- | --- |
| `lilo.trainer.state` | `1` for current operation, explicit `0` for other operations in that lane. Labels: `lilo.trainer_instance_id`, `lilo.definition_id`, `lilo.boot_id`, `lilo.component`, `lilo.lane`, `lilo.operation` |
| Execution lane | `idle`, `accept`, `unload`, `forward`, `forward_backward`, `optim_step`, `save_weights`, `load_weights`, `save_weights_for_sampler`, `skip` |
| Checkpoint lane | `idle`, `save_weights` |
| Sampler lane | `idle`, `save_weights_for_sampler` |

## Delivery limits

Spans are exported in background batches. State reporting starts after the engine
server is constructed and does not measure earlier process initialization. Short
operations can fall entirely between metric samples. Finished child spans may be
visible before their root finishes.

This telemetry is diagnostic, not a durable command ledger. Abrupt process loss
can lose open/buffered spans, including a command root; a new trainer does not
fabricate completion for the old process. Graceful shutdown marks unfinished roots
as incomplete. Unloaded buffered commands finish with error status. Canonical
retry contexts are bounded to the most recent 2,048 accepted commands per engine;
older deduplicated submissions may appear as standalone control submissions.
Experiment labels join recovery attempts without pretending they were one
uninterrupted execution.

Prompts, generated text, token arrays, gradients, checkpoint contents, exception
messages/stacks, API credentials, and unapproved metadata are excluded. Backend
export failures do not change operation results. Notebook/dashboard creation is an
operator action, never part of request execution.
