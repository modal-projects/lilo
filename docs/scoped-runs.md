# Scoped runs

Use `lilo.run` to create a dedicated trainer and sampling deployment for one
training job. Connect with the Tinker SDK inside the context. Other processes
can use the same URL and API key while the context stays open.

```python
import lilo
import tinker
from lilo.engines import qwen3_5_4b_full_64k

engine = qwen3_5_4b_full_64k()
with lilo.run(
    engine=engine,
    warm=True,
    latest=lilo.Pool(min_containers=1, max_containers=2),
) as (url, api_key):
    service = tinker.ServiceClient(base_url=url, api_key=api_key)
    trainer = lilo.create_full_training_client(service, engine.model)
    # Use Tinker's training and sampling methods.
```

`warm=True` waits for the trainer to load before entering the `with` body.
One training model can be active at a time. Samplers start separately, so a ready
trainer does not mean the first sample will be fast. `latest` sets the replica
limits and idle retention time for the latest-policy pool. Its minimum takes
effect when a model is created. Base and pinned samplers have a zero minimum.

Use your Modal profile and the `lilo-proxy` secret, or pass
`proxy_secret=modal.Secret.from_name(...)`. An API key is generated for each run
unless you supply `api_key`. App names default to `lilo-<hash>`.

## Recovering a lost trainer

When a training request reports HTTP 410 with `error="model_lost"`, create a
replacement through the same service using a saved full-training checkpoint:

```python
trainer = service.create_training_client_from_state_with_optimizer(checkpoint_path)
latest = trainer.save_weights_and_get_sampling_client()
```

This works within the existing `lilo.run()` scope even when Modal restarted the
same trainer invocation in a new process. The replacement uses the available
trainer; a second model is still rejected while the first model is healthy or
initializing. The old training client remains lost.

Alternatively, create a full training client with
`lilo.create_full_training_client(service, engine.model)` and explicitly call
`trainer.load_state_with_optimizer(checkpoint_path).result()`.

Use the new latest client; old latest clients receive HTTP 410. Base sampling and
previously pinned versions remain available. Recovery is explicit: the deployment
does not restore checkpoints or replay failed updates automatically.
Resume the data iterator and step counter from the saved checkpoint's position;
updates since that checkpoint must be replayed. No background health polling is
required: application code can recover when an ordinary training request fails.

## Custom engines

An engine defines the model, context limits, GPU layout, backend settings, and
images. Define one in your own Python module and pass it to `lilo.run`:

```python
from dataclasses import replace
from lilo.engines import qwen3_6_27b_full_64k

original = qwen3_6_27b_full_64k()
engine = replace(
    original,
    name="my-27b-recipe",
    training=replace(original.training, seed=42),
)
```

Configuration types are exported from `lilo.engines`: `Engine`,
`EngineModelConfig`, `OptimizerConfig`, and `SamplingConfig`.
Custom images are ordinary `modal.Image` objects.

## Cleanup

Normal exit closes pinned sampler apps before the parent app. Both are ephemeral
and stop after Modal detects owner disconnect if the owning process dies.
Saved checkpoints remain; exiting does not save a checkpoint automatically.

Pinned versions get separate sampler apps, created when requested. If an app
is missing, sampling retries while the owner starts it. The owner checks for
requests once per second and starts at most two apps concurrently. There is no
fixed limit on the number of pinned versions, so requesting many versions can
allocate many separate apps.

After ten minutes without demand, an idle pinned app is stopped. Active sampling
requests prevent this cleanup. The same client handle will recreate the app on
its next request, which can incur another cold start. External Tinker clients
need only the API URL and key; they do not need Lilo or Modal credentials.

A CPU test of the ephemeral pinned apps passed startup retries, idle eviction,
recreation, and normal exit. Killing the owner stopped its child app in about
three minutes, with zero containers remaining. The test used substitute HTTP
servers, so it checked ownership and routing rather than GPU inference.
Its local report, `scoped-pinned-ephemeral-result.json`, is not included here.

## Smoke test

`python scripts/scoped_smoke.py` runs a real training update, base sampling from
another Tinker client, latest sampling, and pinned-version sampling, then exits
the context. Results are written to `/tmp/lilo-scoped-smoke.json`.

`python scripts/scoped_recovery_smoke.py` additionally saves a full checkpoint,
cancels the trainer invocation, restores through a new full training client, and
checks replacement latest sampling, old-handle rejection, and pinned continuity.

Earlier smoke and recovery tests used deployed sampler pools. They covered
training, base/latest/pinned sampling, checkpoint restore, old-handle rejection,
and recreation of an idle pinned pool. Both stopped all test apps with zero
containers remaining. Their local reports, `scoped-smoke-result.json` and
`scoped-recovery-result.json`, are not included here. For checked-in findings
and reproduction commands, see [E2E results](scoped-e2e-results.md) and
[E2E testing](scoped-e2e-testing.md).

## OpenTelemetry

Pass a Modal secret containing OTLP settings to `lilo.run(telemetry_secret=...)`
to export traces from the API, trainer, and sampling workers. Lilo also reports
the trainer's current operation every five seconds. OTLP is the protocol used to
send these measurements to Datadog or another OpenTelemetry-compatible service.

Keep this secret limited to telemetry settings; use separate secrets for API and
proxy credentials. The codegolf example copies only `OTEL_*` keys from its
controller secret. Export failures do not fail training.

Set `user_metadata={"run_id": "my-run", "attempt_id": "replacement-1"}` when
creating the full training client. Lilo attaches these labels to trainer and
sampling traces, including retries. Keep the run ID and choose a new attempt ID
when replacing a trainer. Labels group telemetry; they do not control access.
Prompts, generated code, and other user metadata are excluded.

Already exported metrics remain visible after a checkpoint rollback. Use the
application's saved checkpoint records to determine which steps were retained.

The [observability guide](observability.md) defines the trace boundaries and metric
labels. Physical trainer-state metrics have no model experiment labels. When one
scoped deployment belongs exclusively to one experiment, its owner may include
`lilo.run_id` in `OTEL_RESOURCE_ATTRIBUTES`. Scoped trainers also emit this as a
metric datapoint tag so direct OTLP intake can filter the deployment's metrics.
Do not apply a single experiment resource label to a shared deployment.
