# Scoped runs

Start an engine and connect with Tinker. Other processes can use the same URL and
API key while the context is open.

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
One training model can be active at a time. Samplers start separately on demand;
`latest` sets the latest sampler’s replica limits and scaledown window. Its minimum activates
when a model is created. Base and pinned samplers have a zero minimum.

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

Pinned sampling refreshes `(model_id, version)` demand in the existing ownership
Dict and resolves its route on every sampling retry. Missing routes wait with
backoff while the owner polls demand once per second and opens missing apps.
No Lilo installation or Modal credentials are required in external Tinker clients.
The owner holds app contexts in memory and limits concurrent app creation to two;
there is no version-count cap or persistent provisioning-state machine.

Demand idle for ten minutes is removed together with its route, and the owner
closes the corresponding app. Active sampling leases prevent idle eviction;
abandoned leases expire. Stale demand also expires if an app never started.
A sampling retry recreates demand lost during cleanup; old-app cleanup cannot
remove a replacement route or fresh demand. Existing pinned handles therefore
recreate an idle-evicted app on their next request. The scan grows with recent
requested versions, not all historically published versions.

[Ephemeral pinned lifecycle validation](scoped-pinned-ephemeral-result.json) used
CPU HTTP substitutes with the production owner, app factory, and sampling retry
loop: missing-route/startup retries, idle eviction and recreation, and normal exit
passed. Hard-cancelling the owner stopped its child with a minimum container in
about three minutes, with zero containers remaining. This validates ownership and
routing; it is not a new GPU inference validation.

## Smoke test

`python scripts/scoped_smoke.py` runs a real training update, base sampling from
another Tinker client, latest sampling, and pinned-version sampling, then exits
the context. Results are written to `/tmp/lilo-scoped-smoke.json`.

`python scripts/scoped_recovery_smoke.py` additionally saves a full checkpoint,
cancels the trainer invocation, restores through a new full training client, and
checks replacement latest sampling, old-handle rejection, and pinned continuity.

Earlier deployed-pool validation: [verified run results](scoped-smoke-result.json): training, base/latest/pinned
sampling, and pinned-app shutdown before the parent, with zero remaining containers.

Earlier deployed-pool validation: [verified recovery results](scoped-recovery-result.json): trainer cancellation,
checkpoint restoration, replacement latest sampling, old-handle rejection, and
pinned-app reclamation followed by recreation through the same handle. All test
apps stopped with zero remaining containers.

## OpenTelemetry

Pass an OTLP configuration secret to `lilo.run(telemetry_secret=...)` to enable
traces on the scoped API, trainer and sampling worker, and five-second trainer
operation gauges. Use an OTel-only secret; keep API/proxy credentials separate.
The codegolf example forwards only `OTEL_*` keys from its existing controller
secret. Exporting is best effort and does not require a telemetry volume or a
Datadog-specific dependency.

Set `user_metadata={"run_id": "my-run", "attempt_id": "replacement-1"}` when
creating the full training client. Lilo validates these two labels, snapshots them
through models, artifacts and sampling sessions, and propagates `lilo.run_id` / `lilo.run_attempt_id` to trainer
operations and sampling retries. Use the same run ID and a fresh attempt ID after
replacement. These are correlation labels, not authorization boundaries. Prompts,
generated code and arbitrary metadata are excluded. Metrics exported before a
checkpoint rollback remain historical observations; the application checkpoint
ledger determines committed progress.

The [observability guide](observability.md) defines the trace boundaries and metric
labels. Physical trainer-state metrics have no model experiment labels. When one
scoped deployment belongs exclusively to one experiment, its owner may include
`lilo.run_id` in `OTEL_RESOURCE_ATTRIBUTES`. Scoped trainers also emit this as a
metric datapoint tag so direct OTLP intake can filter the deployment's metrics.
Do not apply a single experiment resource label to a shared deployment.
