# Scoped runs

Start Lilo with an engine definition and use the returned URL and API key with
Tinker. Other processes can connect using the same credentials.

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
    # Existing Tinker forward_backward, optim_step and sampling methods.
```

`warm=True` starts one trainer and waits for it to load before entering the
`with` body. Samplers start separately on demand. It uses an explicit
invocation, not a minimum-container setting. One training model can be active
at a time. Trainers stay alive while the context is open rather than being
reclaimed by an idle cleaner.

`latest` controls replica counts and the scaledown window. Its minimum activates
when a model is created. Base and pinned-version samplers always have a zero minimum.
Pinned versions are created on demand through the normal Tinker sampling API;
there is no separate pinned-pool configuration to provide.

Apps default to `lilo-<hash>`. The sampler functions are `base_sampler` and
`latest_sampler`.

## Replacing a lost trainer

After confirmed trainer loss, create another full training client using the same
service. Creating a second model while the first is active is rejected.

```python
trainer = lilo.create_full_training_client(service, engine.model)
trainer.load_state_with_optimizer(checkpoint_path).result()
latest = trainer.save_weights_and_get_sampling_client()
```

The new model has its own publication history. Latest replicas follow its assignment. Stitch drains old generations, then
the scoped CPU-delta sidecar retires the old container. Modal starts a fresh
replica at the same endpoint to load the new model’s publications. A request for the new model waits until a replica has that
model's required version. Old latest clients receive HTTP 410 after reassignment: use the
new sampling client. Base sampling and pinned publications are unaffected.
The app does not automatically restore checkpoints or replay failed updates.

## Your own engine

An engine definition specifies the complete execution recipe: model source,
context and packing limits, GPU topology, backend settings, images, and sampler
configuration. Pool limits change replica counts, not that recipe.

Create a Python file in your own project; no Lilo checkout or registry edit is needed:

```python
# my_engine.py
from dataclasses import replace
from lilo.engines import qwen3_6_27b_full_64k

original = qwen3_6_27b_full_64k()
engine = replace(
    original,
    name="my-27b-recipe",
    training=replace(original.training, seed=42),
)
```

Pass that object to `lilo.run(engine=engine)`. Full configuration types are exposed
in `lilo.engines`: `Engine`, `EngineModelConfig`, `OptimizerConfig`, and
`SamplingConfig`. Custom images are ordinary `modal.Image` objects. Changes to a
recipe need their own memory and correctness validation.

Use your existing Modal profile and sampler proxy credentials. The default proxy
secret is `lilo-proxy`; override it with `proxy_secret=modal.Secret.from_name(...)`.
The API key is generated per run unless supplied explicitly.

## Cleanup

The API, trainer, base and latest samplers belong to one ephemeral app. On normal
context exit, Lilo stops any separately deployed pinned samplers first, then the
main app. During a run, pinned apps idle for ten minutes are stopped; active
sampling calls hold leases that prevent reclamation. Existing pinned handles
recreate the app on their next request. Saved versions remain available, and
Modal retains stopped app history. Saved checkpoints remain; exit does not implicitly save a checkpoint.

If the owning process is killed, the main app stops on disconnect. Pinned apps may
remain registered but scale to zero when idle. Modal retains stopped app history.

## Smoke test

`python scripts/scoped_smoke.py` runs a real training update, base sampling from
another Tinker client, latest sampling, and pinned-version sampling, then exits
the context. Results are written to `/tmp/lilo-scoped-smoke.json`.

`python scripts/scoped_recovery_smoke.py` additionally saves a full checkpoint,
cancels the trainer invocation, restores through a new full training client, and
checks replacement latest sampling, old-handle rejection, and pinned continuity.

[Verified run results](scoped-smoke-result.json): training, base/latest/pinned
sampling, and pinned-app shutdown before the parent, with zero remaining containers.

[Verified recovery results](scoped-recovery-result.json): trainer cancellation,
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
`lilo.run_id` in `OTEL_RESOURCE_ATTRIBUTES` to filter that deployment's metrics.
Do not apply a single experiment resource label to a shared deployment.
