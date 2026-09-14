# Scoped single-engine runs (draft)

`lilo.run` provisions one ephemeral Modal app and yields `(url, api_key)`.
Use ordinary Tinker clients in the owning process or another process. This initial
implementation requires Python 3.12, matching the bundled serialized runtime images.
It is an experimental API; it does not change the shared deployment entry point.

```python
import lilo
import tinker
from lilo.engines import qwen3_5_4b_full_64k

engine = qwen3_5_4b_full_64k()
with lilo.run(
    engine=engine,
    warm=True,
    max_trainers=1,
    latest=lilo.Pool(min_containers=1, max_containers=2),
    pinned=lilo.Pool(max_containers=2, scaledown_window=300),
) as (url, api_key):
    service = tinker.ServiceClient(base_url=url, api_key=api_key)
    trainer = lilo.create_full_training_client(service, engine.model)
    # Existing Tinker forward_backward, optim_step and sampling APIs.
```

Install the package into your project; no repository checkout or registry edits
are needed to define an engine. The existing Git-based distribution install still
applies (`uv add 'lilo @ git+https://github.com/modal-projects/lilo.git@<revision>'`);
this PR does not publish a PyPI release or prebuilt runtime images. Backend images
are built by Modal and contain the pinned backend dependencies. A Modal profile
and the existing allowed sampler-proxy credentials are required. Pass your own
`proxy_secret=modal.Secret.from_name(...)`, or use the existing `lilo-proxy` secret.
The returned API key is generated for this run and is never printed by Lilo.

## Ownership and capacity

- API, trainer, base and latest samplers belong to the same ephemeral app.
- Trainer min_containers is **zero**. `warm=True` explicitly starts one invocation
  and waits for its backend to be ready. The scoped app registers no idle cleaner
  or demand reconciler; the trainer remains allocated while waiting for requests.
- `max_trainers` defaults to one and caps separate training models/containers.
  One latest slot is registered per possible trainer. This draft does not reuse
  model slots after explicit unload; use a new run when replacing a model.
- Latest minimum containers are activated when a model claims its slot, avoiding
  allocating sampler GPUs for nonexistent models. Base minimum is always zero.
- Pinned-version samplers are deployed on demand in separate apps with minimum
  **zero**. The `pinned` limits apply per version, not globally across all versions.
- Pool settings change replica counts, not the engine's per-replica GPU topology.
- The default 4B recipe uses 4 H100s per trainer and 1 per sampler. Latest minimum
  one plus an active base/pinned sampler can bring the smoke test to 6+ H100s.

Normal exit closes admission and drains the serialized pool-creation function,
then stops recorded pinned apps before exiting the parent app context. Child
names are recorded before deployment so failed/partial deployments are tracked.
Cleanup attempts every child and reports failures instead of claiming success.
Modal retains stopped app history; stopping does not erase that history.

On abrupt owner death the main ephemeral app disconnects and stops; persistent
pinned apps may remain registered. Their zero minimum permits scaledown after
requests finish. There is no claim of immediate child shutdown on SIGKILL.
Completed checkpoints are preserved. No expensive implicit checkpoint runs on exit.
The current trainer invocation has a maximum 24-hour duration; this API does not
make optimizer state survive that deadline, backend failure, or replacement.

## Engine authoring

`lilo.engines.Engine` is a complete recipe: model, trainer/sampler hardware,
`EngineModelConfig`, `SamplingConfig`, images, CPU/RAM, timeout and backend env.
The model string identifies weights/architecture; it does not choose topology,
recomputation, packing, optimizer, or sampler tuning. There are no context/GPU
mutation arguments on `lilo.run`.

```python
# my_engine.py
from dataclasses import replace
from lilo.engines import qwen3_6_27b_full_64k

original = qwen3_6_27b_full_64k()
engine = replace(
    original,
    name='my-27b-recipe',
    training=replace(original.training, seed=42),
)
```

Full typed Megatron configuration is available from
`lilo.engines.EngineModelConfig` / `OptimizerConfig`; backend provider overrides
retain existing semantics. Recipes are shallow-frozen dataclasses; do not mutate
nested mappings after creating a run. Structural validation does not prove memory
fit or numerical correctness. A modified recipe needs its own validation.

Unpinned recipes reuse the existing asset directory, like the shared recipes.
An explicit `revision` triggers HF snapshot preparation. Do not run different
revisions into the same asset directory concurrently: use distinct
`training.hf_checkpoint` paths under `/assets` in your definitions. Content-addressed
asset preparation is future work, not a guarantee of this draft.

## Smoke test

Run `python scripts/scoped_smoke.py` with the Modal profile/environment configured.
It performs real forward/backward and an optimizer update, then base sampling from
an independent Tinker client, latest sampling, and named pinned sampling. It writes
`/tmp/lilo-scoped-smoke.json` without credentials and exits the context. Verify the
recorded apps are stopped with no running containers afterward.
