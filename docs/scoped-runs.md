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
    max_trainers=1,
    latest=lilo.Pool(min_containers=1, max_containers=2),
) as (url, api_key):
    service = tinker.ServiceClient(base_url=url, api_key=api_key)
    trainer = lilo.create_full_training_client(service, engine.model)
    # Existing Tinker forward_backward, optim_step and sampling methods.
```

`warm=True` starts one trainer and waits for it to load. It uses an explicit
invocation, not a minimum-container setting. `max_trainers` limits the number of
training models; each gets its own latest sampler pool. Trainers stay alive while
the context is open rather than being reclaimed by an idle cleaner.

`latest` controls replica counts and the scaledown window. Its minimum activates
when a model is created. Base and pinned-version samplers always have a zero minimum.
Pinned versions are created on demand through the normal Tinker sampling API;
there is no separate pinned-pool configuration to provide.

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
main app. Saved checkpoints remain; exit does not implicitly save a checkpoint.

If the owning process is killed, the main app stops on disconnect. Pinned apps may
remain registered but scale to zero when idle. Modal retains stopped app history.

## Smoke test

`python scripts/scoped_smoke.py` runs a real training update, base sampling from
another Tinker client, latest sampling, and pinned-version sampling, then exits
the context. Results are written to `/tmp/lilo-scoped-smoke.json`.

[Verified run results](scoped-smoke-result.json): training, base/latest/pinned
sampling, and pinned-app shutdown before the parent, with zero remaining containers.
