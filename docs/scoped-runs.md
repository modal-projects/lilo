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

After trainer loss, create a new full training client through the same service,
restore a saved checkpoint, and publish a new sampling client:

```python
trainer = lilo.create_full_training_client(service, engine.model)
trainer.load_state_with_optimizer(checkpoint_path).result()
latest = trainer.save_weights_and_get_sampling_client()
```

Use the new latest client; old latest clients receive HTTP 410. Base sampling and
previously pinned versions remain available. Recovery is explicit: the deployment
does not restore checkpoints or replay failed updates automatically.

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

Pinned apps idle for roughly ten minutes are stopped. Active sampling requests
prevent eviction, and existing pinned handles recreate an app on their next
request. Modal retains stopped app history.
