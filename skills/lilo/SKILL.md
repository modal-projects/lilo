---
name: lilo
description: Work with Lilo training and sampling on Modal, including deployment, GPU resource costs, cold starts, checkpoint recovery, weight synchronization, and trainer utilization.
---

# Working with Lilo

Lilo provides a Tinker-compatible API backed by training and inference resources on Modal. This skill explains the resource and state semantics that matter when using it. Adapt the examples to the user's application and algorithm.

## Getting started

Install Lilo into the application project, for example with `uv add 'lilo @ git+https://github.com/modal-projects/lilo.git'`. Use the Python version and dependencies supported by that revision; scoped deployment support and runtime requirements are documented in `docs/scoped-runs.md`.

For an existing deployment, connect with its URL and API key:

```python
import tinker

service = tinker.ServiceClient(base_url=url, api_key=api_key)
```

To deploy in your own Modal workspace, use an existing Modal profile or authenticate with `uv run modal token new`. Select the intended Modal environment before creating secrets or deploying. Sampler access uses a Modal proxy token stored as `MODAL_PROXY_TOKEN_ID` and `MODAL_PROXY_TOKEN_SECRET` in the `lilo-proxy` secret in that environment; the repository README describes token creation and persistent-service setup.

Modal credentials authorize deployment, the proxy token authorizes sampler access, and the Lilo API key authorizes clients. Scoped deployment generates the client API key unless one is supplied. Existing-endpoint clients only need their URL and API key.

## Full fine-tuning (FFT)

FFT updates the full model and keeps model and optimizer state on a dedicated trainer. Its deployment, publication, and checkpoint costs differ from adapter training.

### Start a scoped deployment

**Prefer a scoped single-tenant deployment for a new FFT run.** It puts the engine, sampler capacity, and resource lifetime under the run's ownership. The alternatives are connecting to an existing endpoint, or operating a persistent shared service for multiple clients. A shared service can still allocate a dedicated trainer for each FFT model; a shared API does not mean shared FFT GPU state.

A scoped setup, using an illustrative built-in engine recipe:

```python
import lilo
import tinker
from lilo.engines import qwen3_5_4b_full_64k

engine = qwen3_5_4b_full_64k()
with lilo.run(engine=engine) as (url, api_key):
    service = tinker.ServiceClient(base_url=url, api_key=api_key)
    training = lilo.create_full_training_client(service, engine.model)
    # The application's training and sampling code runs within this lifetime.
```

An engine recipe defines the model, context limits, trainer/sampler GPU topology, and backend settings. Recipes come from `lilo.engines`; a custom recipe can live in the user's project. Sampler replica counts are a separate choice: scoped runs take `latest=lilo.Pool(min_containers=..., max_containers=..., scaledown_window=...)`. Client-level `rollout` settings belong to the shared-service interface and are rejected by scoped model creation.

A scope permits one active training model. Creating another full training client does not attach to the existing learner. Other processes can connect using the scope's URL and API key while its owner remains alive.

The process holding `lilo.run` owns the deployment. Its preemption, timeout, or disconnect can end the trainer's lifetime too. A controller retry opens fresh resources; it does not restore training. Normal context exit stops owned resources but does not automatically checkpoint. Completed checkpoints persist in the checkpoint Volume. Pinned samplers run in owned ephemeral child apps; abrupt owner death also ends those apps after Modal detects the lost owner, so cleanup is eventual rather than instantaneous.


### Modifying engines

Built-in recipes return frozen dataclasses. Customize one in the application with `dataclasses.replace`, including for nested settings; no Lilo registry edit is needed. For example, to change the sampler memory fraction:

```python
from dataclasses import replace
from lilo.engines import qwen3_5_4b_full_64k

base = qwen3_5_4b_full_64k()
engine = replace(
    base,
    name="my-engine",
    sampling=replace(base.sampling, memory_fraction=0.90),
)
# Pass engine to lilo.run(engine=engine).
```

The value is illustrative. Engine settings describe each replica's model, GPUs, parallelism, memory, and context/packing limits; `lilo.Pool` controls replica counts and idle retention. GPU count, parallelism, and memory requirements interact, so changing a GPU string alone may not produce a viable recipe. The configuration types are exposed from `lilo.engines`.

### Publishing weights for sampling

The trainer and inference replicas hold separate weights. Training an update does not by itself make those weights available to sampling. FFT publication captures and persists incremental weight deltas, which replicas then apply; its cost is separate from training execution.

Prefer the latest pool for training and other sampling. Use a pinned publication for specific evaluations that need a fixed weight version. These choices have different state and resource behavior:

```python
# Publish to the reusable latest-policy pool.
latest = training.save_weights_and_get_sampling_client()

# For an evaluation requiring a specific version, publish and pin it.
publication = training.save_weights_for_sampler(publication_name).result()
pinned = training.create_sampling_client(publication.path)
```

The **latest pool** follows publications from the learner. A handle establishes a minimum version, not a permanently fixed policy; later requests can use newer weights. Reusing the pool amortizes startup across updates.

A **pinned pool** serves the named publication. It is useful when sampling must refer to one exact set of weights, but each distinct version can bring separate inference allocation and startup costs, even when the latest pool is warm.

Neither API saves optimizer state or provides a full training checkpoint. Publication frequency trades synchronization overhead against weight freshness. Larger weight changes can increase delta size; the algorithm determines acceptable freshness, while Lilo determines the cost of making those weights available.

### Where cold-start latency appears

Trainer and sampler startup are separate. Obtaining a sampling handle does not mean inference replicas are ready. Cold starts can recur after resources scale down or when a new pinned pool is needed.

| Call | Possible startup work |
| --- | --- |
| Entering `lilo.run(..., warm=True)` | App and asset preparation, trainer allocation and loading before entering the body. `warm=False` defers trainer warmup. |
| `lilo.create_full_training_client(...)` or its async variant | Model creation and trainer readiness, including loading and distributed initialization when cold. Inference can still be starting when this returns. |
| First `training.forward_backward(...)` operation | First-use compilation; latency appears when waiting for the operation to complete. |
| First `sampler.sample(...)` or `await sampler.sample_async(...)` on a cold pool | Inference allocation, model loading, compilation, and required weight application. |

These initial timings can be much longer than steady-state calls. Keeping inference warm reduces repeated startup but incurs idle allocation cost.

### Checkpoints and recovery

`save_state(name)` saves full training state; `load_state_with_optimizer(path)` restores model and optimizer. Sampling publications cannot substitute for this. A replacement trainer needs restored state and a new sampling publication; in a surviving scope, old latest handles are rejected after trainer reassignment, so resumed sampling uses a new handle.

Full FFT checkpoints include optimizer state and can take minutes to write. The tradeoff is saving overhead versus expensive progress lost to interruption. Roughly one state-losing interruption over 12 hours is a useful planning assumption, not a platform guarantee or checkpoint timer.

If updates take a few minutes and saves take ten minutes, waiting for a save after every update can dominate the run. If an update itself takes an hour, saving each update may be worthwhile. Elapsed progress and save cost matter more than a universal step interval, including before the first useful checkpoint. A reproducible base model usually does not need a step-zero full save.

Capture is ordered with training, while persistence can overlap later work where supported. A ten-minute background write is not necessarily ten minutes of GPU stall. Awaiting it before submitting more work can introduce that stall; queuing more saves than storage can finish adds overhead without making recovery points arrive faster. Sampling needs published weights, not completion of an unrelated full-state save.

A checkpoint captured at update 20 still restores update 20 even if its write finishes during update 22. Its returned path and captured progress need to survive the controller, and it becomes a recovery point only after successful persistence. Until then, recovery uses the previous completed checkpoint. The checkpoint does not preserve the controller's Python memory.

An optimizer timeout can have an unknown outcome: a blind retry can apply an update twice. That differs from an isolated sampling failure, which does not itself establish trainer loss.

## Cost and utilization

Lilo costs follow allocated GPU resources and time, rather than a hosted per-token price. Training and inference have separate allocations. Small batches do not shrink the configured trainer topology, and an allocated trainer can cost money while waiting for input, weight publication, or the controller.

Think in terms of useful training progress per allocated GPU-hour. More inference capacity can reduce trainer waiting but also increase total spend. Warm replicas trade idle cost for lower startup latency. Overlapping sampling and training can help when the application's algorithm permits it; changing policy-lag or update semantics is an application decision.

Modal container logs and GPU metrics help distinguish startup, active training, and waiting between operations. A busy GPU during forward/backward does not imply good utilization across the whole loop. Lilo's optional OTLP telemetry can provide finer attribution; configuration is described in the installed revision's `docs/observability.md`.
