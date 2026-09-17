# Codegolf rollout stall diagnosis, 2026-09-17

Experiment: `tailrl-hero-v1`, independent branch `codex/multilora-codegolf`.
The observations below describe the original run; the fixes now also live in PR #15.

## Findings

All eight rollout containers were still the original containers, started between
04:56 and 05:02 UTC. At the final inventory, seven had 256 registered LoRA
versions and sampled GPU utilization of 0%; the eighth had 251 versions and
100% utilization. These are point samples, not averaged utilization metrics.
The configuration sets `max_loaded_loras=256`. Versions from all three trainer
attempts remain represented in the registry.

Historical logs show the first LRU eviction at 08:37:41 UTC, when the adapter
count reached 257. Later sidecar tracebacks time out inside
`_ensure_adapter`'s `/load_lora_adapter` call, before generation. The sidecar
holds its registration lock during that call, so other registrations on that
replica wait behind it. The HTTP timeout is 3000 seconds. Existing registered
adapters can still generate, explaining occasional progress despite the stalls.

The installed SGLang build is `modal-projects/sglang` revision
`d050d06437d96196fc68d5b4e5c246408790d537` (v0.5.17 overlay), pinned by
`src/lilo/providers/modal/rollout_image.py`.

## Reproduced reference-count bug

In the installed `python/sglang/srt/managers/tokenizer_manager.py`:

- `_handle_batch_output`, lines 2468–2499, releases the LoRA reference when a
  request finishes, including an aborted request.
- `_wait_one_response`, lines 1742–1746, then invokes
  `_handle_abort_finish_reason` for the returned abort.
- `_handle_abort_finish_reason`, lines 1641–1655, releases the same reference
  again for status 503, 500, or 499.

One request can therefore take the counter from 1 to 0 to -1.
`ConcurrentCounter.wait_for_zero` requires equality with zero.
`tokenizer_control_mixin.py:581–586` unregisters an eviction victim and waits for
that counter before unloading it, while holding `lora_update_lock`.
Consequently a negative counter can block all subsequent adapter updates.

`scripts/repro_sglang_lora_abort_counter.py` extracts the actual installed abort
helper and counter with AST and runs them in a separate CPU process. After the
common finished-output release, all three abort codes produced -1 and blocked
the zero wait. The single-release control completed. This is an isolated
code-path reproduction, not an inspection of private counters in the live
processes. The live capacity, eviction, timeout, and idle-GPU evidence strongly
supports this as the cause. Real `/generate` 503 responses occurred at 06:31,
before the first eviction.

## Why FFT differs

`src/lilo/providers/modal/fft_pool_app.py` starts SGLang with `enable_lora=False`.
Its sidecar (`src/lilo/inference/full_sidecar.py`) uses Stitch's staged full-weight
updates (`stage_weight_update` / `update_weights_from_cpu`). It does not acquire
LoRA registry references or accumulate named adapter versions, so it does not
exercise this eviction path. Multi-LoRA explicitly enables LoRA and registers
a new immutable adapter name for each published version.

Controller preemptions at approximately 06:38 and 07:47 UTC explain the training
restarts. Clients with a step-50 checkpoint restored it; the others restarted at
zero. These restarts increase the number of old adapters but are separate from
the persistent inference stall.

## Fix and validation

The PR #15 lifetime fix is carried into this experiment branch. Each logical
request owns a registry reference shared with its parallel children. Errors before
dispatch release immediately; errors after dispatch retain state until scheduler
completion. Acquisition is cancellation-safe and atomic with registry eviction.
The response consumer no longer releases or removes completed request state.

The patched image passes 32 SGLang lifetime tests. A one-H200, 64K-context probe
with a two-version cache passed oversized-input rejection, parallel sampling,
streaming HTTP disconnection, parallel-group abortion, queue rejection, repeated
eviction, and implicit reload of an old version. See `scripts/validate_lora_eviction.py`.

Future codegolf launches use 64 retained adapter versions per replica and save
adapter + optimizer checkpoints every 20 updates. The 3,000-second sidecar HTTP
timeout and per-replica registration lock are unchanged. A timeout alone would
not repair a backend load stuck in SGLang; poisoned old replicas need replacement.
These changes do not modify the already-running deployment.

## Reproduce plots and inventory

From this worktree with its `.venv` and Modal environment configured:

```bash
MODAL_ENVIRONMENT=kailash-dev .venv/bin/python scripts/plot_multilora_codegolf.py
MODAL_ENVIRONMENT=kailash-dev .venv/bin/python scripts/inspect_multilora_codegolf.py
```

Plots and diagnostic receipts are under
`scripts/results/multilora-codegolf/tailrl-hero-v1/`. Plots use each model ID as a
separate attempt, keep previous attempts gray, and show the latest saved
held-out evaluation at each step. `seconds` excludes publication/evaluation
and includes rollout waiting; unfinished waits have no metric yet.
