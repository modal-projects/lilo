# Profiling a training step

The Miles LoRA trainer path supports an opt-in `torch.profiler` capture plus
always-on per-phase wall-clock timing. Profiling is **off by default**: it is
enabled only when `LILO_TORCH_PROFILE_STEP` is set, and unsetting it (or
deploying without it) disables the profiler entirely; the other two variables
have no effect on their own. The per-phase timers are always on and cannot be
disabled.

Set these environment variables in the deploying shell before `modal deploy`;
`trainer_deployment_env()` forwards them onto the trainer container:

- `LILO_TORCH_PROFILE_STEP`: 0-based optimizer step index to trace. When the
  first `forward_backward` of that step arrives, the selected rank-local Ray
  actors start a CPU+CUDA profiler and the backend process starts a CPU-only
  controller trace. The capture stops just before the first forward of the
  following step. Rank traces cover the actors' forward/backward, optimizer and
  slot export; the controller trace also covers the post-step sampler-weight
  save and publish, which run only in the backend process.
- `LILO_TORCH_PROFILE_RANKS`: comma-separated trainer ranks to trace, or
  `all`. Defaults to `0`; traces are large (~250 MB per rank per step).
- `LILO_TORCH_PROFILE_DIR`: output directory. Defaults to
  `<checkpoint volume mount>/torch-profile/<LILO_DEFINITION_ID>`. Each rank
  writes `rank{N}.trace.json.gz` (Chrome trace, loadable in
  `chrome://tracing` or Perfetto) and `rank{N}.key_averages.txt`; the backend
  writes `controller.trace.json.gz` / `controller.key_averages.txt`.

Deploying a separate app for a profiling run:

```bash
LILO_TORCH_PROFILE_STEP=2 \
  uv run modal deploy -m lilo.providers.modal.app
```

Always-on timing: every backend op prints one `lilo_step_timing` JSON line to
the trainer log per phase, and `optim_step` responses carry `timing/*` metrics
(`timing/forward_backward_s`, `timing/optim_step_s`, `timing/trainer_step_s`,
`timing/idle_wait_s`, `timing/save_sampler_weights_s`,
`timing/publish_weights_s`, per-phase `_calls` counts, and
`timing/optimizer_step`). These ride the existing `optim_step` response metrics
that the Tinker client/cookbook merge into per-step logged metrics, rather than
the trainer logging to W&B directly. Note that sampler save/publish for step k
runs after `optim_step` k returns, so `timing/save_sampler_weights_s` and
`timing/publish_weights_s` reported at step k+1 refer to the checkpoint taken
after step k.

To copy the rank-0 trace off the checkpoint volume into the W&B run's **Files**
tab as `profiler/rank0.trace.json.gz` (downloadable straight into Perfetto; pass
`--trace` to pick another file):

```bash
uv run scripts/upload_torch_profile.py \
  --dir <trace dir> --entity modal-labs --project miles-lora-longcontext \
  --run-id <wandb run id>
```

`WANDB_API_KEY` is read from the environment.

Request-path attribution: set `LILO_REQUEST_TIMING=1` alongside the profiler
envs to have every hop emit one `lilo_request_mark` JSON line with an epoch
`ts`. Marks cover the control-plane receive/decompress/forward and
`retrieve_future` long-polls (`cp.*`), engine decode, submit, execution, and
`retrieve_future` (`engine.*`), and the localhost JSON hop to the backend
(`engine.backend_post.*`, `backend.request.*`). Set it in the deploying shell
before `modal deploy`; it forwards to the server function and trainers, and
`scripts/tinker_client_timing.py` provides matching client-side `client/*`
metrics for a cookbook training loop.
