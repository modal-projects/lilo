# Deterministic Miles training

This optional path makes the supported Qwen3.5-9B-Base multi-LoRA workload
numerically reproducible between isolated and packed clients. It is stacked on
the Miles backend. Ordinary deployments retain their existing configuration.

## Enable the complete path

Build the patched FA3 wheel and its binary manifest from pinned source:

```bash
.venv/bin/python scripts/export_fa3_test_wheel.py
export LILO_MILES_FA3_ARTIFACTS="$PWD/scripts/results/fa3_deterministic/artifacts"
```

Set that environment variable before importing or deploying the Modal app. It
selects the patched trainer image; select the
`qwen3_5_9b_base_miles_lora_deterministic` deployment definition for the six-client
trainer, or `qwen3_5_9b_base_miles_lora_deterministic_single` for an isolated trainer.
Both definitions enable FP64 reductions, deterministic model/loss/GDN/batching
configuration, and deterministic FA3 rollout inference. They use the existing
Miles backend, model assets, publication, and inference pool infrastructure.

The artifact directory is a generated local build output, not a committed wheel.
The image builder checks its revision and wheel checksum. Inside the image,
installed CUDA binary hashes must match the manifest before Transformer Engine's
head-dimension-256 restriction is relaxed. Use artifacts produced by the included
builder; a manifest is an integrity check, not independent proof of correctness.

Without the artifact option, `deterministic_attention=True` retains the FA2
fallback. That fallback is repeatable for a fixed batch, but is not the validated
path for exact parity across different client packing. The image option alone
also does not enable the trainer settings: use the deterministic definition.

## Runtime changes

- Set the model's batch-invariant mode before model construction.
- Accumulate selected tensor-parallel collectives and vocabulary sums in FP64,
  casting back once; model weights and GEMMs retain their existing precision.
- Compile cross-entropy helpers with dynamic shapes and deterministic compiler
  options to avoid cold-start shape and timing-dependent reduction choices.
- Pin FLA normalization and recurrence Triton configurations, including nested
  autotuners, before training starts.
- Preserve each adapter's isolated microbatch boundaries and accumulation order,
  then combine whole microbatches from different clients within the token budget.
- Enable deterministic inference and retain FP32 GDN prefix-cache accumulators.
- Use the patched FA3 hdim256 backward and the guarded Transformer Engine exception.

The FA3 patch uses the existing semaphore-ordered dQ TMA accumulation path. The
SM90 deterministic specialization uses a 64x32 tile to fit its shared-memory
staging. The normal 64x80 specialization is unchanged. This trades throughput for
repeatability; it does not serialize the entire trainer or disable grouped LoRA
GEMMs. The patch is a local upstream candidate, not an upstream release.

## Scope and validation

The supported setup is Hopper SM90, head dimension 256, text-only training with
DP=1 and VPP=1, and one update per dynamic work unit. The supplied FA3 build
includes only the required head-dimension-256 specializations; other head sizes,
GPU architectures, and forward specializations are outside this path.

Matching training trajectories additionally requires identical initial weights
and optimizer state, ordered prompts, per-request seeds, and policy versions.
The forward-only scoring API is not guaranteed invariant to packing.
Numerical determinism does not make asynchronous policy lag or changed training
order equivalent. It also does not establish reward quality or RL convergence.

The patch has attention forward/backward and packed-client validation. Run the
included GPU checks independently of online RL:

```bash
.venv/bin/python scripts/run_fa3_deterministic.py
.venv/bin/python scripts/run_fa3_deterministic.py --forward-backward
.venv/bin/python scripts/run_fa3_deterministic.py --upstream-tests
```

These commands build and run on Modal; results remain under `scripts/results/`.
No online RL runs, datasets, plotting scripts, or generated artifacts are included
in this change.
