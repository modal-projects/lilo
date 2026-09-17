# Four-client multi-LoRA codegolf experiment

This experiment lives on `codex/multilora-codegolf`, based on PR #15 at
`140e642` with the subsequent LoRA reference-lifetime fixes carried over.
Experiment configuration stays separate from PR #15 and the shared `lilo` app.
The launcher creates a separate Modal app, results volume, and rollout pool.

The runner reuses `examples/codeforces-codegolf/codegolf/train.py`, including
TailRL advantages, reward, prompt formatting, sandbox judging, evaluation,
checkpointing, and bounded asynchronous rollout buffers. Each client uses the
reference `tailrl` configuration with LoRA-specific overrides: Qwen/Qwen3.5-9B (not Qwen3.5-9B-Base), 4 prompts
× 8 completions per update, 16,384 generated tokens, temperature 1, Adam 1e-5,
PPO clipping [0.8, 1.2], seed 42, checkpoint every 20 updates, and evaluation of
16 problems with 8 samples each every 20 updates. All four clients share the
same dataset/split; each has independent sampling, adapter, and optimizer state.
The FFT reference uses Adam 1e-6 and checkpoints every 50 updates. The LoRA
runner defaults to 1e-5 (override with `--learning-rate`) and checkpoints every
20 updates to limit recovery loss. Historical `tailrl-hero-v1` and
`tailrl-hero-v2` used 1e-6; `tailrl-hero-v3-lr1e5` starts fresh at 1e-5.

The trainer uses 8 H200s with TP8/CP1/DP1, four rank-32 adapters with alpha 32,
full activation recomputation, and a 65,536-token context/packing budget. This
is intentionally a different parallel layout from the FFT reference's
TP2/CP2/DP2: the current Miles integration requires DP1 and CP1. Multiple long
sequences run in multiple microbatches, not simultaneously in one giant batch.
The rollout pool has 4–8 separate one-H200 replicas with 64K context. These GPUs
are additional to the eight trainer GPUs. Each replica retains up to 64 adapter
versions and reloads evicted versions on demand.

The reference's PPO `weights` give each completion equal total loss weight.
Miles PPO does not consume a separate weights field. The experiment folds its
nonnegative sequence weight into the advantage, preserving
`w * min(r*A, clip(r)*A)` without adding any token/client normalization. Tests
compare losses and finite-difference gradients across signs and clip regimes.

## Run and monitor

From this checkout, configure Modal access, `MODAL_ENVIRONMENT=kailash-dev`, and
the existing `lilo-api` / `lilo-proxy` secrets. The launcher itself needs no
printed or persisted API key. Install the project plus pytest, numpy, and the
codegolf dependencies in a Python 3.12 environment.

The independent `lilo-multilora-codegolf` Volume must contain `problems.json`,
`reference-validated.json`, and `reference-split.json` copied from the original
FFT dataset. The dataset SHA-256 is
`0d8d24a6abfe30ab5144e396b32880aba0a76be093dab0fbe8153dd6dd952db5`.
No source experiment files are modified. SGLang reserves two tokens at the
context boundary: the optional inference probe uses 65,518 prompt tokens plus
16 generated tokens.

```bash
export MODAL_ENVIRONMENT=kailash-dev
# Keep the hero build identical to the validated smoke build.
export LILO_MILES_COMMIT=ef3807c0ef659d7c6d8494c4933bd7ee0332700f
.venv/bin/python scripts/multilora_codegolf.py launch --run capacity-v1 --phase capacity --steps 2
.venv/bin/python scripts/multilora_codegolf.py status --run capacity-v1
# Only after full-context capacity and restore checks pass:
.venv/bin/python scripts/multilora_codegolf.py launch --run smoke-v1 --phase smoke --steps 3
.venv/bin/python scripts/multilora_codegolf.py status --run smoke-v1
# While smoke-v1 is running, check all four adapters at the inference boundary:
.venv/bin/python scripts/multilora_codegolf_context_probe.py --run smoke-v1
# Only after all four clients complete the short real RL run:
.venv/bin/python scripts/multilora_codegolf.py launch --run tailrl-hero-v3-lr1e5 --phase hero --steps 500 --learning-rate 1e-5
.venv/bin/python scripts/multilora_codegolf.py status --run tailrl-hero-v3-lr1e5
```

Launch detaches a local supervisor and starts a remote Modal controller. The
supervisor records the app/call IDs, URL, Miles SHA, and logs under
`scripts/results/multilora-codegolf/<run>/`. Client metrics, rollout records,
checkpoints, and evaluations persist under `<run>/<run>-client-<0..3>/` on the
results volume. `status` downloads current summaries without raw rollouts.
The supervisor stops its own trainer/control app and rollout pool on completion
or failure. For manual termination, stop both app IDs listed in the supervisor
record; do not stop the shared `lilo` app.

Capacity checks submit full-64K sequences from all four resident clients, verify
finite outputs and actual parameter updates, and check model/optimizer
checkpoint restoration and inference publication for every slot. The short RL
run preserves full batch, generation, and evaluation budgets; only the number
of updates is reduced. It fails on the first trainer failure so a persistent bug
cannot be concealed by repeated restarts. The hero run uses the reference's
bounded checkpoint recovery loop. A few passing steps establish functional
operation, not long-run convergence or equivalence of LoRA and FFT rewards.

## Validation receipts

`capacity-v1` passed on Miles `ef3807c0ef659d7c6d8494c4933bd7ee0332700f`:
four clients on one engine/boot, two full-64K updates each, finite outputs,
nonzero gradients, and changed adapter outputs. All four native checkpoint
round trips had zero maximum logprob difference in both trainer and inference.
The second update took 6.6–14.9 seconds per client including queueing; the first
included approximately 174 seconds of warmup. GPU samples during the test were
about 34 GiB used per H200 (samples, not an exact peak-memory measurement).

[Capacity test app](https://modal.com/apps/modal-labs/kailash-dev/ap-v02kVD1OJNaLzaXimzDXQo).
Local receipts and logs remain in `scripts/results/multilora-codegolf/capacity-v1/`.
The separate inference-boundary probe initially expected tokens beyond
SGLang's two-token reservation (worker limit plus scheduler limit); the probe now accounts for that reservation.

`smoke-v1` completed three real TailRL updates per client (12 successful
optimizer updates, 32 rollouts each), with finite nonzero gradient norms.
Mean absolute trainer-versus-behavior logprob differences ranged from 0.0057
to 0.0156 across updates; these diagnostics include asynchronous policy lag.
All four adapters also passed the full-context inference boundary probe.
The initial evaluation pass rates were 8.6–17.2%. Historical single-turn FFT
runs started at 12.5% on smaller evaluations; this is a sanity check, not a
controlled reward comparison or evidence of convergence.

CPU validation: the backend suite passed 459 tests (2 skipped); the final
experiment and original codegolf tests passed 91 tests.

The smoke controller finished successfully, including all four final model +
optimizer checkpoints and all four step-3 evaluations (128 samples/client).
Final pass rates were 11.7–12.5%; three updates do not establish learning
quality. No trainer recovery was needed.

[Smoke test app](https://modal.com/apps/modal-labs/kailash-dev/ap-I6Y8BYvTdawfzUJUblCZuY).
