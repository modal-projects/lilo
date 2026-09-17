# Four-client multi-LoRA codegolf experiment

This experiment lives on `codex/multilora-codegolf`, stacked on PR #15 at
`140e642`. It does not change or redeploy the merge-ready PR or the shared `lilo`
app. The launcher creates a separate Modal app, results volume, and rollout pool.

The runner reuses `examples/codeforces-codegolf/codegolf/train.py`, including
TailRL advantages, reward, prompt formatting, sandbox judging, evaluation,
checkpointing, and bounded asynchronous rollout buffers. Each client uses the
reference `tailrl` configuration: Qwen/Qwen3.5-9B (not Qwen3.5-9B-Base), 4 prompts
× 8 completions per update, 16,384 generated tokens, temperature 1, Adam 1e-6,
PPO clipping [0.8, 1.2], seed 42, checkpoint every 50 updates, and evaluation of
16 problems with 8 samples each every 20 updates. All four clients share the
same dataset/split; each has independent sampling, adapter, and optimizer state.

The trainer uses 8 H200s with TP8/CP1/DP1, four rank-32 adapters with alpha 32,
full activation recomputation, and a 65,536-token context/packing budget. This
is intentionally a different parallel layout from the FFT reference's
TP2/CP2/DP2: the current Miles integration requires DP1 and CP1. Multiple long
sequences run in multiple microbatches, not simultaneously in one giant batch.
The rollout pool has 4–8 separate one-H200 replicas with 64K context. These GPUs
are additional to the eight trainer GPUs.

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
No source experiment files are modified. SGLang reserves one token at the
context boundary: the optional inference probe uses 65,519 prompt tokens plus
16 generated tokens.

```bash
.venv/bin/python scripts/multilora_codegolf.py launch --run capacity-v1 --phase capacity --steps 2
.venv/bin/python scripts/multilora_codegolf.py status --run capacity-v1
# Only after full-context capacity and restore checks pass:
.venv/bin/python scripts/multilora_codegolf.py launch --run smoke-v1 --phase smoke --steps 3
.venv/bin/python scripts/multilora_codegolf.py status --run smoke-v1
# While smoke-v1 is running, check all four adapters at the inference boundary:
.venv/bin/python scripts/multilora_codegolf_context_probe.py --run smoke-v1
# Only after all four clients complete the short real RL run:
.venv/bin/python scripts/multilora_codegolf.py launch --run tailrl-hero-v1 --phase hero --steps 500
.venv/bin/python scripts/multilora_codegolf.py status --run tailrl-hero-v1
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
of updates is reduced. It fails after one recovery attempt so a persistent bug
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
The separate inference-boundary probe initially expected one token beyond
SGLang's reserved-token limit; the probe now accounts for that reservation.
