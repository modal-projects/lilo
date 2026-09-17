# Lilo 27B Long-Context Rungs (Qwen3.8-27B LoRA r32 on LongRLVR)

## Summary

We ran the Lilo Miles-LoRA backend (`lilo-27b` Modal app, env `micah-dev`)
through three context-length rungs — 16k, 64k, and 128k — training
Qwen3.8-27B LoRA rank-32 for 5 steps each on the LongRLVR recipe. All three
rungs completed cleanly on 8×H200 trainer nodes with TP/CP/DP topologies
scaled by rung. A fourth rung at 256k **failed**: Miles' Tinker loss path
OOMs in the fp32 logits-chunk upcast (`math_utils._upcast_chunk_to_fp32`)
on both TP1×CP8 and TP2×CP4 topologies — the ~29 GiB fp32 chunk cannot fit
next to the ~105–107 GiB already allocated on a 139.8 GiB H200.

## GPU types actually used

- Trainer requests were `gpu="H100:8"` for the 16k and 64k rungs, but Modal
  **auto-upgrades `H100` requests to H200**. Verified by running `nvidia-smi`
  inside live containers (`modal container exec`):
  - 64k trainer `ta-01M2PPWKD4KWBT6HEA7QPE19QR`: `NVIDIA H200` ×8.
  - 128k trainer `ta-01M2PVSPSQMV2GX35SHBKD56ZR`: `NVIDIA H200` ×8.
  - 256k trainer `ta-01M2Q2DFTHR9S0AEQQBH5M2XJR`: `NVIDIA H200` ×8.
  - **16k trainer: device not directly verifiable** (container gone, no
    device-name line in logs); it was the same `gpu="H100:8"` request in the
    same window, so it was almost certainly also H200 — unverified.
- Rollout pools verified by nvidia-smi: 64k = 8×H200 TP1
  (`ta-01M2PQCZMVRYVF7ZM12P4PW77R`), 128k = 4×H200 TP2
  (`ta-01M2PVT4K22G2T81T78HMD8Q5R`), 256k = 2×H200 TP4
  (`ta-01M2Q2DJFVE6GAV7FQC7N5VAHR`).
- Strict pinning: `gpu="H200!"` is rejected by the server
  (`InvalidError: H200! is not a valid GPU type`) — the `!` suffix exists
  only for `H100` (to prevent the auto-upgrade). `gpu="H200"` is already
  the un-upgradable top tier, so **H200 is now the pinned GPU for trainer
  and rollout** (`GPU_TYPE = "H200"` in all three 27B definitions).
- Probe (`lilo-27b-gpu-probe`, 8-GPU `nvidia-smi -L` calls):
  `H100!:8` scheduled in 3.0 s on a real `NVIDIA H100 80GB HBM3`;
  `H200:8` scheduled in ~3–4 min on a real `NVIDIA H200`.

## Per-rung results

Peak memory is `lilo_memory` `max_allocated_gb` / `max_reserved_gb` from the
trainer actor (uniform across all 8 ranks). Step metrics are client-side
`cmp/*` values; steady-state shown below, full per-step tables follow.

### 16k — `lilo-27b-16k-5` (W&B `uwcvqxpb`)

https://wandb.ai/modal-labs/miles-lora-longcontext/runs/uwcvqxpb

Trainer topology **TP4×CP1×DP2**; rollout H200 TP1 ×8 containers.
Peak memory: not captured (instrumentation landed after this rung's trainer
was already up). No OOM, no errors. ~22.5 min for 5 steps.

| step | step_time_s | train_time_s | rollout_time_s | samples_per_s | tok/GPU/s | reward_mean | resp_len_mean | trunc_ratio |
|---|---|---|---|---|---|---|---|---|
| 0 | 311.4 | 300.8 | 234.8 | 0.411 | 621.9 | 0.389 | 837.0 | 0.0 |
| 1 | 135.5 | 123.6 | 173.3 | 0.944 | 1389.0 | 0.418 | 962.3 | 0.0 |
| 2 | 177.0 | 165.9 | 111.4 | 0.723 | 1160.9 | 0.528 | 1471.9 | 0.047 |
| 3 | 140.5 | 126.2 | 78.5 | 0.911 | 1385.3 | 0.578 | 666.2 | 0.0 |
| 4 | 135.2 | 125.3 | 68.0 | 0.947 | 1428.2 | 0.713 | 696.7 | 0.0 |

prompt_len_mean ≈ 11.4k. reward_mean 0.389 → 0.713.

### 64k — `lilo-27b-64k-5` (W&B `h2yt8e23`)

https://wandb.ai/modal-labs/miles-lora-longcontext/runs/h2yt8e23

Trainer topology **TP4×CP2×DP1**, `seq_length 65536`;
rollout H200 TP1 ×8 containers (max_running 16, concurrency 8).
Peak trainer memory: `max_allocated_gb ≈ 41.57`, `max_reserved_gb ≈ 48.65`
(fb and optim identical). 262 transient "request queue is full" 503
reroutes in a ~1-min rollout-saturation burst (~04:07 UTC); retried
transparently, no failures. ~72 min for 5 steps.

| step | step_time_s | train_time_s | rollout_time_s | samples_per_s | tok/GPU/s | reward_mean | resp_len_mean | trunc_ratio |
|---|---|---|---|---|---|---|---|---|
| 0 | 1333.3 | 1316.5 | 151.6 | 0.096 | 489.9 | 0.434 | 699.6 | 0.0 |
| 1 | 598.6 | 587.4 | 161.8 | 0.214 | 1065.9 | 0.614 | 953.8 | 0.031 |
| 2 | 815.3 | 803.0 | 167.1 | 0.157 | 794.6 | 0.296 | 1148.0 | 0.008 |
| 3 | 546.3 | 531.0 | 122.3 | 0.234 | 1152.6 | 0.666 | 545.0 | 0.0 |
| 4 | 501.3 | 482.4 | 124.6 | 0.255 | 1148.7 | 0.660 | 540.3 | 0.0 |

prompt_len_mean ≈ 35–40k. reward_mean 0.434 → 0.660.

### 128k — `lilo-27b-128k-5` (W&B `t9feh1se`)

https://wandb.ai/modal-labs/miles-lora-longcontext/runs/t9feh1se

Trainer topology **TP2×CP4×DP1**, `seq_length 131072`;
rollout H200 TP2 ×4 containers (max_running 8, concurrency 4).
Peak trainer memory: `max_allocated_gb ≈ 80.79`, `max_reserved_gb ≈ 96.07`.
776 transient 503 reroutes over the run (4-container rollout pool is the
bottleneck); no failures. ~58 min for 5 steps.

| step | step_time_s | train_time_s | rollout_time_s | samples_per_s | tok/GPU/s | reward_mean | resp_len_mean | trunc_ratio |
|---|---|---|---|---|---|---|---|---|
| 0 | 742.6 | 713.5 | 172.4 | 0.172 | 932.1 | 0.476 | 807.7 | 0.0 |
| 1 | 603.1 | 584.0 | 205.0 | 0.212 | 1035.8 | 0.419 | 1062.4 | 0.039 |
| 2 | 677.0 | 648.9 | 290.8 | 0.189 | 1056.9 | 0.450 | 1154.1 | 0.047 |
| 3 | 465.3 | 444.6 | 147.2 | 0.275 | 1397.2 | 0.676 | 546.4 | 0.0 |
| 4 | 472.8 | 451.0 | 138.7 | 0.271 | 1399.5 | 0.658 | 562.2 | 0.0 |

prompt_len_mean ≈ 38–44k. reward_mean 0.476 → 0.658.

### 256k — smoke only, OOM (no run)

Definition attempted: **TP1×CP8×DP1**, `max_tokens_per_gpu=32768`,
`seq_length 262144`; fallback: **TP2×CP4, `max_tokens_per_gpu=65536`**.
Both OOMed in `forward_backward` on a 252k-token datum:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 29.14 GiB.
GPU 7 has a total capacity of 139.80 GiB of which 23.13 GiB is free.
Process 1 has 116.67 GiB memory in use. Of the allocated memory 105.68 GiB
is allocated by PyTorch
```

thrown from `miles/backends/training_utils/loss_hub/math_utils.py:955`
`_upcast_chunk_to_fp32` (attempt 1) and a torchinductor graph doing the same
fp32 logits allocation `empty_strided_cuda((s10, 1, 124160), ..., torch.float32)`
(attempt 2). The fp32 upcast chunk scales with vocab width × microbatch
rows, not the CP shard length alone; neither topology leaves ~29 GiB free.
Options to revisit: vocab-sharded/chunked fp32 upcast in `math_utils`,
smaller `max_tokens_per_gpu` with more microbatches, or TP2×CP8 on >8 GPUs.

## What each rung required

- **16k**: `LILO_APP_NAME` forwarded into the trainer container env
  (`deployment.py`); `qwen3_8_27b_miles_lora_16k.py` definition (TP4×DP2).
- **64k**: CP support end-to-end —
  `MilesBackendConfig.context_parallel_size` (+ `dp = world // (tp·cp)`,
  `--context-parallel-size` arg, topology metadata); two Tinker-path
  patches in `miles_runtime/actor.py`: (1) `_gather_tinker_logprobs_across_cp`
  — gather each rank's local zigzag logprob shard to full response via
  zero-pad + non-differentiable `all_reduce`, so gradients flow only into
  the rank's own shard (Miles scales loss by intra_dp_cp.size), repointed at
  every `get_log_probs_and_entropy` import site
  (`logit_processors`, `loss`, `tinker_losses`, `megatron_actor`,
  `megatron_model`, `fsdp_actor`); (2) neutralize `slice_log_prob_with_cp`
  — Miles also CP-slices `rollout_log_probs`/`teacher_log_probs` at data
  load (`training_utils/data.py`), which mismatches the now-gathered lp —
  repointed on `cp_utils`, `data`, `mm_data`, `math_utils`. Client script
  gained `--context-length` and `--engine-model`.
- **128k**: definition only — TP2×CP4 (halves per-GPU tokens vs 64k),
  rollout bumped to TP2×4 containers.
- **256k**: definition only; OOM as above.

## Exact definitions and commands

All trainer defs share: `GPUS=8`, `MODEL_NAME="Qwen/Qwen3.8-27B"`,
`model_type qwen3.8-27B`, LoRA r32/α32, `MAX_LORA_SLOTS=6`,
`--recompute-granularity full --recompute-method uniform --recompute-num-layers 1`,
`GPU_TYPE="H200"` (post-pin; was `"H100"` for rungs 1–2), rollout
`ROLLOUT_GPU_TYPE="H200"`, `CATALOG_VISIBLE` True for 16k / False for the rest.

| rung | MAX_CONTEXT_LENGTH | TP | CP | DP | max_tokens_per_gpu | rollout TP / containers / running / concurrency |
|---|---|---|---|---|---|---|
| 16k | 16,384 | 4 | 1 | 2 | 16,384 | 1 / 8 / 32 / 16 (16k def values) |
| 64k | 65,536 | 4 | 2 | 1 | 32,768 | 1 / 8 / 16 / 8 |
| 128k | 131,072 | 2 | 4 | 1 | 32,768 | 2 / 4 / 8 / 4 |
| 256k | 262,144 | 1 | 8 | 1 | 32,768 | 4 / 2 / 4 / 2 |

Deploy (all rungs, in-place on `lilo-27b`):

```
LILO_APP_NAME=lilo-27b MODAL_ENVIRONMENT=micah-dev \
MODAL_SERVER_URL=https://api.modal.com \
uv run modal deploy -m lilo.providers.modal.app
```

Training client (detached spawn via `run_client_modal.py` wrapper →
`scripts/run_longrlvr_lilo_lora.py`; 64k example):

```
python scripts/run_longrlvr_lilo_lora.py --steps 5 \
  --base-url https://modal-labs-micah-dev--lilo-27b-server.us-west.modal.run \
  --base-model Qwen/Qwen3.8-27B \
  --engine-model qwen3_8_27b_miles_lora_64k \
  --wandb-group lilo-27b --run-name lilo-27b-64k-5 \
  --trainer-gpus 8 --max-steps-off-policy 1 \
  --context-length 65536 --max-generation-tokens 4096 \
  --std-normalize-advantages --per-token-loss-scale \
  --log-path /root/lilo-27b-64k-5_metrics.json
```

Same shape for 128k (`--context-length 131072`,
`--engine-model qwen3_8_27b_miles_lora_128k`); 16k used
`--context-length 16384` default and no `--engine-model` (16k def is
catalog-visible by model name). Hidden definitions are addressed by
`base_model=<DEFINITION_ID>` — `definition_for()` in
`control_plane/http.py` matches `DEFINITION_ID` before the
catalog-visible `MODEL_NAME` lookup.

## Dataset caveat

LongRLVR prompts do not scale to the rung cap: `prompt_len_mean` was
~11.4k at 16k and ~35–44k at both 64k and 128k. The 128k rung therefore
exercised ~1/3 of its context budget on real data; only the smoke-test
datums (58k / 122k) verified the full CP path at scale.

## Unblocks log

- `CATALOG_VISIBLE=False` on 64k/128k/256k: the registry test requires
  (model_name, parameterization) uniqueness among visible defs; the 16k def
  already occupies `(Qwen/Qwen3.8-27B, lora)`. Still addressable by
  `DEFINITION_ID` as `base_model` (see above).
- `lilo_memory` lines initially absent — actor `logger.info` suppressed for
  the `lilo` module; switched to `print(flush=True)` (commit `164d261`).
- First 64k smoke `training.forward` failed
  `size of tensor a (13) vs b (5)` — the lead's repoint-list trim missed
  `megatron_utils/actor.py` (binds `get_log_probs_and_entropy` at import
  for `forward_only`); restored full repoint set (`2c5e3c1`).
- 64k train `forward_backward` failed `size of tensor a (62018) vs b
  (31008)` — lp was full-length but `rollout_log_probs` was CP-sliced at
  data load; fixed by the `slice_log_prob_with_cp` neutralization
  (`3c5e23d`).
- `--base-model` doubles as the HF tokenizer name, so the hidden def id
  crashed HF lookup; added `--engine-model` (`153b370`).
- Rollout-pool saturation produces "request queue is full" 503s that are
  retried by the reroute layer — transient, not fatal (262 at 64k, 776 at
  128k).
- Modal operational quirks: `modal run` fails "Token not found" on files
  under `/home/ubuntu/work/` (copy to `/tmp`); `modal app logs` /
  `FunctionCall.get` intermittently `AuthError: Token not found` — use the
  SDK `_App.logs.fetch()` path; spawning via `app.run(detach=True)` +
  `fn.spawn()` survives client disconnects.
- Stopping the trainer container alone does not free its GPUs — the engine
  health-check respawns an 8×H200 trainer within minutes. Freeing it
  requires stopping the `lilo-27b` app (redeploy takes ~5 s).
