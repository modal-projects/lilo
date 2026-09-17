# Qwen3.8-27B + LoRA (r32) on LongRLVR with Miles on raw Modal — long-context baseline

Launcher: `scripts/e2e_longrlvr_qwen3_8_27b_miles_lora.py` (training-gym `TrainConfig` + `MilesRecipe`,
Miles image `radixark/miles:dev-202609151226`, model `Qwen/Qwen3.8-27B`, dataset `Guanzheng/LongRLVR-Data`).
W&B: `modal-labs/miles-lora-longcontext`, group `baseline-miles-27b`. Modal env `micah-dev`.

Fixed across rungs: GRPO group 8, 16 groups/step (128 samples), 1.5x oversampling, lr 1e-4 constant,
Adam (0.9, 0.95), wd 0, LoRA r32/a32/dropout 0 on qkv/o_proj/fc1/fc2 + GDN in_proj/out_proj, TIS on,
per-token loss, clip 0.2/0.28, KL off, temp 1.0, `enable_thinking=False`, recompute full/uniform/1,
`use_dynamic_batch_size`, max generation 4096, fully-async rollout (`max_weight_staleness=1`),
SGLang mem fraction 0.8, 32 max running requests. Only model, context length, prompt cap/floor and
topology change per rung. 5 training steps per rung (plus a 1-step prove run at 16k).

## Hardware: one pinned GPU type, H200

Requirement (Micah): a single pinned GPU type for the whole 27B benchmark on both Miles and Lilo,
preferring whichever has more available compute.

- Modal's strict form exists only for H100 (`H100!` = "do not auto-upgrade to H200"). `H200!` is rejected
  by the server (`InvalidError: H200! is not a valid GPU type`); plain `H200` is already an exact request.
- Scheduling probe (`scripts/probe_gpu_scheduling.py`, 2-node x 8-GPU clustered function, RDMA):
  `H200:8` landed in **453 s**; `H100!:8` had **not landed after ~20 min** (client gave up).
- Decision: **`H200` everywhere** (trainer + rollout; launcher default `--gpu-type H200`). Both earlier
  rungs, although launched with `--gpu-type H100`, were auto-upgraded by Modal to H200 (verified with
  `nvidia-smi`: `NVIDIA H200, 143771 MiB`), so all three completed rungs ran on identical hardware and
  nothing was redone.

## Per-rung table

| Rung | GPU (actual) | Nodes | Trainer TP×CP×DP | tok/GPU | Rollout | Peak trainer mem (nvidia-smi) | Step time steady | Train (fwd/bwd) | Log-probs | Rollout wait | tok/s/trainer-GPU | Reward per step | Resp len | Trunc | Prompt tokens (final step) | Result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16k | H200 141 GB | 2 (1 trainer + 1 rollout) | TP4×CP1×DP2 | 16384 | 8 eng × 1 GPU | ~31 GB | 517 s (step 0) → 149–172 s | 277 → 110–126 s | 169 → 37–45 s | 66/35/2.5/2.5/2.4 s | ~1.15–1.27k | 0.548, 0.477, 0.312, 0.574, 0.767 | 668/864/1078/803/550 | 0/0/0/0.039/0 | mean 11.2k (10.4k–12.2k), cap 12288 | OK, no OOM |
| 64k | H200 141 GB | 2 | TP4×CP2×DP1 | 32768 | 8 eng × 1 GPU | ~71 GB | 1550 s (step 0) → 1038 / 819 / 773 s | 1016 → 582 s | 392 → 189 s | 137/44/9/9/12 s | ~850 | 0.464, 0.427, 0.409, 0.508, 0.643 | 799/815/891/902/491 | 0 | mean 44.6k (32.2k–60.2k), cap 61440, floor 30720 | OK, no OOM |
| 128k | H200 141 GB | 2 | TP2×CP4×DP1 | 32768 | 8 eng × 1 GPU | 102–109 GB (step 0–1) growing to **~140.7 GB** (steps 2–4, allocator cache; no OOM) | 1452 s (step 0) → 994 / 972 / 888 s | 929 → 679 s | 359 → 207 s | 159/72/15/15/17 s | ~1.2k (67k tok × 128 / 888 s / 8) | 0.493, 0.339, 0.329, 0.543, 0.537 | 684/968/1079/876/449 | 0/0/0.016/0.016/0 | mean **66.8k** (63.7k–71.9k), cap 126976, floor 63488 | OK, no OOM; **prompts only ~52% of cap (dataset-limited)** |
| 256k | H200 | — | CP8 planned | 32768 | — | — | — | — | — | — | — | — | — | LongRLVR has **no** prompts ≥ 127k tokens (max ≈ 72k) | **Not run — needs decision** (see below) |

Notes on columns:
- "Rollout wait" is Miles `perf/rollout_time` in fully-async mode: time the trainer waited for the
  next batch, not generation wall time. Generation overlaps training after step 0.
- Step 4 `perf/step_time` is not flushed to W&B before the run exits (Miles logs perf after the next
  rollout); the last complete row (step 3) is used as the steady-state number for 64k and 128k.
- tok/s/trainer-GPU = 128 samples × mean(prompt+response) / step time / 8 trainer GPUs.
- Rollout GPUs show ~121–123 GB used on every rung: SGLang static preallocation (`mem_fraction 0.8`),
  not demand; KV cache for 128k × 32 requests fit on one H200 per engine.

## W&B runs

| Rung | Raw Miles run | `cmp/*` re-log (`miles-27b-<ctx>k-5`) |
|---|---|---|
| 16k prove (1 step) | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/sour-cilantro-1d2b0f829b5f | — |
| 16k | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/aquamarine-strut-8cb234449949 | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/vv9l76wt |
| 64k | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/violent-curb-decfe37c85e3 | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/5uvc8lks |
| 128k | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/fried-vault-4f0063789cf8 | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/nakynr4k |

The `cmp/*` runs (`scripts/relog_miles_cmp_27b.py`) carry `cmp/reward_mean, cmp/response_len_mean,
cmp/step_time_s, cmp/train_time_s, cmp/rollout_time_s, cmp/samples_per_s, cmp/tokens_per_gpu_per_s,
cmp/truncated_ratio` so they overlay the 9B runs. training-gym does not forward `exp_name` to Miles,
so the raw run titles are the group name.

## Exact launcher args per rung

All from `/home/ubuntu/repos/lilo` with `MODAL_SERVER_URL=https://api.modal.com MODAL_ENVIRONMENT=micah-dev`:

```bash
# 16k prove (1 step) and 5 steps — launched with --gpu-type H100 (then the default); Modal scheduled H200
uv run scripts/e2e_longrlvr_qwen3_8_27b_miles_lora.py --context-k 16 --steps 1
uv run scripts/e2e_longrlvr_qwen3_8_27b_miles_lora.py --context-k 16 --steps 5
#   -> Context 16384 (prompt cap 12288, floor 0); TP4xCP1xDP2, max_tokens_per_gpu=16384, 8 engines x TP1

# 64k — --gpu-type H100 default, Modal scheduled H200
uv run scripts/e2e_longrlvr_qwen3_8_27b_miles_lora.py --context-k 64 --steps 5 --cp 2
#   -> Context 65536 (prompt cap 61440, floor 30720 = 0.5x cap); TP4xCP2xDP1, max_tokens_per_gpu=32768

# 128k — default is now --gpu-type H200 (exact)
uv run scripts/e2e_longrlvr_qwen3_8_27b_miles_lora.py --context-k 128 --steps 5 --tp 2 --cp 4
#   -> Context 131072 (prompt cap 126976, floor 63488); TP2xCP4xDP1, max_tokens_per_gpu=32768
```

## What each rung required over the previous one

- **16k → 64k:** context parallel 2 (TP4×CP2, DP 2→1). `max_tokens_per_gpu` stays 32768 = ctx/CP.
  A prompt floor of 0.5×cap was needed to get prompts that actually use the budget (unfiltered
  LongRLVR is mostly 12k–40k tokens). Trainer memory 31 → 71 GB/GPU; steady step time 150 → 770 s
  (throughput per trainer GPU roughly ⅔ of 16k because DP dropped to 1 and CP adds all-to-all traffic).
- **64k → 128k:** CP4 with TP2 (TP2×CP4) to keep `max_tokens_per_gpu` at 32768 — TP4×CP2 would have
  doubled the per-GPU token budget to 65536 and the 64k memory profile (71 GB at 32768 tok/GPU) said
  that would not fit. Trainer memory 102–109 GB in the first steps and ~140 GB (of 143.7 GB) from
  step 2 on — it fits on H200 but there is essentially no headroom; on H100 (80 GB) this topology
  would OOM. Steady step time 770 → 890 s; per-GPU token throughput actually recovered to ~1.2k tok/s
  because the sequences are ~1.5× longer at only ~15% higher step time. Rollout still fits 1 H200 per
  SGLang engine at 128k context (no TP2 engines needed).
- **128k → 256k (not run):** the plan was CP8 (TP1×CP8, DP1) at 32768 tok/GPU. Two blockers surfaced:
  1. **Dataset:** LongRLVR documents top out at ~72k tokens. The 128k final-step prompts were 63.7k–71.9k
     (mean 66.8k, i.e. ~52% of the 127k cap) and a 3,001-row streaming scan of the dataset found 0 rows
     ≥ ~127k tokens (max 264k chars ≈ 72k tokens; 171 rows ≥ 63k tokens). A 256k rung would train on the
     same ~67k-token sequences as 128k, i.e. it would measure nothing new about 256k unless prompts are
     synthetically extended (concatenating documents) — a methodology decision, not mine to make.
  2. **Memory / nodes:** TP1×CP8 puts full 27B bf16 weights (~54 GB) on every GPU instead of ~27 GB at
     TP2, on top of the ~140 GB nvidia-smi peak already observed at TP2×CP4 with the same 32768 tok/GPU.
     Fitting 256k realistically means a second trainer node (TP2×CP8 over 16 GPUs) → 3 nodes total,
     which exceeds the 2-node envelope that requires a human decision.

## Unblocks / deviations made

- **TorchInductor compile hang (16k prove, attempt 1):** rank 0 stuck in an Inductor subprocess compile
  while the other TP ranks waited in NCCL reduce-scatter → 600 s watchdog timeout. Fixed at config
  level with `TORCHINDUCTOR_COMPILE_THREADS=1` (+ `PYTHONFAULTHANDLER=1`) in the recipe environment;
  no Miles source change. Retry was clean.
- **W&B entity:** `entity="modal-labs"` added to `WandbConfig` (first dry run had an empty entity).
- **GPU type:** default switched from `H100` to exact `H200` (see Hardware section). No redo of 16k/64k
  since they had already landed on H200.
- **Prompt floor:** `min_prompt_fraction` defaults to 0.5 for ctx > 16k (0 at 16k).
- **Step 4 perf metrics:** not flushed by Miles before exit; steady-state numbers use step 3.
- **Dataset-materialization log line** for 128k (prompt-length distribution over the 256 prepared
  prompts) was not captured (client timed out on app logs); the final-step trace (128 samples) is
  reported instead.

## Failures / anomalies

- No OOMs, no NCCL timeouts after the Inductor mitigation, no truncation above 4%.
- 128k trainer nvidia-smi memory grew from ~107 GB (step 0–1) to ~140.7 GB (step 2+) — consistent with
  the CUDA caching allocator expanding on longer-response batches (Megatron reports 26.9 GB allocated
  / 27.8 GB reserved idle between steps). It did not OOM, but 256k or larger response budgets at this
  topology should be expected to.
- `H100!:8` two-node clustered probe did not schedule within ~20 min in `micah-dev` at the time of the
  test (H200 did in 7.5 min).

## Artifacts

- Logs: `logs/16k_*`, `logs/64k_*`, `logs/128k_*` (launcher logs, app logs, W&B history JSON,
  `nvidia-smi` samples), `logs/probe_h200.log`, `logs/probe_h100strict.log`.
- Final-step traces (128 samples each): `traces/sour-cilantro-1d2b0f829b5f/step_0000.json`,
  `traces/aquamarine-strut-8cb234449949/step_0004.json`, `traces/violent-curb-decfe37c85e3/step_0004.json`,
  `traces/fried-vault-4f0063789cf8/step_0004.json`.
- Modal apps: 64k `ap-d6HDmE9FodS3Pp5XEOOxNF`, 128k `ap-MbINJXtPgXTCavSPyxVWlB` (all stopped).
