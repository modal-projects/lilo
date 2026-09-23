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
| 256k (padded) — 1-node trainer probe | H200 141 GB | 2 (1 trainer + 1 rollout) | TP1×CP8×DP1 | 32768 | 8 eng × 1 GPU | **OOM** (111 GB in use, +30 GiB alloc failed) | — | — | — | — | — | — | — | — | padded to ~250k | **OOM in compute_log_prob** (TP1 = 54 GB weights/GPU) |
| 256k (padded) | H200 141 GB | **3** (2 trainer + 1 rollout) | TP2×CP8×DP1 (16 trainer GPUs) | 32768 | 8 eng × 1 GPU | ~98–103 GB (nvidia-smi after backward); 27 GB allocated / 88 GB reserved idle | 3889 s (step 0) → 2938 / 2817 / 2863 s (**~48 min**) | 2546 → 2200–2300 s | 826 → 608–631 s | 506/283/68/67/66 s | ~0.7k (250k tok × 128 / 2860 s / 16) | 0.403, 0.461, 0.311, 0.558, 0.479 | 802/784/989/935/503 | 0/0/0.016/0.008/0 | mean **248k** (232k–258k), cap 258048, floor 232243, `pad_to` 250k — **synthetic** (distractor-padded) | OK, no OOM |
| 128k (padded), **30 steps** (`miles-27b-128k-pad-30`) | H200 141 GB | 2 | TP2×CP4×DP1 | 32768 | 8 eng × 1 GPU | ~140 GB (allocator cache, as natural 128k); no OOM | 2610 s (step 0) → **1546–1818 s, mean 1666 s (27.8 min)** over steps 1–28 | 1774 → 1194–1405 s (mean ~1290 s) | 609 → 349–411 s | 219/109 s then 25–29 s | ~1.1k (113.5k tok × 128 / 1666 s / 8) | 30 steps, mean **0.744**; steps 0–4 0.477 → steps 25–29 **0.874**; steps 10–29 0.818 ± 0.118 sd (n=20); max 1.136 (step 27) — full series in §Padded 30-step runs | 863 → 226–549 (late mean ~290) | 0 except 0.008 at steps 5, 17 | mean **113.5k** (84.7k–127.0k), cap 126976, floor 0, `pad_to` 120k — **synthetic** | OK, completed 30/30, final adapter saved |
| 256k (padded), **30 steps** (`miles-27b-256k-pad-30`) | H200 141 GB | 3 | TP2×CP8×DP1 | 32768 | 8 eng × 1 GPU | as 5-step run | 3718 s (step 0) → ~2690–2800 s (~45.5 min) | — | — | 63–150 s | ~0.7k | in flight (step 18/30 at 18:50 UTC 2026-09-17); rewards 0–12: 0.440, 0.394, 0.460, 0.518, 0.543, 0.534, 0.758, 0.795, 0.643, 0.778, …, 0.899 | 899 → ~265 | 0 | mean **247k** (232k–258k), `pad_to` 250k — **synthetic** | pending |

Notes on columns:
- "Rollout wait" is Miles `perf/rollout_time` in fully-async mode: time the trainer waited for the
  next batch, not generation wall time. Generation overlaps training after step 0.
- Step 4 `perf/step_time` is not flushed to W&B before the run exits (Miles logs perf after the next
  rollout); the last complete row (step 3) is used as the steady-state number for 64k and 128k.
- tok/s/trainer-GPU = 128 samples × mean(prompt+response) / step time / trainer GPUs (8, or 16 for
  the 3-node 256k topology).
- Rollout GPUs show ~121–123 GB used on every rung: SGLang static preallocation (`mem_fraction 0.8`),
  not demand; KV cache for 128k × 32 requests fit on one H200 per engine.

## W&B runs

| Rung | Raw Miles run | `cmp/*` re-log (`miles-27b-<ctx>k-<steps>`) |
|---|---|---|
| 16k prove (1 step) | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/sour-cilantro-1d2b0f829b5f | — |
| 16k | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/aquamarine-strut-8cb234449949 | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/vv9l76wt |
| 64k | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/violent-curb-decfe37c85e3 | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/5uvc8lks |
| 128k | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/fried-vault-4f0063789cf8 | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/nakynr4k |
| 256k padded, 2-node TP1×CP8 probe (OOM) | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/matte-degree-093dd63e9914 | — |
| 256k padded, 3-node TP2×CP8 probe (1 step) | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/molten-area-e262106588e6 | — |
| 256k padded, 3-node TP2×CP8 (5 steps) | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/merciless-barracuda-46b27a74bc00 | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/7jiwji5w (tok/s normalised to 16 trainer GPUs) |
| 128k padded, 30 steps (`miles-27b-128k-pad-30`, group `miles-27b`) | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/mild-bleed-b1d6255af892 | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/xw5cx2je |
| 256k padded, 30 steps (`miles-27b-256k-pad-30`, group `miles-27b`) | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/cool-muntin-b1d6255af892-a2 | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/yz1wuem8 |

The `cmp/*` runs (`scripts/relog_miles_cmp_27b.py`) carry `cmp/reward_mean, cmp/response_len_mean,
cmp/step_time_s, cmp/train_time_s, cmp/rollout_time_s, cmp/samples_per_s, cmp/tokens_per_gpu_per_s,
cmp/truncated_ratio, cmp/prompt_len_mean` so they overlay the 9B runs. training-gym does not forward `exp_name` to Miles,
so the raw run titles are the group name.

### `cmp/train_time_s` semantics (v2 re-logs)

The re-logs above set `cmp/train_time_s = perf/actor_train_time` (fwd/bwd + optimizer only). Miles'
train stage also runs a forward-only pass to recompute old-policy log-probs
(`compute_log_prob`, timed as `perf/log_probs_time`) because the baseline used
`use_rollout_logprobs=False` + TIS; that pass is ~27–30% of fwd/bwd at every rung and
O(tokens). Under the old mapping it showed up as `cmp/step_time_s − cmp/train_time_s`
(the "non-train" residual: 42 / 203 / 375 / 595 s at 16k / 64k / 128k / 256k), which reads as
idle/overlap time next to Lilo's `time/total − time/train_step`. It is trainer GPU compute on the
critical path; the real idle time is `perf/train_wait_time` (1 / 2 / 3 / 5 s). Lilo's `ppo` loss
consumes the sampler log-probs directly and has no equivalent pass.

The `-v2` runs re-log the same sources with `cmp/train_time_s = perf/train_time` (the whole train
stage, comparable to Lilo's `time/train_step`) and add `cmp/fwd_bwd_time_s`, `cmp/logprob_time_s`,
`cmp/wait_time_s`, `cmp/non_train_time_s`:

| Rung | Raw Miles run | `cmp/*` v2 re-log | step | train (`perf/train_time`) | fwd/bwd | log-probs | wait / non-train |
|---|---|---|---|---|---|---|---|
| 16k | `aquamarine-strut-8cb234449949` | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/0uvx0qho (`miles-27b-16k-5-v2`) | 150–172 s | 149–171 s | 111–126 s | 38–45 s | ~1 s |
| 64k | `violent-curb-decfe37c85e3` | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/avmnf2uu (`miles-27b-64k-5-v2`) | 819 s | 818 s | 617 s | 201 s | ~2 s |
| 128k padded, 30 steps | `mild-bleed-b1d6255af892` | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/pzoqlw9d (`miles-27b-128k-pad-30-v2`) | 1655 s | 1652 s | 1280 s | 373 s | ~3 s |
| 256k padded, 30 steps | `cool-muntin-b1d6255af892-a2` | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/mwjyfa17 (`miles-27b-256k-pad-30-v2`) | 2754 s | 2749 s | 2159 s | 590 s | ~5 s |

Steady-state medians over steps ≥ 1. Groups match the originals (`baseline-miles-27b` for 16k/64k,
`miles-27b` for the padded 30-step runs).

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

# 256k, padded prompts. 2-node probe (OOM), then 3 nodes (2 trainer + 1 rollout)
uv run scripts/e2e_longrlvr_qwen3_8_27b_miles_lora.py --context-k 256 --steps 1 --tp 1 --cp 8 --pad-to-tokens 250000 --run-suffix=-probe-2node
uv run scripts/e2e_longrlvr_qwen3_8_27b_miles_lora.py --context-k 256 --steps 1 --tp 2 --cp 8 --actor-nodes 2 --pad-to-tokens 250000 --run-suffix=-probe-3node
uv run scripts/e2e_longrlvr_qwen3_8_27b_miles_lora.py --context-k 256 --steps 5 --tp 2 --cp 8 --actor-nodes 2 --pad-to-tokens 250000
#   -> Context 262144 (prompt cap 258048, floor 232243 = 0.9x cap, pad_to 250000); TP2xCP8xDP1, max_tokens_per_gpu=32768

# 30-step statistical runs (W&B group miles-27b), padded, launched in parallel
uv run scripts/e2e_longrlvr_qwen3_8_27b_miles_lora.py --context-k 128 --steps 30 --tp 2 --cp 4 --actor-nodes 2 \
    --pad-to-tokens 120000 --min-prompt-fraction 0 --wandb-group miles-27b --run-name miles-27b-128k-pad-30
uv run scripts/e2e_longrlvr_qwen3_8_27b_miles_lora.py --context-k 256 --steps 30 --tp 2 --cp 8 --actor-nodes 2 \
    --pad-to-tokens 250000 --wandb-group miles-27b --run-name miles-27b-256k-pad-30
```

## Prompt padding scheme (`--pad-to-tokens`)

LongRLVR documents top out at ~72k tokens (3,001-row scan: 0 rows ≥ 127k), so the 256k rung — and the
"padded 128k" comparison run — use **synthetic** prompts: each row keeps its real document, question,
ground-truth answer and citation chunks, and other LongRLVR rows' documents are added as distractors.
The numbers measure the systems cost of a ~250k-token sequence and the model's ability to answer under
distraction, **not** comprehension of a genuinely 250k-token document.

Algorithm (deterministic; two independent 256k preparations produced byte-identical length stats):

1. Distractor pool = first 64 parseable documents of a second, seed-0 shuffled LongRLVR stream.
2. Per row: `rng = random.Random(f"{seed}:{question}")` (seed 0). Render the base chat prompt with the
   Qwen3.8-27B tokenizer (`enable_thinking=False`), measure tokens, compute chars/token, and set
   `target_chars = pad_to_tokens × ratio`.
3. Shuffle the pool with `rng`, append distractor documents in that order until `target_chars` is
   reached; insert the real document at an `rng`-chosen position. The real document's chunk order is
   preserved; all `<CHUNK_n>` IDs are renumbered globally and the row's `ref_chunks` remapped so the
   chunk-citation F1 reward is unchanged.
4. Re-render and tokenize; if over the prompt cap, trim distractor chunks from the ends and repeat.
   Rows still over the cap (or, at 256k, under the 0.9×cap floor) are dropped (`pad_failed` / `too_short`).
5. The label carries `prompt_tokens` (actual rendered length) and `base_prompt_tokens`, so real prompt
   lengths are available per sample; W&B `rollout/total_lengths` − `rollout/response_lengths` gives the
   per-step prompt mean (`cmp/prompt_len_mean`).

Achieved distributions (materialization log lines):

| Run | prompts | scanned | too_short | pad_failed | min | Q1 / median / Q3 | max | mean | base-doc mean |
|---|---|---|---|---|---|---|---|---|---|
| 256k, 5 steps (`pad_to` 250k, cap 258048, floor 232243) | 120 | 551 | 168 | 262 | 232,358 | 241,336 / 249,432 / 254,932 | 258,035 | 248,057 | 39,871 |
| 128k-pad-30 (`pad_to` 120k, cap 126976, floor 0) | 720 | 1,148 | 0 | 428 | 84,730 | 105,433 / 114,910 / 123,282 | 126,973 | 113,467 | 39,610 |
| 256k-pad-30 (`pad_to` 250k, cap 258048, floor 232243) | 720 | 3,287 | 964 | 1,592 | 232,285 | 240,236 / 247,757 / 255,007 | 258,044 | 247,249 | 37,885 |

`pad_failed` is high because the char→token estimate overshoots and the trim loop gives up once the
prompt is still over the cap; enough rows survive. Materializing 720 padded 256k prompts took ~2.5 h
of single-process tokenization on the cluster head (GPUs idle) — a known cost of the current
implementation.

The padded 128k prompt set is shared for the Lilo lane on Modal volume
`miles-longrlvr-qwen3-27b-lora-128k-data` (env `micah-dev`), file
`2ebeaefb-f731-43da-927c-86c257a25ebe-bd73959668bda69b.parquet` (columns `prompt` = padded chat
messages, `label` = JSON with question, ground_truth, renumbered ref_chunks, prompt_tokens,
base_prompt_tokens).

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
- **128k → 256k (padded prompts):**
  1. **Dataset:** LongRLVR documents top out at ~72k tokens (128k final-step prompts were 63.7k–71.9k,
     ~52% of the cap; a 3,001-row scan found 0 rows ≥ 127k). 256k therefore uses distractor-padded
     prompts (see "Prompt padding scheme"); the 256k row is a systems number on synthetic sequences.
  2. **Memory / nodes:** the 2-node plan TP1×CP8×DP1 (one 8-GPU trainer node) **OOMed** in
     `compute_log_prob` at step 0: 111 GB in use on GPU 7 (full 27B bf16 weights ≈ 54 GB at TP1 plus
     activations), failed to allocate a further 30 GiB. The working topology is **TP2×CP8×DP1 over two
     trainer nodes** (16 GPUs, 27 GB weights/GPU) + one rollout node = **3 nodes**, exceeding the original
     2-node envelope (approved by Micah for this rung). At that topology trainer memory is ~98–103 GB/GPU,
     with headroom, and steady step time is ~48 min (log-probs ~610 s, actor fwd/bwd ~2200–2300 s);
     per-trainer-GPU throughput drops to ~0.7k tok/s (16 GPUs share one 128-sample batch).
     Rollout still fits one H200 per SGLang engine at 256k context (prefix-cache hit ~87%).

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

- Logs: `logs/16k_*`, `logs/64k_*`, `logs/128k_*`, `logs/256k_*` (launcher logs, app logs, W&B history
  JSON, `nvidia-smi` samples), `logs/probe_h200.log`, `logs/probe_h100strict.log`.
- Final-step traces (128 samples each): `traces/sour-cilantro-1d2b0f829b5f/step_0000.json`,
  `traces/aquamarine-strut-8cb234449949/step_0004.json`, `traces/violent-curb-decfe37c85e3/step_0004.json`,
  `traces/fried-vault-4f0063789cf8/step_0004.json`.
- Modal apps: 64k `ap-d6HDmE9FodS3Pp5XEOOxNF`, 128k `ap-MbINJXtPgXTCavSPyxVWlB`, 256k 5-step
  `ap-DUrSuzBftUsmHVuhkILM00` (stopped); 128k-pad-30 `ap-wRXILTBrv9i1OLMtzNulyQ`, 256k-pad-30
  `ap-hPgnIN4adZYyZjgG2tA4K7` (running at time of writing).
