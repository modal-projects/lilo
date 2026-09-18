# Miles vs Lilo — step-0 LoRA gradient comparison on an identical batch

Qwen/Qwen3.5-9B, LoRA r32/α32, LR 1e-4, Adam(0.9, 0.95, eps 1e-8), wd 0, 16k
context, TP4 × DP2 on 8×H100. Motivation: on the same pinned 360-prompt
LongRLVR file, Miles (`slim-hill-371ae1cfcb8e`) reaches reward ~0.46 and Lilo
(`czzwpqau`) ~0.67, and the step-14 Miles adapter re-scores weaker on a neutral
sampler. Every rollout-side surface had been ruled out; this experiment tests
the trainer-side LoRA update directly: one `forward_backward` + one Adam step
in each trainer on byte-identical serialized training data.

W&B: project `modal-labs/miles-lora-longcontext`, group `graddiff-step0`.
Branch: `devin/1789691530-graddiff-step0` (harness under `scripts/graddiff/`).

## Verdict

**Yes — there is a concrete update-rule difference, and it is per-sample loss
weighting.** On identical tokens, rollout logprobs, advantages and LoRA-A
initialisation, the two trainers agree on the forward pass (trainer-logprob
mean |Δ| 9.7e-3, k2 3.8e-4 — the same magnitude as either trainer vs the
sampler) and on every optimizer-side quantity (DP mean-reduce, Adam
hyper-parameters and first-step |ΔB| ≈ lr, no clipping active, α/r), but the
gradient that reaches the optimizer is not the same function of the data:

- **Miles** effectively minimises a *per-sample mean*: each response's PPO
  objective is divided by that response's own token count. Its Megatron model
  config has `calculate_per_token_loss=False` even though the launcher passes
  `--calculate-per-token-loss` (`args.calculate_per_token_loss=True`); the
  flag is not propagated into the Bridge-built Qwen3.5 model config, so
  `megatron/core/pipeline_parallel/schedules.py` takes the non-per-token branch
  and divides each microbatch's loss by that microbatch's `num_tokens` (the
  response length under dynamic single-sample microbatching) and by
  `num_microbatches`. Runtime proof: the instrumented run logs
  `args_calculate_per_token_loss=true`,
  `model_config_calculate_per_token_loss=false` on all 8 ranks and one
  `loss_function` call per sample with `num_tokens = response_len`
  (236, 239, 479, 784, 1083, 1228, 1474, …).
- **Lilo** effectively minimises a *token mean*: the Tinker client pre-divides
  advantages by the batch's total action tokens (151,790 here). Lilo's Miles
  backend also never sets `--calculate-per-token-loss`, so it goes through the
  same per-microbatch division — but its Tinker `loss_mask` is
  `ones(target_len)` (prompt + response, `miles/tinker/runtime.py`), so the
  divisor is the ~10–16k full sequence length, nearly constant across datums,
  and the client's 1/T weighting dominates. (The full-length divisor is
  inferred from source; re-running Lilo with advantages scaled by
  `L_full_i` moved the cosine to Miles by only ±0.04, which is the expected
  insensitivity to a ≤1.6× per-sample perturbation, so it is neither confirmed
  nor refuted empirically.)

Relative to Lilo, Miles therefore weights response *i* by `mean_len / len_i`:
in this batch a 236-token response gets ~17× the weight of a 4096-token one.
This reproduces the measurements quantitatively:

- raw full batch: B-grad norm ratio Miles/Lilo 1.46 (uniform across modules)
  with flattened cosine 0.65;
- emulating Miles' weighting inside Lilo (`adv_i × T /(n · len_i)`, same A)
  brings the ratio to 1.06 and the cosine to 0.81 (layers ≥10: 0.95–0.97),
  against a Miles-vs-Miles bit-identical-rerun floor of 0.88 (layers 0–3:
  0.65–0.72, layers ≥15: ≥0.99); group 13 reaches 0.97;
- the short-response group 0, where 1/len_i is nearly constant, already
  agrees without reweighting (ratio 0.99, cosine 0.94), and the disagreement
  grows with response-length spread (group 8: 1.52 / 0.88 → 1.08 / 0.91).

Everything else matched: per-datum PPO losses agree to 1e-6 (same clipping,
masking, advantages), Lilo's full-batch gradient equals the sum of its 16
per-group gradients (cosine 0.9999), every response window contributes a
non-zero Lilo gradient, and TIS/KL/entropy are off in both arms.

What this does *not* establish: that the weighting difference is what costs
Miles the reward. That is the natural hypothesis (sample-mean GRPO
down-weights long, mostly-correct responses; with `max_gen 4096` and
`≥10k` prompts the response-length spread is large), but it needs a
multi-step run. Discriminating check: run Miles with the flag actually
propagated to the model config (or Lilo with sample-mean advantages,
`adv_i × T/(n·len_i)`) on the pinned prompts and compare the reward curve to
`slim-hill` / `czzwpqau`.

## Setup

- **Batch.** The step-0 export of `czzwpqau` was unrecoverable (written to
  `/root` inside the client container; W&B keeps one rendered group per
  step), so the batch was regenerated: seed-0 first batch of the pinned
  prompt file (sha256 `4a54952d…9008f` verified), sampled on `lilo-dp2` from a
  fresh r32 LoRA with zero-init B (= base model), temperature 1.0,
  `max_tokens 4096`, thinking off. 24 groups rolled, 8 constant-reward groups
  dropped → 16 groups × 8 = 128 datums, 151,790 action tokens, reward mean
  0.325, mean response 1186 tokens (166–4096). Advantages std-normalised per
  group and divided by total action tokens exactly as the `czzwpqau` client.
  Serialized per datum: `input_tokens`, `target_tokens`, `sampled_logprobs`,
  `advantages`, `mask`, `reward`, `response_len`. Stored at Modal volume
  `lilo-graddiff:batch/batch.json`; `scripts/graddiff/make_batch.py`.
- **Lilo arm** (`scripts/graddiff/lilo_arm.py`). Isolated deployment
  `lilo-graddiff` (`LILO_APP_NAME=lilo-graddiff`; `lilo`/`lilo-dp2`
  untouched) with an env-gated dump hook in
  `src/lilo/backends/miles_runtime/graddiff.py` (DP-local and reduced grads,
  params before/after). Fresh r32 LoRA, one PPO `forward_backward`
  (clip 0.8/1.28), one `optim_step` at 1e-4.
- **Miles arm** (`scripts/graddiff/miles_arm.py`, `miles_hook.py`). Raw Miles
  pinned to the `slim-hill` era (`radixark/miles:dev-202608120325`, Miles
  `ef3807c`, Megatron-LM `73b54618`, Megatron-Bridge `40b93089`), same flags
  as `slim-hill` but `--use-rollout-logprobs` and TIS off, `--eps-clip 0.2
  --eps-clip-high 0.28`, KL/entropy 0, per-token loss flag, dynamic batching
  with `max_tokens_per_gpu 16384`, TP4 × DP2. The fixed batch is injected with
  `--load-debug-rollout-data`; a before-train-step hook copies Lilo's LoRA-A
  into Miles (bit-exact on 80/80 language-model modules, including the fp32
  master copies) and dumps grads/params/logprobs at optimizer-step entry.
  `MILES_GRADDIFF_PROBE=1` additionally logs the model-config flag and every
  `loss_function` call.
- **Comparison** (`scripts/graddiff/compare.py`, `pergroup.py`,
  `lilo_reweight.py`, `lilo_perdatum.py`, `lilo_window.py`). Adapter names are
  aligned `slot{k}.module.module.<path>.adapters.{k}.linear_{in,out}.weight`
  ↔ Miles `<path>.adapter.linear_{in,out}.weight`; gradients are the DP mean
  of the dumped locals. Qwen3.5-9B has 8 full-attention layers (qkv/proj
  targets) and 32 MLP layers (fc1/fc2), so 80 B modules receive gradient;
  A-gradients are exactly 0 with B = 0.

Reproduce (all with `MODAL_SERVER_URL=https://api.modal.com MODAL_PROFILE=`):

```bash
uv run modal run scripts/graddiff/make_batch.py
LILO_APP_NAME=lilo-graddiff LILO_GRADDIFF_DUMP_DIR=/graddiff/lilo_dump \
  uv run modal deploy -m lilo.providers.modal.definitions.qwen3_5_9b_miles_lora_16k_dp2
uv run modal run scripts/graddiff/lilo_arm.py
uv run modal run scripts/graddiff/miles_arm.py            # full batch
uv run modal run scripts/graddiff/miles_arm.py --datum-range 104-111   # one group
uv run modal run scripts/graddiff/lilo_reweight.py --groups -1,13,7
uv run python scripts/graddiff/compare.py --work ~/work/graddiff
```

## Results

All comparisons are on LoRA-B (`linear_out`) gradients, DP-mean of `grads_local` per TP rank, fp64.
"raw ratio" = ||Miles|| / ||Lilo||; "corr ratio" = raw × T/t_group where t_group = response tokens in the sub-batch
(corrects Miles' own 1/T′ advantage normalization vs Lilo's fixed 1/T = 1/151790). cos is scale-free.
LoRA-A copied bit-exact (80/80) into Miles before each run; A-source noted per row.

### Forward / scalar parity (full batch, run ufl8zhau vs Lilo run 1)

| quantity | Lilo | Miles | Δ / note |
|---|---|---|---|
| train loss | 0.150236 | 0.150400 | Δ ≈ 1.6e-4 (Miles repeat: 0.150282) |
| train grad_norm | 0.08784 | 0.12825 | ratio 1.46 |
| trainer vs sampler logprobs, mean \|Δ\|/token | 0.0102 | 0.0103 | essentially identical |
| Lilo-vs-Miles trainer logprobs | mean \|Δ\| 0.0097, k2 3.8e-4 | | forward parity OK |
| advantages | GRPO (r−μ)/(σ+ε) | same rule | match up to Lilo's 1/T factor |
| per-datum loss | — | — | matched to 1e-6 |
| DP reduce | mean | mean | verified per-param |
| optim step | Adam lr1e-4 β.9/.95 | ChainedOptimizer Adam, same | clip 1.0 inactive both |
| param delta | \|\|d\|\|/(lr√N) ≈ 0.98–1.00; max\|d\| ≈ lr; frac\|d\|<0.5·lr ≈ 0.0002–0.005 (both arms) | | textbook first Adam step |

### Gradient comparisons

| comparison | raw ratio | corr ratio | flat cos | per-mod cos min/mean | per-layer cos (0–3 / ≥8 range) |
|---|---|---|---|---|---|
| full batch, Lilo run1 vs Miles ufl8zhau | 1.460 | — | 0.651 | 0.328 / 0.697 | 0.39–0.47 / 0.72–0.76 |
| └ per-type cos | fc1 0.705, fc2 0.642, proj 0.689, qkv 0.562 | | | | |
| Miles run1 vs repeat dc5gfl4s (same A) | 1.000 | — | mean 0.941, min 0.401 | | 0.65–0.69 / 0.98–0.996 |
| g0 (datums 0–7, short) vs Lilo g0 | 77.9 | 0.988 | 0.944 | min 0.919 | n/a |
| g7 (56–63, long) vs pergroup g7 | 18.71 | 1.171 | 0.783 | 0.313 / 0.876 | 0.35–0.46 / 0.89–0.98 |
| g8 (64–71) vs pergroup g8 | n/a | 1.52 | 0.88 | n/a | n/a |
| g9 (72–79, longest) vs pergroup g9 | 9.16 | 1.402 | 0.884 | 0.686 / 0.910 | 0.69–0.75 / 0.92–0.97 |
| g13 (104–111, mixed) vs pergroup g13 | 32.65 | 1.263 | 0.784 | 0.675 / 0.783 | 0.69–0.75 / 0.78–0.83 |
| same-A full (lgwngzw6) vs pergroup full | 1.486 | — | 0.602 | 0.278 / 0.682 | 0.32–0.38 / 0.72–0.76 |
| same-A full (lgwngzw6) vs sample-mean-reweighted Lilo | 1.059 | — | 0.806 | n/a | n/a (local dump partial) |
| sample-mean-reweighted g13 vs miles g13 (s1 A) | 1.028 | — | 0.970 | n/a | n/a |
| sample-mean-reweighted g7 vs miles g7 (s1 A) | 1.110 | — | 0.786 | n/a | n/a |
| g7 same-A Miles repeat (s24oefd8 vs m973jic8) | 0.942 | — | 0.920 | 0.654 / 0.951 | 0.66–0.77 / 0.97–0.998 |

### fullmask reweight (adv′ᵢ = advᵢ·T·L_fullᵢ/(n·lenᵢ)) vs slot-0-A Miles refs

Lilo reweight client got slot0 A, bit-exact 80/80 vs `lilo_perdatum/g13_d0` A — same A as `*_s0` dumps.
Raw norm ratios are dominated by the ~L_full (≈11k) scale factor and are not meaningful as equality tests.

| comparison | raw ratio | flat cos | per-mod cos min/mean | per-layer (0–3 / ≥8) |
|---|---|---|---|---|
| fullmask_g13 vs miles_dump_g13_s0 (8dpix0pk) | 0.0022 | 0.950 | 0.825 / 0.964 | 0.86–0.90 / 0.95–0.99 |
| fullmask_g7 vs miles_dump_g7_s0 (s24oefd8) | 0.0016 | 0.823 | 0.421 / 0.884 | 0.43–0.54 / 0.88–0.98 |
| fullmask_full vs miles_dump_full_s0 (xdcls4ok) | 9.2e-5 | 0.842 | 0.463 / 0.895 | 0.49–0.58 / 0.93–0.97 |

### Probe run (graddiff-g13-probe3, run moww3myv) — decisive config finding

From `rank*_probe_config.json` (all 8 ranks identical):

| field | value |
|---|---|
| args.calculate_per_token_loss | **true** |
| model_config.calculate_per_token_loss | **false** |
| ddp_config.grad_reduce_in_fp32 | true |
| ddp_config.average_in_collective | false |
| ddp_config.gradient_scaling_factor | null |
| ddp_config.use_distributed_optimizer | true |

Per-microbatch loss calls (`rank*_probe_loss_calls.jsonl`): 1 sample per microbatch;
`num_tokens = loss_mask.sum() = response_length` (236, 239, 479, 784, 1083, 1228, 1474, …).
Because the model config has per-token-loss **False**, Megatron `schedules.py` divides each microbatch by
its own num_tokens → **Miles weights datums by 1/len_i (sample-mean)** despite args=True.
Lilo's Megatron backend likewise runs per-token-loss=False but its `loss_masks = ones(target_len)` divide by
**full sequence length** L_full — the weighting asymmetry between the two arms.

### W&B runs (project miles-lora-longcontext, group graddiff-step0, entity modal-labs)

| run id | URL | what |
|---|---|---|
| ufl8zhau | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/ufl8zhau | Miles full batch run1 (slot0-A of Lilo run1) |
| f2j2dtfl | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/f2j2dtfl | Miles full-batch pre-fix attempt (partial) |
| dc5gfl4s | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/dc5gfl4s | Miles full batch repeat (nondeterminism baseline) |
| hzrxow1o | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/hzrxow1o | Miles g0 |
| jovndej2 | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/jovndej2 | Miles g8 (slot1 A) |
| q54u69cg | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/q54u69cg | Miles g7 (slot1 A) |
| 6rxivplk | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/6rxivplk | Miles g13 (slot1 A) |
| vho4bkcp | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/vho4bkcp | Miles g9 (slot1 A) |
| lgwngzw6 | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/lgwngzw6 | Miles full (slot1 A) |
| 8dpix0pk | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/8dpix0pk | Miles g13 (slot0 A, g13_d0) |
| s24oefd8 | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/s24oefd8 | Miles g7 (slot0 A) |
| m973jic8 | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/m973jic8 | Miles g7 (slot0 A) repeat |
| jnoxlsq8 | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/jnoxlsq8 | Miles g13 probe (config logged; loss-call wrap inert) |
| moww3myv | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/moww3myv | Miles g13 probe3 (loss-call jsonl) |
| xdcls4ok | https://wandb.ai/modal-labs/miles-lora-longcontext/runs/xdcls4ok | Miles full (slot0 A) — ref for fullmask_full |

W&B artifact `graddiff-step0-dumps` (batch, outputs, tables, probe logs, manifest of Modal-volume dump paths): run https://wandb.ai/modal-labs/miles-lora-longcontext/runs/9ux0px6q.
