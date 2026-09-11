# Multi-LoRA through Tinker, Lilo, and Miles

`scripts/e2e_miles_multi_lora.py` keeps two Tinker training clients alive on the
same control-plane server: GSM8K at rank 16 and DAPO-Math at rank 32. Both use the
Miles backend. Sampling and training run concurrently across clients. The test
uses Tinker APIs for forward/backward, optimizer steps, named sampler exports,
and inference; it does not run Miles's native rollout driver or push weights
directly to SGLang.

The default uses the existing `qwen3_5_9b_base_miles_lora_16k` definition:
Qwen3.5-9B-Base, four H100 trainer GPUs with TP4, four adapter slots, maximum
rank 32, and 16k context. Inference uses the definition's separate autoscaling
LoRA pool (one H200 per replica, at most two replicas). This is a transport/lifecycle E2E adaptation
of the upstream two-dataset test, not a reproduction of its Qwen3-4B learning
curve or GRPO hyperparameters.

## Prepare data

From the repository root, in an environment with the project, `datasets`, and
`transformers` installed:

```bash
python scripts/prepare_miles_multi_lora_data.py --rows 320
```

This downloads 320 eligible training examples per dataset from `openai/gsm8k`
and `open-r1/DAPO-Math-17k-Processed`. It keeps numeric-answer examples that fit
the context budget and writes source row indices. For another tokenizer or
context/generation budget, pass `--model`, `--context-length`, and
`--max-new-tokens` consistently. The preparation command uses no GPUs.

Alternatively supply your own JSONL files, with one object per line:

```json
{ "prompt": "What is 6 times 7?", "answer": "42" }
```

Answers must be numeric; dataset readers also accept `#### 42` or `\boxed{42}`.
The rollout grader accepts a final boxed number or a response that is entirely
numeric. It is deliberately an exact numeric smoke-test grader, not a symbolic
math benchmark grader. Dataset contents are hashed in the run report.

## Run on an isolated app

Configure Modal credentials, access to the existing Lilo secrets/volumes, and
`TINKER_API_KEY`, then run:

```bash
python scripts/e2e_miles_multi_lora.py \
  --gsm8k scripts/data/multi_lora/gsm8k.jsonl \
  --dapo scripts/data/multi_lora/dapo.jsonl \
  --steps 20 --groups 16 --samples 8 --max-new-tokens 8192
```

This launches an ephemeral Lilo control-plane app. Before importing the app's
GPU definitions it sets `LILO_TRAINER_MAX_CONTAINERS=1`. Each client is created
sequentially so partial creation can be cleaned up, but both remain alive while
rollouts, training, and publications run concurrently. The capacity limit is
not the proof of colocation: the test independently verifies actual placement.
Each adapter keeps up to four prompt groups in flight, preserving dataset order
when assembling the training batch.
The shared inference pool is deployed by Lilo's normal path and retains its
normal idle scale-down behavior; this script does not stop shared pool apps.

To use an already running server:

```bash
python scripts/e2e_miles_multi_lora.py \
  --base-url https://YOUR-LILO-SERVER \
  --app-id ap-YOUR-CONTROL-PLANE-APP-ID \
  --gsm8k scripts/data/multi_lora/gsm8k.jsonl \
  --dapo scripts/data/multi_lora/dapo.jsonl
```

The app ID and Modal credentials are required for read-only placement and
artifact inspection. Use an otherwise idle Miles trainer with at least two
free slots. Existing-server mode does not change its scaling limits. It fails
if the scheduler places the clients on separate engines rather than silently
accepting that configuration.

## Assertions and report

The timestamp-independent unique report is written under `scripts/results/`
(or `--output` with a unique suffix). It records:

- Distinct model IDs sharing an engine instance and boot ID, with both listed
  by that engine's resident-model endpoint. Placement is rechecked throughout.
- Initial named sampler artifacts associated with the correct model/definition,
  followed by successful inference through Lilo's LoRA sidecar and snapshot
  bulletin. Each client publishes increasing versions at each RL iteration.
- An A-only supervised update measurably changing A's logprobs while idle B's
  logprobs remain unchanged and B's exported adapter files retain identical
  hashes. B then receives its own supervised warmup. Before these baselines,
  both clients receive a zero-loss, zero-learning-rate kernel warmup; each
  optimizer's step counter advances once, with zero gradients and moments.
- Concurrent math rollouts and importance-sampling updates with per-prompt-group
  mean-centered rewards. Generated token IDs and their logprobs are preserved;
  prompt advantages are zero and targets remain correctly shifted.
- Finite training/optimizer metrics, reward means, nonzero-advantage group counts,
  and elapsed times for each adapter.
- New runs also record mean signed, mean absolute, p95 absolute, and maximum
  trainer-versus-rollout logprob differences on generated tokens. These use the
  actual forward/backward outputs and sampled token spans, including zero-
  advantage groups. They do not add model calls or change the training loss.
  The original 20-step run predates this instrumentation.
- Final trainer/sampler logprob parity, both samplers changing from initialization,
  and an old named A snapshot retaining its original logprobs after newer exports.
- Unloading A removes it from the trainer; B can still forward and sample on the
  same engine. Remaining models are unloaded on success or failure; cleanup
  errors fail the run and are recorded.

Default absolute tolerances are 0.2 for trainer/inference logprob parity,
1e-5 for isolation and old-snapshot stability, and 1e-5 minimum observable weight
change. These are configurable gates, not measured accuracy guarantees.

A sparse math batch may give all samples the same reward, yielding zero RL
advantages. This is reported explicitly and does not fail the infrastructure
smoke test: the supervised warmups separately require actual parameter updates.
Passing does not demonstrate RL convergence, full optimizer-state isolation,
or parity with independent single-adapter training runs. It also does not prove
that every concurrent request pair became one packed microbatch: it proves
concurrent clients share one trainer, leaving coalescing to the normal scheduler.

No production endpoint was added. Engine authentication tokens are used only
for the resident-model read and are never written to the report.

## CPU verification

```bash
python -m pytest tests/scripts/test_miles_multi_lora_e2e.py \
  tests/backends/test_miles.py tests/providers/test_miles_definition.py -q
```

These checks verify failure gates, concurrent dispatch, publication API usage,
and that the TITO payload passes the real Miles adapter's target-shift checks.
They do not replace the GPU E2E run.

The monitored 20-step configuration uses 128 responses per adapter per step
(16 prompt groups × 8 samples), with up to 8192 generated tokens. A PNG with
separate reward, nonzero-advantage, gradient-norm, and loss curves is refreshed
after every completed step. Replot a partial or completed report with:

```bash
python scripts/plot_miles_multi_lora.py scripts/results/YOUR_REPORT.json
```

The 16k definition retains up to 64 registered adapter versions per rollout
replica so both adapters' named snapshots fit across the run.

## Runtime findings (2026-09-09)

The monitored run `ap-IpedNHSv7gfmIJ5IG83AqX` completed all 20 RL updates
for both adapters. Final training-batch rewards were 0.9609375 (GSM8K) and
0.625 (DAPO-Math). Its report is
`scripts/results/miles_multi_lora.6fdddf5e81.json`.
The subsequent final trainer/SGLang parity assertion failed at tolerance 0.2;
the overall E2E therefore failed, and the later snapshot/unload assertions were
not reached. Model cleanup completed without reported errors, and the owned
rollout pool was stopped. The original harness did not persist the final
logprob arrays before asserting. Final inference arrays were subsequently
recovered from retained Modal sampling tasks; the original final trainer arrays
have not been recovered.

The harness now saves probe tokens and final parity arrays/errors before
asserting. A bounded reproduction can check parity after each controlled
supervised update without running RL rollouts:

```bash
python scripts/e2e_miles_multi_lora.py \
  --gsm8k scripts/data/multi_lora/gsm8k.jsonl \
  --dapo scripts/data/multi_lora/dapo.jsonl \
  --steps 0 --diagnostic-updates 20 --max-new-tokens 512
```

This mode uses the same fixed example for both adapters, checks the existing
one-update warmups first, then applies further supervised updates. It stops at
the first parity failure and saves both arrays with their publication versions.
Plot its report with `python scripts/plot_miles_parity.py REPORT.json`.

### Focused parity investigation (2026-09-10)

`scripts/results/miles_parity_diagnostic.02aefd65cf.json` reproduced the failure
after two controlled supervised updates, without RL rollouts:

| Check                                        |  Rank 16 |  Rank 32 |
| -------------------------------------------- | -------: | -------: |
| Initial maximum trainer/inference difference | 0.069036 | 0.069036 |
| After one supervised update                  | 0.093382 | 0.130505 |
| After two updates: maximum                   | 0.336016 | 0.240213 |
| After two updates: mean absolute             | 0.020156 | 0.018643 |
| After two updates: completion-token maximum  | 0.042977 | 0.058215 |

The largest errors were on prompt tokens. The original final inference arrays
were recovered from Modal task results, and loading the original final PEFT
snapshots into a fresh SGLang process reproduced those prompt logprobs exactly.
Both original exports had 478 finite tensors, correct rank/alpha configuration,
and nonzero adapter updates. This reduces the likelihood of transient serving
state, but does not independently prove export equivalence to the trainer.

Independent Hugging Face forwards applied each exported adapter explicitly as
`linear_B(linear_A(x)) * alpha / rank`. All 239 adapter module names matched;
no modules were omitted. On the two-update snapshots, HF BF16 versus SGLang
maximum errors were 0.132343 and 0.124945. Holding the exported weights and HF
implementation fixed while changing BF16 computation to float32 produced
maximum differences of 0.477988 and 0.551466. Float32 matmul used highest
precision (TF32 disabled). This establishes substantial numerical sensitivity
of the fixed-input maximum-error gate; it does not localize a specific kernel
or recover the original final trainer values. The original 0.2 tolerance has
not been loosened, and the original E2E remains failed.

Reproduce the independent reference with:

```bash
modal run scripts/diagnose_miles_parity_reference.py \
  --report scripts/results/miles_parity_diagnostic.02aefd65cf.json \
  --additional-report scripts/results/miles_multi_lora.6fdddf5e81.json \
  --dtype-name float32
```

Use `--dtype-name bfloat16` for the BF16 comparison. Both modes read immutable
snapshots and perform forwards only. The controlled trainer and rollout pool
were stopped after reproducing the failure; both reference apps completed.

The monitored attempts exposed and fixed missing `lora_pool` KV routing,
prompt-logprob parsing of SGLang's `[null, token_id]` first-token entry, and
overlapping volume refresh/adapter registration. Refresh and registration now
share a lock, preventing another request from reloading the mounted volume
while SGLang reads adapter files.

Attempt `ap-wwA7mXjqUTJHYkQCoyr7lm` verified both ranks resident in the same
trainer, successful forwards, named PEFT publication, a healthy SGLang pool,
and loading both adapters into that pool. It failed while parsing initial
sampler prompt logprobs and was stopped before any RL updates. Its report is
`scripts/results/miles_multi_lora.bcc05053d7.json`; this is partial validation,
not a passed E2E or evidence of learning.

The next attempt passed initial sampler parity and completed an A-only update,
then failed the strict idle-B logprob check. A diagnostic retry measured exactly
repeatable trainer baselines but hit a cross-backend parity maximum of 0.124
(TP4 trainer versus TP1 inference). The cross-backend tolerance was adjusted to
0.2; the isolation tolerance remains unchanged. The harness now records numeric
baselines and parity errors and, on isolation failure, compares B's before/after
snapshot manifests and sampler logprobs. These findings remain under investigation.

Attempt `ap-YkFCEu0Xyj8zrIU8LO5WXF` measured A's logprob change as 3.576 and
idle B's trainer change as 0.122. B's exported adapter weights and configuration
had identical SHA256 hashes before/after, and its sampler logprob change was
exactly zero. Subsequent trainer forwards repeated the post-update values
exactly. The test therefore now primes backward kernels with the zero-loss
warmups before measuring isolation, while retaining the strict isolation
tolerance and adding mandatory exported-weight equality.

Attempt `ap-37OOam9OJnx8BNxqDcGPcS` allocated four H100 trainer GPUs and
loaded both adapters. Its H200 rollout pool reached HTTP 200 readiness.
The first zero-weight, zero-learning-rate warmup reported loss 0.0 but
gradient norm 0.841177, failing the zero-gradient assertion. Both apps were
stopped. This invalidates the assumption that the proposed warmup leaves
optimizer moments zero; the harness retains the assertion and no RL steps
have completed. Report: `scripts/results/miles_multi_lora.c91f469055.json`.

A diagnostic retry (`ap-upaSq1ZU0m9pQbuTQJIZ9v`) found zero gradient buffers
before backward, nonzero gradients including the MTP head after backward,
and unchanged frozen base weights. The checkpoint's MTP head was built even
with `enable_mtp_training=False`: Miles's LoRA provider setup does not apply
that flag to checkpoint-derived provider settings. Lilo now disables the
provider's MTP head during actor initialization when that flag is false.
The GPU regression remains the zero-weight loss check above.

With that fix, `ap-IpedNHSv7gfmIJ5IG83AqX` passed both zero-loss warmups
with gradient norm exactly zero, initial sampler parity (maximum error 0.069),
and the A-only update isolation check (A change 2.826, B change exactly zero,
identical B export hashes). Both clients then completed their first 128-response
rollout batches and submitted RL updates. Completion of the 20-step run is
tracked separately in `scripts/results/miles_multi_lora.6fdddf5e81.json`.
