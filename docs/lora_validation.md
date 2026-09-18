# LoRA validation

Multi-LoRA RL with Lilo's Miles backend on GSM8K, DAPO Math, and Codeforces codegolf.
Step 0 includes compilation and cold-start times.

## GSM8K and DAPO Math: Qwen3.5-9B-Base

### Numerical parity

30-step deterministic runs: six clients share a 4×H100 TP4 trainer and exactly
match isolated baselines on generated tokens, rewards, logprobs, and adapter
exports. This validates experiment revision `1a4148f`.

![Six-client numerical parity](assets/lora-validation/qwen3-5-9b-parity.png)

[Config and verification](assets/lora-validation/deterministic-parity.json).

### Async RL: Lilo versus native Miles

Both runs complete 30 updates for each of six clients on one 4×H100 TP4 trainer
and eight H200 inference workers. They use the same dataset bytes, prompt order,
grader, learning recipe, and one-batch prefetch. GSM8K uses rank 16 and a 4k
generation cap; DAPO uses rank 32 and 8k. Each update uses 8 prompts × 8 samples
and learning rate 1e-5. Native Miles uses its Tinker gateway and a fixed rollout
pool on Modal.

![Six-client async math: Lilo versus native Miles reward and step timings](assets/lora-validation/qwen3-5-9b-async-math.png)

Faint curves show all three clients per dataset and system; bold curves show the
trailing five-update mean across those clients. Step time includes waiting,
rollout, training, and publication.

| Metric | Lilo + Miles | Native Miles |
| --- | ---: | ---: |
| GSM8K median step | 21.95 s | 17.65 s |
| DAPO median step | 84.28 s | 74.81 s |
| GSM8K mean training reward | 0.659 | 0.642 |
| DAPO mean training reward | 0.156 | 0.169 |
| Training interval | 43.27 min | 38.98 min |

Medians exclude each client's first two updates. The training interval runs from
the first client pipeline start through final publication, excluding infrastructure
startup and initial warmup. Each system has one nondeterministic run; equal GPU
allocation does not imply equal total GPU-hours or generated token counts.

![Six-client async math: training reward versus elapsed time](assets/lora-validation/qwen3-5-9b-async-math-walltime.png)

The native run passed all 180 updates and raw-rollout checks, with finite metrics
and maximum policy lag one. Its deployment uses a router backfill fix and private
HTTP adapter transfer with local caching; transfer and native router retries are
included in timing. [Full setup differences and verification](assets/lora-validation/native-async-math.json).
The separate checkpoint roundtrip checks belong to the Lilo run.

[Lilo config and metrics](assets/lora-validation/async-math.json) ·
[Native Miles config and metrics](assets/lora-validation/native-async-math.json).

## DAPO Math: Qwen3.5-9B cost estimate

12 clients sharing one 4×H100 trainer and 2–6 H200 inference GPUs. Optimistic
Lilo GPU-cost accounting versus projected Tinker token charges; the Tinker bill
has not been measured on this workload.

![DAPO cost estimate](assets/lora-validation/qwen3-5-9b-dapo-cost-estimate.png)

[Cost assumptions](assets/lora-validation/dapo-cost-estimate.json).

For the separate 1–32-client sweep on a fixed 8×H200 trainer and eight H200
inference replicas, see [workload tuning and memory budgeting](multi-lora.md#how-to-optimize-lilo-workloads-for-token-pricing).
It includes total/per-client TPS, token cost, batch sizes, trainer activity over
time, and the sweep's 4 h 43 min elapsed time.

## Codeforces codegolf: Qwen3.5-9B

Four rank-32 clients, 500 updates each, sharing an 8×H200 TP8 trainer and 4–8 H200
inference workers. Async TailRL, learning rate 1e-5, 64k context, 16k generation cap.
Final sample pass rates on 16 held-out problems reach 82–88%.

![Codegolf learning curves](assets/lora-validation/qwen3-5-9b-codegolf-learning.png)

Client-observed training, publication, and rollout timings:

![Codegolf operation timings](assets/lora-validation/qwen3-5-9b-codegolf-timing.png)

[Config and metrics](assets/lora-validation/codegolf.json).
[Figure sources and renderer](assets/lora-validation/README.md).
