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

### Async RL

Six clients, 30 steps each, sharing one 4×H100 trainer and eight H200 inference
workers. GSM8K uses rank 16 and a 4k generation cap; DAPO uses rank 32 and 8k.
All six checkpoint round trips reproduce logprobs and adapter exports exactly.

![Six-client async RL](assets/lora-validation/qwen3-5-9b-async-math.png)

[Config and metrics](assets/lora-validation/async-math.json).

## DAPO Math: Qwen3.5-9B cost estimate

12 clients sharing one 4×H100 trainer and 2–6 H200 inference GPUs. Optimistic
Lilo GPU-cost accounting versus projected Tinker token charges; the Tinker bill
has not been measured on this workload.

![DAPO cost estimate](assets/lora-validation/qwen3-5-9b-dapo-cost-estimate.png)

[Cost assumptions](assets/lora-validation/dapo-cost-estimate.json).

## Codeforces codegolf: Qwen3.5-9B

Four rank-32 clients, 500 updates each, sharing an 8×H200 TP8 trainer and 4–8 H200
inference workers. Async TailRL, learning rate 1e-5, 64k context, 16k generation cap.
Final sample pass rates on 16 held-out problems reach 82–88%.

![Codegolf learning curves](assets/lora-validation/qwen3-5-9b-codegolf-learning.png)

Client-observed training, publication, and rollout timings:

![Codegolf operation timings](assets/lora-validation/qwen3-5-9b-codegolf-timing.png)

[Config and metrics](assets/lora-validation/codegolf.json).
[Figure sources and renderer](assets/lora-validation/README.md).
