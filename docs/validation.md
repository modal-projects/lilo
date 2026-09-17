# Validation

These runs compare Lilo full-parameter RL training with
[Miles](https://github.com/radixark/miles) on Qwen3.5 and Qwen3.6 reasoning
workloads, with context lengths up to 64K. The plots show reward and timing;
they do not establish identical tokens or updates between runs. Lilo includes
the corresponding trainer definitions.

The first step often includes compilation and cold starts.

## DAPO Math

### Qwen3.5-4B

30-step Lilo and Miles FFT runs on DAPO Math with an 8k generation cap:

![Qwen3.5-4B DAPO Math: Lilo versus Miles](assets/validation/qwen3-5-4b-dapo.png)

### Qwen3.5-9B: Lilo and Miles

This is a 29-step Qwen3.5-9B DAPO Math comparison with async RL and a 20k
generation cap.

![Qwen3.5-9B DAPO Math: Lilo versus Miles](assets/validation/qwen3-5-9b-dapo-vs-miles.png)

Lilo and Miles have comparable reward and step time in this run.

### Qwen3.6-27B

Lilo and Miles FFT runs on DAPO Math with an 8k generation cap. Lilo completed
30 steps; the ongoing Miles run is shown through its latest completed step:

![Qwen3.6-27B DAPO Math: Lilo versus Miles](assets/validation/qwen3-6-27b-dapo.png)

### Qwen3.6-35B-A3B

30-step Lilo FFT on DAPO math with 8k generation cap:

![Qwen3.6-35B-A3B DAPO Math](assets/validation/qwen3-6-35b-a3b-dapo.png)

## LongRLVR: Qwen3.5-35B-A3B

The [LongRLVR dataset](https://huggingface.co/datasets/Guanzheng/LongRLVR-Data)
tests longer contexts. Separate agentic experiments used TerminalBench; those
results are not shown here.

### Asynchronous Lilo and Miles

Lilo vs Miles using async RL on LongRLVR with up to 64k context + generation length:

![Async LongRLVR: Lilo versus Miles](assets/validation/qwen3-5-35b-a3b-longrlvr-async-vs-miles.png)

### Sync RL

Sync RL Lilo vs Miles on LongRLVR:

![Lilo synchronous LongRLVR](assets/validation/qwen3-5-35b-a3b-longrlvr-lilo-sync.png)

![Miles synchronous LongRLVR](assets/validation/qwen3-5-35b-a3b-longrlvr-miles-sync.png)

### Qwen3.5-9B async

This 30-step async FFT run uses Qwen3.5-9B on eight H200s with TP2/CP2/DP2,
16 groups of eight samples per step, an 8K generation cap, and up to 64K context.

The bottom row shows model FLOPs utilization (MFU) and tokens/s/GPU. The MFU
calculation includes the forward pass used for activation recomputation and
uses 989 TFLOP/s dense BF16 per H200 as the denominator. Recompilation accounts
for the ramp over the first five steps; afterward, MFU stays at 26–31%.

![Qwen3.5-9B async LongRLVR](assets/validation/qwen3-5-9b-longrlvr-lilo-async.png)

## Planning a run

See [Working with Full Fine-Tunes](full-fine-tunes.md) before designing an RL
experiment with Lilo.

