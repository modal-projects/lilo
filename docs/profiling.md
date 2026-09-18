# Profiling a training step

Lilo can record a [`torch.profiler`](https://pytorch.org/docs/stable/profiler.html)
trace of one training step on the Miles LoRA trainer. The trace shows, on a
timeline, what the GPU and CPU were doing during that step: the forward/backward
pass, the optimizer update, the LoRA weight export, and the gaps in between.
It is the fastest way to answer "where does my step time go?"

Profiling is off by default and has no overhead until you turn it on.

## 1. Enable it

Pick the step you want to trace and set one environment variable in the shell
you deploy from. The trainer inherits it:

```bash
LILO_TORCH_PROFILE_STEP=2 uv run modal deploy -m lilo.providers.modal.app
```

Then run your training loop as usual. Steps are counted from 0, so `2` traces
the third optimizer step. Use a small step index (1 or 2) rather than 0, so
warm-up work such as compiling kernels is not in the trace.

Optional knobs:

| Variable | Default | What it does |
| --- | --- | --- |
| `LILO_TORCH_PROFILE_STEP` | unset (off) | Which step to trace. Setting it turns profiling on. |
| `LILO_TORCH_PROFILE_RANKS` | `0` | Which trainer GPUs (ranks) to trace, e.g. `0,4` or `all`. Each trace is ~250 MB, so start with one. |
| `LILO_TORCH_PROFILE_DIR` | `/checkpoints/torch-profile/<definition id>` on the checkpoint volume | Where trace files are written. |

Deploy without `LILO_TORCH_PROFILE_STEP` to turn profiling off again.

## 2. What gets captured

- **One optimizer step.** Recording starts when the first `forward_backward`
  of the chosen step arrives at the trainer and stops right before the next
  step begins, so the trace covers all of that step's forward/backward calls,
  the `optim_step`, and the LoRA weight export that follows.
- **Rank 0 only, by default.** Trainers run one process per GPU; the trace is
  per process. Rank 0 is representative for data-parallel training; trace more
  ranks when you suspect imbalance (e.g. across tensor-parallel or context-parallel
  groups).
- **CPU and GPU activity**, with Lilo's phases labelled `lilo/forward_backward`,
  `lilo/optim_step`, `lilo/forward_only`, and `lilo/export_slot_peft` so you
  can find them quickly among the kernel names.
- A separate, CPU-only trace of the trainer's controller process, which covers
  the post-step sampler-weight save and publish.

Output files, per traced rank:

- `rank<N>.trace.json.gz` — the timeline, in Chrome trace format.
- `rank<N>.key_averages.txt` — a plain-text table of the top operators by time,
  handy for a first glance without a UI.

plus `controller.trace.json.gz` / `controller.key_averages.txt` for the
controller process.

## 3. View the trace in Perfetto

1. Download the trace from the checkpoint volume:

   ```bash
   uv run modal volume get lilo-checkpoints \
     /torch-profile/<definition id>/rank0.trace.json.gz .
   ```

   Or attach it to your W&B run so it lives next to the metrics (it appears
   under the run's **Files** tab as `profiler/rank0.trace.json.gz`):

   ```bash
   uv run scripts/upload_torch_profile.py --dir <trace dir> \
     --entity <entity> --project <project> --run-id <wandb run id>
   ```

2. Open <https://ui.perfetto.dev>, click **Open trace file**, and pick the
   `.json.gz` file (no need to decompress it). Everything runs locally in your
   browser.

3. Reading the timeline:
   - The top rows are **CPU threads** (Python and the PyTorch dispatcher); the
     rows named `stream N` are **GPU streams**, where kernels actually run.
   - Press `W`/`S` to zoom and `A`/`D` to pan. Type `lilo/` in the search box
     (`Ctrl+F` / `Cmd+F`) to jump between Lilo's phases.
   - Click a slice to see its duration and, for GPU kernels, the CPU call that
     launched it. Select a range (drag on the timeline) to get a time breakdown
     of everything inside it.

What to look for:

- **Idle GPU streams** inside `lilo/forward_backward`: gaps mean the GPU is
  waiting on the CPU (data prep, Python overhead) or on communication.
- **Long communication kernels** (`ncclKernel_*`, all-gather / reduce-scatter):
  the cost of tensor/context/data parallelism.
- **Time outside the labelled phases**: work between steps that is not
  training, such as weight export or checkpoint writes.
