# LoRA validation figure sources

These are aggregate metric snapshots and a supplied cost estimate. They contain no
model weights, prompts, completions, or dashboard credentials. The snapshots
retain every plotted update and all participating clients.

| Snapshot | Source | Coverage |
| --- | --- | --- |
| `deterministic-parity.json` | `exact_rl_8x8_30step/run-v1` report, independent baseline/client rollout records, and final verification | 30 updates, six shared clients and two isolated baselines |
| `async-math.json` | `upstream_miles_30step/run-v5/multi/async_rl_report.json` | 30 updates for each of six clients, plus checkpoint checks |
| `codegolf.json` | Modal volume `lilo-multilora-codegolf`, run `tailrl-hero-v3-lr1e5` | Updates 1–500 and evaluations 0–500 every 20 updates for all four clients |
| `dapo-cost-estimate.json` | Experiment author's supplied cost comparison | Separate 12-client DAPO scenario: shared 4×H100 trainer, 2–6 H200 inference GPUs |

The historical source directory name `upstream_miles_30step` identifies a Lilo run
using its Miles backend, not a standalone Miles training run.

The cost snapshot preserves the supplied rounded amounts and assumptions. It
does not include a matched Tinker run, a raw token/GPU billing ledger, or the exact
price snapshot used to calculate the estimate. The pricing link documents the
rate categories; the renderer does not fetch current rates or recalculate the
supplied costs. The displayed Lilo component sum differs from its supplied total
by $0.01.

The math sources were archived in the experiment workspace under
`scripts/results/`. Their original report hashes and available revision identifiers
are retained in the JSON files. Codegolf was fetched from the volume on the
`captured_at` date; its hash identifies the complete fetched snapshot before
projection to the plotted metrics. Its configuration records the dataset hash,
hardware, model, optimizer settings, and per-client settings. Source hashes identify
the underlying records; the large original archives are not included here.

`render.py` reads only the four adjacent JSON files. It asserts complete update
ranges and the plotted numerical equality, preserves raw training observations,
and labels smoothing and timing boundaries. No smoothing is applied to the
deterministic parity curves or Codeforces evaluation points.

```bash
uv run docs/assets/lora-validation/render.py
```

The deterministic run predates the upstream rebase of PR #28. Its exact-parity
results describe that recorded experiment, not a GPU revalidation of the current
PR head. The asynchronous math snapshot retains its original grader's rewards;
it is not an evaluation against the deterministic run's newer grader. Codegolf
uses a separate workload and reward and has no directly matched FFT control here.
