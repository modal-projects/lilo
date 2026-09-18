# lilo-27b launch helpers

Operational helpers for the Qwen3.8-27B long-context ramp. Not part of the
library; kept here so a second session can reproduce a run without guessing.

Common env for every command below:

```bash
export MODAL_SERVER_URL=https://api.modal.com
export MODAL_ENVIRONMENT=micah-dev
PY=/path/to/lilo/.venv/bin/python
```

## 1. Deploy an isolated Lilo app

`LILO_APP_NAME` names the app and is what keeps this deployment independent of
the shared `lilo` app (it is also read by the trainer/reconciler at runtime).

```bash
LILO_APP_NAME=lilo-27b $PY -m modal deploy src/lilo/providers/modal/app.py
```

A second, parallel app is the same command with a different name:

```bash
LILO_APP_NAME=lilo-27b-b $PY -m modal deploy src/lilo/providers/modal/app.py
```

It prints its own server URL (`…--lilo-27b-b-server.us-west.modal.run`), which
is what the client's `base_url` must point at.

## 2. Deploy the chained client app

```bash
$PY -m modal deploy scripts/lilo27b/run_client_chained.py
```

Each invocation resumes from the latest checkpoint in
`/runs/<run_name>/checkpoints.jsonl` on the `lilo-27b-runs` Volume, reuses the
W&B id in `/runs/<run_name>/wandb_run_id.txt`, and spawns its successor before
Modal's 24 h function cap. Redeploying is safe while a run is in flight: with
the default rolling strategy, old containers keep their assigned input.

## 3. Launch a run

```bash
$PY scripts/lilo27b/spawn_chained_128k.py   # 8-GPU trainer, 120k-token prompts
$PY scripts/lilo27b/spawn_chained_256k.py   # 24-GPU (3 x 8) trainer, TP8xCP3
```

## 4. Sample-mean arm on the isolated `lilo-27b-b` app

`--sample-mean-advantages` scales each trajectory's advantage by
`1/(n_trajectories * len_i)` so every sample contributes equally to the
(sum-reduced) server loss; `--per-token-loss-scale` scales by `1/total_tokens`
(token-mean). `--per-token-loss-scale` alone unset is a raw sum, not
sample-mean. The run logs `cmp/sample_mean_advantages` next to the other
`cmp/*` series.

```bash
LILO_APP_NAME=lilo-27b-b $PY -m modal deploy src/lilo/providers/modal/app.py
LILO_CLIENT_APP_NAME=lilo27b-client-chained-b $PY -m modal deploy scripts/lilo27b/run_client_chained.py
LILO_CLIENT_APP_NAME=lilo27b-client-chained-b \
LILO_BASE_URL=https://modal-labs-micah-dev--lilo-27b-b-server.us-west.modal.run \
  $PY scripts/lilo27b/spawn_chained_128k_samplemean.py
```

Per-run artifacts land on the `lilo-27b-runs` Volume under `/<run_name>/`:
`metrics.jsonl` (per-step), `iteration_NNNNNN/train_rollout_summaries.jsonl`
(per-sample reward / response length), `checkpoints.jsonl`, `config.json`,
`timing_spans.jsonl`, `logs.log`.
