"""Re-log a Miles 27B run's per-step metrics into a fresh W&B run using the cmp/* keys.

Runs on Modal so the wandb-secret is available. Reads the source run's history
via the W&B API (Miles logs rollout/, train/ and perf/ metrics in separate
rows; they are merged on ``rollout/step``), then writes cmp/* metrics with the
same definitions as the 9B re-log into a new run ``miles-27b-<ctx>k-<steps>``
in group ``baseline-miles-27b``.

Usage:
    MODAL_ENVIRONMENT=micah-dev uv run --with modal modal run \
        scripts/relog_miles_cmp_27b.py --source-run <run_id> --context-k 16 --steps 5 \
        --topology "1x8 H200 TP4xCP1xDP2 + 8x1 sglang" \
        --overrides '{"4": {"perf/step_time": 148.6, "perf/actor_train_time": 110.4}}'
"""

import json

import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install("wandb")
app = modal.App("relog-miles-cmp-27b", image=image)

ENTITY = "modal-labs"
PROJECT = "miles-lora-longcontext"
GROUP = "baseline-miles-27b"
SAMPLES = 128.0  # 16 groups x 8 per step
TRAINER_GPUS = 8.0


def merge_by_step(history: list[dict], overrides: dict[int, dict]) -> dict[int, dict]:
    merged: dict[int, dict] = {}
    for row in history:
        step = row.get("rollout/step")
        if step is None:
            continue
        merged.setdefault(int(step), {}).update(
            {k: v for k, v in row.items() if v is not None and not k.startswith("_")}
        )
    for step, extra in overrides.items():
        merged.setdefault(step, {}).update(extra)
    return merged


def cmp_metrics(m: dict) -> dict:
    d: dict[str, float] = {}
    if "rollout/raw_reward" in m:
        d["cmp/reward_mean"] = float(m["rollout/raw_reward"])
    if "rollout/response_lengths" in m:
        d["cmp/response_len_mean"] = float(m["rollout/response_lengths"])
    if "rollout/total_lengths" in m and "rollout/response_lengths" in m:
        d["cmp/prompt_len_mean"] = float(m["rollout/total_lengths"]) - float(
            m["rollout/response_lengths"]
        )
    if "rollout/truncated" in m:
        d["cmp/truncated_ratio"] = float(m["rollout/truncated"])
    if "perf/actor_train_time" in m:
        d["cmp/train_time_s"] = float(m["perf/actor_train_time"])
    if "perf/rollout_time" in m:
        d["cmp/rollout_time_s"] = float(m["perf/rollout_time"])
    if "train/loss" in m:
        d["cmp/loss"] = float(m["train/loss"])
    if "perf/step_time" in m:
        st = float(m["perf/step_time"])
        d["cmp/step_time_s"] = st
        d["cmp/samples_per_s"] = SAMPLES / st
        if "rollout/total_lengths" in m:
            d["cmp/tokens_per_gpu_per_s"] = (
                SAMPLES * float(m["rollout/total_lengths"]) / st / TRAINER_GPUS
            )
    return d


@app.function(secrets=[modal.Secret.from_name("wandb-secret")], timeout=600, retries=0)
def run(
    source_run: str, context_k: int, steps: int, topology: str, overrides: str
) -> str:
    import wandb

    src = wandb.Api().run(f"{ENTITY}/{PROJECT}/{source_run}")
    ov = {int(k): v for k, v in json.loads(overrides or "{}").items()}
    merged = merge_by_step(list(src.scan_history()), ov)

    new = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        name=f"miles-27b-{context_k}k-{steps}",
        config={
            "source_run": source_run,
            "framework": "miles+modal",
            "model": "Qwen/Qwen3.8-27B",
            "dataset": "Guanzheng/LongRLVR-Data",
            "context": context_k * 1024,
            "topology": topology,
            "note": "cmp/* re-log of the source run; values copied from its metrics",
        },
    )
    for step in sorted(merged):
        d = cmp_metrics(merged[step])
        if d:
            new.log(d, step=step)
    url = new.url
    new.finish()
    return url


@app.local_entrypoint()
def main(
    source_run: str,
    context_k: int = 16,
    steps: int = 5,
    topology: str = "",
    overrides: str = "{}",
):
    print(run.remote(source_run, context_k, steps, topology, overrides))
