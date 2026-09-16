"""Backfill common comparison metrics for a Miles baseline W&B run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import wandb


def _read_rows(path: Path) -> dict[int, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = {}
        for row in csv.DictReader(stream):
            step = int(row.get("internal_step", row.get("step", "0")))
            rows[step] = {
                key: float(value)
                for key, value in row.items()
                if value not in (None, "")
                and key not in {"trace_status", "fatal_or_oom_or_nccl"}
            }
        return rows


def _merge_rows(paths: list[Path]) -> list[dict[str, float]]:
    merged: dict[int, dict[str, float]] = {}
    for path in paths:
        for step, row in _read_rows(path).items():
            merged.setdefault(step, {}).update(row)
    return [merged[step] | {"step": step} for step in sorted(merged)]


def _comparison_row(
    row: dict[str, float],
    trainer_gpus: int,
    prompt_len_mean: float | None,
) -> dict[str, float]:
    if "cmp/reward_mean" in row:
        return {
            key: value
            for key, value in row.items()
            if key.startswith("cmp/") and isinstance(value, (int, float))
        }
    step_time = row.get("step_time_s", row.get("step_time", 0.0))
    response_len = row.get("response_length_mean", row.get("response_len", 0.0))
    samples = row.get("samples", 128.0)
    prompt_len = row.get("prompt_length_mean", prompt_len_mean)
    result = {
        "cmp/reward_mean": row.get("reward_mean", row.get("reward", 0.0)),
        "cmp/response_len_mean": response_len,
        "cmp/step_time_s": step_time,
        "cmp/rollout_time_s": row.get("rollout_time_s", 0.0),
        "cmp/samples_per_s": samples / step_time if step_time else 0.0,
    }
    if "train_time_s" in row:
        result["cmp/train_time_s"] = row["train_time_s"]
    if step_time and prompt_len is not None:
        result["cmp/tokens_per_gpu_per_s"] = (
            (prompt_len + response_len) * samples / step_time / trainer_gpus
        )
    if "train_loss" in row:
        result["cmp/loss"] = row["train_loss"]
    if "truncated_ratio" in row:
        result["cmp/truncated_ratio"] = row["truncated_ratio"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        action="append",
        type=Path,
        required=True,
        help="CSV containing baseline fields or already-normalized cmp/* fields.",
    )
    parser.add_argument("--name", default="miles-6a-baseline")
    parser.add_argument("--group", default="baseline-miles")
    parser.add_argument("--project", default="miles-lora-longcontext")
    parser.add_argument("--entity", default="modal-labs")
    parser.add_argument("--trainer-gpus", type=int, default=8)
    parser.add_argument(
        "--prompt-len-mean",
        type=float,
        help="Fallback Miles prompt length for the Lilo-compatible throughput metric.",
    )
    args = parser.parse_args()

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        group=args.group,
        name=args.name,
        config={
            "trainer_gpus": args.trainer_gpus,
            "prompt_len_mean": args.prompt_len_mean,
            "sources": [str(p) for p in args.source],
        },
    )
    for row in _merge_rows(args.source):
        wandb.log(
            _comparison_row(row, args.trainer_gpus, args.prompt_len_mean),
            step=int(row["step"]),
        )
    run.finish()
    print(run.url)


if __name__ == "__main__":
    main()
