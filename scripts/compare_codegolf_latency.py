"""Compare short codegolf runs, separating sampling, judging, and learner waits."""

from __future__ import annotations

import argparse
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal


def collect(run, steps):
    volume = modal.Volume.from_name("lilo-multilora-codegolf")

    def read(path):
        return json.loads(b"".join(volume.read_file(f"{run}/{path}")))

    result = {
        "run": run,
        "manifest": read("manifest.json"),
        "controller": read("controller.json"),
    }

    def client(index):
        prefix = f"{run}-client-{index}"
        rows = []
        for step in range(1, steps + 1):
            try:
                metric = read(f"{prefix}/metrics/{step:04d}.json")
            except (FileNotFoundError, modal.exception.NotFoundError):
                break
            groups = []
            for entry in volume.listdir(f"{run}/{prefix}/rollouts/step-{step:04d}"):
                if not entry.path.endswith(".json"):
                    continue
                record = json.loads(b"".join(volume.read_file(entry.path)))
                if (
                    record.get("sampling_ticket")
                    != metric["pipeline"]["sampling_ticket"]
                ):
                    raise ValueError(f"Stale rollout record: {entry.path}")
                lengths = [len(row["tokens"]) for row in record["rows"]]
                groups.append(
                    {
                        "problem_id": record["problem_id"],
                        "samples": len(lengths),
                        "mean_tokens": statistics.mean(lengths),
                        "max_tokens": max(lengths),
                        "sampling_seconds": record["sampling_seconds"],
                        "judge_seconds": record["judge_seconds"],
                    }
                )
            if sum(group["samples"] for group in groups) != metric["samples"]:
                raise ValueError(
                    f"Incomplete rollout records: {run} client {index} step {step}"
                )
            rows.append({"metric": metric, "groups": groups})
        return {"client": index, "steps": rows}

    with ThreadPoolExecutor(max_workers=4) as pool:
        result["clients"] = list(pool.map(client, range(4)))
    return result


def summarize(run):
    rows = [row for client in run["clients"] for row in client["steps"]]
    groups = [group for row in rows for group in row["groups"]]
    if not rows:
        return {"updates": 0}
    return {
        "updates": len(rows),
        "samples": sum(group["samples"] for group in groups),
        "sampling_group_seconds": statistics.mean(
            group["sampling_seconds"] for group in groups
        ),
        "judge_group_seconds": statistics.mean(
            group["judge_seconds"] for group in groups
        ),
        "group_max_tokens": statistics.mean(group["max_tokens"] for group in groups),
        "completion_tokens": statistics.mean(
            row["metric"]["completion_tokens"] for row in rows
        ),
        "rollout_batch_seconds": statistics.mean(
            row["metric"]["pipeline"]["rollout_batch_seconds"] for row in rows
        ),
        "learner_rollout_wait_seconds": statistics.mean(
            row["metric"]["pipeline"]["rollout_wait_seconds"] for row in rows
        ),
        "update_seconds": statistics.mean(
            row["metric"]["pipeline"]["update_seconds"] for row in rows
        ),
        "publish_seconds": statistics.mean(
            row["metric"]["pipeline"].get("publish_seconds", 0) for row in rows
        ),
        "step_components_seconds": statistics.mean(
            row["metric"]["seconds"]
            + row["metric"]["pipeline"].get("publish_seconds", 0)
            for row in rows
        ),
        "rollout_persist_seconds": statistics.mean(
            row["metric"]["pipeline"]["rollout_persist_seconds"] for row in rows
        )
        if all("rollout_persist_seconds" in row["metric"]["pipeline"] for row in rows)
        else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results/multilora-codegolf/latency-comparison",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = {}
    for label, name in (("before", args.before), ("after", args.after)):
        path = args.output / f"{label}.json"
        run = (
            json.loads(path.read_text()) if args.offline else collect(name, args.steps)
        )
        if run["run"] != name:
            raise ValueError(f"Cached run does not match {name}")
        path.write_text(json.dumps(run, indent=2) + "\n")
        runs[label] = run
    for key in (
        "config",
        "dataset_sha256",
        "clients",
        "rank",
        "context",
        "trainer_gpu",
        "tp",
        "cp",
        "dp",
    ):
        if runs["before"]["manifest"][key] != runs["after"]["manifest"][key]:
            raise ValueError(f"Mismatched configuration: {key}")
    summary = {label: summarize(run) for label, run in runs.items()}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    lines = [
        "# Three-step codegolf latency comparison",
        "",
        "Four clients per run; 4 prompts × 8 completions per update. "
        "Both runs start from the base model with the same configuration. "
        "Outputs are stochastic.",
        "",
        "Sampling measures the client request through receipt of all eight completions, "
        "including transport, queueing, retries, and inference. Judging is timed separately. "
        "Rollout batch time includes all four groups and judging. Learner rollout wait "
        "includes local rollout persistence. Step components exclude checkpoint/evaluation "
        "and metrics commit overhead.",
        "",
        "| Metric (mean) | Before | After |",
        "|---|---:|---:|",
    ]
    for key in summary["before"]:
        a, b = summary["before"].get(key), summary["after"].get(key)
        lines.append(
            f"| {key} | {a:.2f} | {b:.2f} |"
            if a is not None and b is not None
            else f"| {key} | {a} | {b} |"
        )
    lines.extend(
        [
            "",
            "Step 1 includes trainer warmup. Step 3 takes the final checkpoint and evaluation; "
            "those costs are outside the step-components metric. The raw per-client "
            "measurements are in before.json and after.json.",
        ]
    )
    (args.output / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
