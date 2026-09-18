"""Validate and plot the measured six-update window from each isolated sweep run."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import re

PRICES = {
    "h200_gpu_second": 0.001261,
    "training_million": 1.463,
    "output_million": 1.995,
    "prompt_million": 0.66,
    "cached_prompt_million": 0.132,
}


def summarize(report):
    if report["status"] != "completed":
        raise ValueError("Cannot summarize an incomplete run")
    count = report["config"]["clients"]
    if len(report["clients"]) != count:
        raise ValueError("Missing clients")
    start, end = report["measurement_start"], report["measurement_end"]
    wall = end - start
    if wall <= 0:
        raise ValueError("Nonpositive measurement interval")
    clients = []
    all_rows = []
    for c in report["clients"]:
        if c["status"] != "completed" or [r["step"] for r in c["steps"]] != list(
            range(1, 9)
        ):
            raise ValueError("Incomplete update sequence")
        rows = c["steps"][2:]
        if any(r["rollout_start"] < start or r["completed_at"] > end for r in rows):
            raise ValueError("Measured work crosses the common window")
        if any(
            r["policy_lag"] > 2 or r["optimizer"]["update_successful:mean"] != 1
            for r in rows
        ):
            raise ValueError("Invalid training update")
        training = sum(r["training"]["tokens:sum"] for r in rows)
        output = sum(r["output_tokens"] for r in rows)
        prompt = sum(r["prompt_tokens"] for r in rows)
        if training != prompt + output - len(rows) * 64:
            raise ValueError("Training/sampling token totals do not reconcile")
        elapsed = c["completed_at"] - start
        clients.append(
            dict(
                client=c["index"],
                output_tokens=output,
                prompt_tokens=prompt,
                training_tokens=training,
                output_tps=output / elapsed,
                output_tps_common_window=output / wall,
                mean_step_seconds=statistics.mean(r["step_seconds"] for r in rows),
                elapsed_seconds=elapsed,
            )
        )
        all_rows.extend(rows)
    output = sum(c["output_tokens"] for c in clients)
    prompt = sum(c["prompt_tokens"] for c in clients)
    training = sum(c["training_tokens"] for c in clients)
    trainer_cost = 8 * wall * PRICES["h200_gpu_second"]
    inference_cost = 8 * wall * PRICES["h200_gpu_second"]
    tinker_training = training / 1e6 * PRICES["training_million"]
    tinker_output = output / 1e6 * PRICES["output_million"]
    tinker_prompt = (
        prompt
        / 1e6
        * (PRICES["prompt_million"] / 8 + PRICES["cached_prompt_million"] * 7 / 8)
    )
    tinker_prompt_no_cache = prompt / 1e6 * PRICES["prompt_million"]
    return dict(
        run=report["run"],
        clients=count,
        measured_steps_per_client=6,
        seconds=wall,
        output_tokens=output,
        prompt_tokens=prompt,
        training_tokens=training,
        output_tps=output / wall,
        output_tps_per_client=output / wall / count,
        training_tps=training / wall,
        mean_step_seconds=statistics.mean(c["mean_step_seconds"] for c in clients),
        train_call_seconds=statistics.mean(r["train_seconds"] for r in all_rows),
        publication_seconds=statistics.mean(r["publish_seconds"] for r in all_rows),
        rollout_wait_seconds=statistics.mean(
            r["rollout_wait_seconds"] for r in all_rows
        ),
        mean_output_length=output / (count * 6 * 64),
        truncated_fraction=sum(r["truncated"] for r in all_rows) / (count * 6 * 64),
        cost=dict(
            lilo_training_gpu=trainer_cost,
            lilo_inference_gpu=inference_cost,
            lilo_total_gpu=trainer_cost + inference_cost,
            tinker_training=tinker_training,
            tinker_output=tinker_output,
            tinker_prompt=tinker_prompt,
            tinker_total=tinker_training + tinker_output + tinker_prompt,
            tinker_total_no_cache=tinker_training
            + tinker_output
            + tinker_prompt_no_cache,
            lilo_per_client=(trainer_cost + inference_cost) / count,
            tinker_per_client=(tinker_training + tinker_output + tinker_prompt) / count,
            lilo_per_million_output=(trainer_cost + inference_cost) * 1e6 / output,
            tinker_per_million_output=(tinker_training + tinker_output + tinker_prompt)
            * 1e6
            / output,
        ),
        per_client=clients,
    )


def validate_isolation(runs):
    pools = set()
    apps = set()
    last_finished = None
    for state, report in runs:
        if (
            not state.get("drained")
            or state.get("remaining_tasks")
            or state.get("cleanup_errors")
        ):
            raise ValueError("Run did not drain cleanly")
        if state["app_id"] in apps or state["pool_app"] in pools:
            raise ValueError("Deployment reused across points")
        if last_finished is not None and report["started_at"] <= last_finished:
            raise ValueError("Runs overlap")
        apps.add(state["app_id"])
        pools.add(state["pool_app"])
        last_finished = state["finished_at"]


def validate_resources(path, report, state, lifetimes):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    samples = [r for r in records if "tasks" in r]
    start, end = report["measurement_start"], report["measurement_end"]
    before = [r for r in samples if r["time"] <= start]
    if not before:
        raise ValueError("Resource observations do not cover the measured interval")
    measured = [before[-1]] + [r for r in samples if start < r["time"] < end]
    if end - measured[-1]["time"] > 45:
        raise ValueError("Resource observations end too early")
    if any(b["time"] - a["time"] > 45 for a, b in zip(measured, measured[1:])):
        raise ValueError("Resource observation gap exceeds 45 seconds")
    for r in measured:
        trainer = [
            t for t in r["tasks"] if t["app_id"] == state["app_id"] and t["gpu_count"]
        ]
        inference = [
            t
            for t in r["tasks"]
            if t["app_name"] == state["pool_app"] and t["gpu_count"]
        ]
        if len(trainer) != 1 or trainer[0]["gpu_count"] != 8 or len(inference) != 8:
            raise ValueError("Expected one 8-GPU trainer and eight inference replicas")
        if any(t["gpu_count"] != 1 for t in inference):
            raise ValueError("Inference replicas must each have one GPU")
        if any(
            t["gpu_type"] != "H200" or t["started_at"] <= 0 or t["finished_at"]
            for t in trainer + inference
        ):
            raise ValueError("A measured GPU was not running")
    spanning = [
        t for t in lifetimes if t["started_at"] <= start and t["finished_at"] >= end
    ]
    trainer = [t for t in spanning if t["app_id"] == state["app_id"]]
    inference = [t for t in spanning if t["app_name"] == state["pool_app"]]
    if (
        len(trainer) != 1
        or trainer[0]["gpu_count"] != 8
        or len(inference) != 8
        or any(t["gpu_count"] != 1 for t in inference)
    ):
        raise ValueError("GPU lifetimes do not cover the complete measured interval")
    return dict(
        lifetimes=lifetimes,
        samples=len(measured),
        first_observed=measured[0]["time"],
        last_observed=measured[-1]["time"],
        trainer_gpus=8,
        inference_gpus=8,
        gpu_type="H200",
        observations=measured,
    )


def controller_workload_hash(recorded_hash, current_source):
    # The only permitted driver change was disabling CPU preemption after two
    # aborted 16-client attempts. All workload and resource settings must match.
    legacy = current_source.replace("    nonpreemptible=True,\n", "")
    canonical = hashlib.sha256(legacy.encode()).hexdigest()
    current = hashlib.sha256(current_source.encode()).hexdigest()
    if recorded_hash not in {canonical, current}:
        raise ValueError("Controller changed beyond the CPU preemption setting")
    return canonical


def validate_configuration(runs, directories):
    configs = [
        {k: v for k, v in r["config"].items() if k != "clients"} for _, r in runs
    ]
    if any(c != configs[0] for c in configs):
        raise ValueError("Workload or hardware configuration changed across points")
    sources = []
    root = Path(__file__).resolve().parents[1]
    definition = "src/lilo/providers/modal/definitions/qwen3_5_9b_dapo_sweep.py"
    template = (root / definition).read_text()
    controller = "scripts/dapo_client_sweep_point.py"
    controller_source = (root / controller).read_text()
    for d in directories:
        manifest = json.loads((d / "source-sha256.json").read_text())
        manifest[controller] = controller_workload_hash(
            manifest[controller], controller_source
        )
        expected = re.sub(
            r'SWEEP_ISOLATION_ID = "[^"]*"',
            f'SWEEP_ISOLATION_ID = "{d.name}"',
            template,
        )
        if hashlib.sha256(expected.encode()).hexdigest() != manifest[definition]:
            raise ValueError("Provider settings differ beyond the experiment ID")
        sources.append(
            {
                k: v
                for k, v in manifest.items()
                if (
                    k.startswith("src/lilo/")
                    and not k.endswith("qwen3_5_9b_dapo_sweep.py")
                )
                or k
                in {
                    "scripts/dapo_client_sweep_point.py",
                    "scripts/dapo_sweep_workload.py",
                    "scripts/dapo_sweep_resources.py",
                }
            }
        )
    if any(s != sources[0] for s in sources):
        raise ValueError("Runtime source code changed across points")
    return sources[0]


def validate_placement(report, directory):
    wanted = {c["model_id"] for c in report["clients"]}
    initial, final = report["placement"], report["placement_final"]
    if set(final["resident_model_ids"]) != wanted:
        raise ValueError("Unexpected clients remained on the trainer")
    if any(initial[k] != final[k] for k in ["engine_instance_id", "engine_boot_id"]):
        raise ValueError("Trainer restarted during the run")
    if set(initial["resident_model_ids"]) != wanted:
        recovery = json.loads((directory / "startup-recovery.json").read_text())
        if not (
            recovery["verified_at"] < report["measurement_start"]
            and recovery["unload_completed"]
            and recovery["orphan_had_no_published_sampler"]
            and set(recovery["placement"]["resident_model_ids"]) == wanted
            and set(initial["resident_model_ids"])
            == wanted | {recovery["orphan_model_id"]}
        ):
            raise ValueError("Startup orphan was not removed before measurement")


def render(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    counts = [r["clients"] for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, key, title, unit in [
        (
            axes[0, 0],
            "output_tps",
            "Total useful output throughput",
            "Generated tokens / second",
        ),
        (
            axes[0, 1],
            "output_tps_per_client",
            "Throughput per client (total ÷ clients)",
            "Generated tokens / second / client",
        ),
        (
            axes[1, 0],
            "mean_step_seconds",
            "Average step time per client",
            "Seconds / update",
        ),
    ]:
        individual_key = {
            "output_tps_per_client": "output_tps_common_window",
            "mean_step_seconds": "mean_step_seconds",
        }.get(key)
        if individual_key:
            for j, r in enumerate(rows):
                offsets = np.linspace(-0.12, 0.12, r["clients"]) if r["clients"] > 1 else [0]
                ax.scatter(
                    x[j] + np.array(offsets),
                    [c[individual_key] for c in r["per_client"]],
                    s=16,
                    color="#818c94",
                    alpha=0.55,
                    label="Individual clients" if j == 0 else None,
                )
        ax.plot(
            x, [r[key] for r in rows], marker="o", color="#287e9e", lw=2,
            label="Mean" if individual_key else None,
        )
        if individual_key:
            ax.legend(fontsize=8)
        for a, r in zip(x, rows):
            ax.annotate(
                f"{r[key]:,.0f}",
                (a, r[key]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.set_ylim(bottom=0)
    ax = axes[1, 1]
    ax.bar(
        x - 0.19,
        [r["cost"]["lilo_per_client"] for r in rows],
        width=0.38,
        label="Lilo GPU cost",
        color="#287e9e",
    )
    ax.bar(
        x + 0.19,
        [r["cost"]["tinker_per_client"] for r in rows],
        width=0.38,
        label="Tinker token-price estimate",
        color="#d49145",
    )
    ax.set_title("Cost per client for six measured updates")
    ax.set_ylabel("USD")
    ax.legend(fontsize=9)
    for ax in axes.flat:
        ax.set_xticks(x, counts)
        ax.set_xlabel("Concurrent training clients")
        ax.grid(axis="y", alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "DAPO Math · Qwen3.5-9B-Base · fixed 8×H200 trainer + 8×H200 inference"
        + (f"\nPartial sweep: {len(rows)} of 6 points completed" if len(rows) < 6 else ""),
        fontsize=14,
    )
    fig.text(
        0.5,
        0.014,
        "8 updates/client; first 2 excluded. Async within measured window. No checkpoints/evaluations.\nLilo: GPU-only wall cost. Tinker: same tokens, 7/8 prompt copies assumed cached; not a Tinker run.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(output / "sweep.png", dpi=180)
    fig.savefig(output / "sweep.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    bottom = np.zeros(len(rows))
    for key, label, color in [
        ("rollout_wait_seconds", "Waiting for next sampled batch", "#d49145"),
        ("train_call_seconds", "Training call, including server queue", "#287e9e"),
        ("publication_seconds", "Publishing updated weights", "#72aa7c"),
    ]:
        values = np.array([r[key] for r in rows])
        axes[0].bar(x, values, bottom=bottom, label=label, color=color)
        bottom += values
    axes[0].set_title("Where each client's update time went")
    axes[0].set_ylabel("Mean seconds / update")
    axes[0].legend(fontsize=8)
    axes[1].plot(
        x, [r["mean_output_length"] for r in rows], marker="o", color="#287e9e"
    )
    axes[1].set_title("Output length affects throughput")
    axes[1].set_ylabel("Mean generated tokens / completion")
    axes[1].set_ylim(0, 4300)
    right = axes[1].twinx()
    right.plot(
        x,
        [100 * r["truncated_fraction"] for r in rows],
        marker="s",
        color="#d49145",
        linestyle="--",
    )
    right.set_ylabel("Completions reaching token limit (%)", color="#d49145")
    right.set_ylim(0, 100)
    for ax in axes:
        ax.set_xticks(x, counts)
        ax.set_xlabel("Concurrent training clients")
        ax.grid(axis="y", alpha=0.15)
    fig.suptitle(
        "DAPO scaling diagnostics · client wall times, not GPU utilization", fontsize=13
    )
    fig.tight_layout()
    fig.savefig(output / "diagnostics.png", dpi=180)
    fig.savefig(output / "diagnostics.pdf")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    runs = [
        (
            json.loads((d / "supervisor.json").read_text()),
            json.loads((d / "report.json").read_text()),
        )
        for d in a.runs
    ]
    validate_isolation(runs)
    for (_, report), directory in zip(runs, a.runs):
        validate_placement(report, directory)
    source_hashes = validate_configuration(runs, a.runs)
    resource_checks = [
        validate_resources(
            d / "resources.jsonl",
            r,
            s,
            json.loads((d / "gpu-lifetimes.json").read_text()),
        )
        for (s, r), d in zip(runs, a.runs)
    ]
    rows = sorted([summarize(r) for _, r in runs], key=lambda r: r["clients"])
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "summary.json").write_text(
        json.dumps(
            dict(
                prices=PRICES,
                prices_checked_on="2026-09-18",
                price_sources=[
                    "https://modal.com/pricing",
                    "https://tinker-docs.thinkingmachines.ai/tinker/models/",
                ],
                rows=rows,
            ),
            indent=2,
        )
        + "\n"
    )
    (a.output / "source-sha256.json").write_text(
        json.dumps(source_hashes, indent=2) + "\n"
    )
    (a.output / "source-manifests.json").write_text(
        json.dumps(
            {
                "normalization": "The common source-sha256.json removes only the controller's nonpreemptible=True decorator argument; original per-run hashes follow.",
                "runs": {
                    d.name: json.loads((d / "source-sha256.json").read_text())
                    for d in a.runs
                },
            },
            indent=2,
        )
        + "\n"
    )
    with (a.output / "summary.csv").open("w", newline="") as stream:
        flat_rows = [
            {
                **{k: v for k, v in r.items() if k not in {"cost", "per_client"}},
                **r["cost"],
            }
            for r in rows
        ]
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat_rows)
    for (_, report), directory, resources in zip(runs, a.runs, resource_checks):
        (a.output / f"resources-{report['config']['clients']:02d}.json").write_text(
            json.dumps(resources, indent=2) + "\n"
        )
        # Reports contain metrics, IDs and configuration; no prompts, outputs, auth headers or credentials.
        (a.output / f"clients-{report['config']['clients']:02d}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        recovery = directory / "startup-recovery.json"
        if recovery.exists():
            (
                a.output / f"startup-recovery-{report['config']['clients']:02d}.json"
            ).write_text(recovery.read_text())
        state = json.loads((directory / "supervisor.json").read_text())
        (a.output / f"isolation-{report['config']['clients']:02d}.json").write_text(
            json.dumps({k: v for k, v in state.items() if k != "url"}, indent=2) + "\n"
        )
    render(rows, a.output)
    for r in rows:
        print(
            r["clients"],
            round(r["output_tps"]),
            round(r["output_tps_per_client"]),
            round(r["mean_step_seconds"], 2),
            r["cost"],
        )


if __name__ == "__main__":
    main()
