"""Extract trainer activity from saved sweep logs, without inventing GPU utilization."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics

TOKEN_BUDGET = 114_688
OPERATIONS = {
    "forward_backward": ("Forward/backward call", "#27898e"),
    "optim_step": ("Optimizer", "#d58a25"),
    "export_slot": ("Snapshot capture", "#8763b4"),
}
STAMP = r"(?P<time>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)"
EVENT = re.compile(
    rf"\[{STAMP} main\].*?ft op=execute phase=(?P<phase>start|end) "
    r"cell=(?P<cell>\S+) fn=(?P<name>\w+)(?P<detail>[^\n]*)"
)
SUBMISSION = re.compile(r'\{"event":\s*"train_start"')
BATCH_SIZE = re.compile(
    rf"\[{STAMP} actor_[^\]]+\].*?Using dynamic global_batch_size=(?P<count>\d+)"
)


def timestamp(value):
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp()


def operation_intervals(log, start, end):
    """Pair embedded server timestamps, never log delivery time or rounded duration."""
    events = sorted(EVENT.finditer(log), key=lambda m: timestamp(m["time"]))
    pending, intervals = {}, []
    for event in events:
        key = (event["cell"], event["name"])
        if event["name"] not in OPERATIONS:
            continue
        when = timestamp(event["time"])
        if event["phase"] == "start":
            if key in pending:
                raise ValueError(f"Duplicate operation start: {key}")
            pending[key] = when
        else:
            if key not in pending:
                raise ValueError(f"Missing operation start: {key}")
            began = pending.pop(key)
            if "ok=true" not in event["detail"] or when <= began:
                raise ValueError(f"Failed or invalid operation: {key}")
            if when > start and began < end:
                intervals.append(
                    dict(
                        operation=key[1],
                        start=max(began, start) - start,
                        end=min(when, end) - start,
                    )
                )
    if pending:
        raise ValueError(f"Unfinished operations: {pending}")
    intervals.sort(key=lambda x: x["start"])
    if any(a["end"] > b["start"] for a, b in zip(intervals, intervals[1:])):
        raise ValueError("Trainer operations overlap; cannot sum their durations")
    return intervals


def extract(report, log):
    start, end = report["measurement_start"], report["measurement_end"]
    intervals = operation_intervals(log, start, end)
    sizes = [
        (timestamp(m["time"]) - start, int(m["count"]))
        for m in BATCH_SIZE.finditer(log)
    ]
    for interval in intervals:
        if interval["operation"] != "forward_backward":
            continue
        counts = {
            count
            for when, count in sizes
            if interval["start"] <= when <= interval["end"]
        }
        if len(counts) != 1:
            raise ValueError(
                f"Missing or ambiguous forward/backward batch size: {counts}"
            )
        interval["sequences"] = counts.pop()
    submitted = defaultdict(list)
    for match in SUBMISSION.finditer(log):
        record, _ = json.JSONDecoder().raw_decode(log[match.start() :])
        submitted[record["model_id"]].append(record["time"])
    batches = []
    for client in report["clients"]:
        times = sorted(submitted[client["model_id"]])
        if len(times) != 8:
            raise ValueError(f"Expected eight training submissions: {client['index']}")
        for row, when in zip(client["steps"], times, strict=True):
            if not row["rollout_end"] <= when <= row["completed_at"]:
                raise ValueError("Submission does not match client update")
            if row["step"] <= 2:
                continue
            if not start <= when <= end:
                raise ValueError("Training submission outside measurement window")
            # Miles reconstructs the last target token: its packing lengths include
            # one more token per sequence than the returned training tokens metric.
            tokens = row["prompt_tokens"] + row["output_tokens"]
            if tokens != row["training"]["tokens:sum"] + 64:
                raise ValueError("Packing and loss-token totals disagree")
            batches.append(
                dict(
                    client=client["index"],
                    step=row["step"],
                    submitted=when - start,
                    tokens=tokens,
                )
            )
    counts = Counter(item["operation"] for item in intervals)
    clients = report["config"]["clients"]
    if counts["optim_step"] != clients * 6 or counts["export_slot"] != clients * 5:
        raise ValueError(f"Missing measured optimizer/snapshot operations: {counts}")
    if sum(x.get("sequences", 0) for x in intervals) != clients * 6 * 64:
        raise ValueError(
            "Forward/backward sequence total does not match measured updates"
        )
    seconds = {
        name: sum(x["end"] - x["start"] for x in intervals if x["operation"] == name)
        for name in OPERATIONS
    }
    wall = end - start
    return dict(
        clients=clients,
        wall_seconds=wall,
        measurement_start=start,
        log_sha256=hashlib.sha256(log.encode()).hexdigest(),
        configured_microbatch_token_budget=TOKEN_BUDGET,
        hardware_token_capacity=None,
        microbatch_fill=None,
        intervals=intervals,
        batches=sorted(batches, key=lambda x: x["submitted"]),
        operation_seconds=seconds,
        operation_counts=dict(counts),
        call_occupancy=sum(seconds.values()) / wall,
        outside_calls_seconds=wall - sum(seconds.values()),
        mean_client_batch_tokens=statistics.mean(x["tokens"] for x in batches),
    )


def window_occupancy(intervals, left, right):
    return sum(
        max(0, min(x["end"], right) - max(x["start"], left)) for x in intervals
    ) / (right - left)


def render(runs, output):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import numpy as np

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(6, 1, figsize=(14, 13), layout="constrained")
    fig.suptitle(
        "Trainer activity over time · same 8×H200 trainer\n"
        "Solid bars = server calls in progress; white = outside those calls",
        fontsize=16,
    )
    for ax, run in zip(axes, runs, strict=True):
        for name, (label, color) in OPERATIONS.items():
            spans = [
                (x["start"] / 60, (x["end"] - x["start"]) / 60)
                for x in run["intervals"]
                if x["operation"] == name
            ]
            ax.broken_barh(spans, (0, 1), facecolors=color, linewidth=0)
        # Full 60-second windows only, centered at each x; do not pad end gaps.
        centers = np.linspace(30, run["wall_seconds"] - 30, 500)
        busy = [window_occupancy(run["intervals"], t - 30, t + 30) for t in centers]
        ax.plot(centers / 60, busy, color="#172333", lw=1.6)
        ax.set(
            xlim=(0, run["wall_seconds"] / 60),
            ylim=(0, 1.08),
            yticks=[0, 0.5, 1],
            yticklabels=["0%", "50%", "100%"],
            xlabel="Minutes after warmup",
            ylabel="Time in calls",
        )
        ax.set_title(
            f"{run['clients']} clients · "
            f"{run['call_occupancy']:.1%} of window inside trainer calls",
            loc="left",
            fontsize=11,
        )
        ax.grid(axis="y", alpha=0.15)
    handles = [Patch(color=color, label=label) for label, color in OPERATIONS.values()]
    handles.append(
        Line2D([0], [0], color="#172333", label="Time in calls over 60 seconds")
    )
    fig.legend(handles=handles, loc="outside lower center", ncol=4)
    fig.savefig(output / "trainer-activity.png", dpi=180)
    fig.savefig(output / "trainer-activity.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), layout="constrained", sharey=True)
    fig.suptitle(
        "Client batch size over time · 64 sequences per update\n"
        "Prompt + answer tokens; plotted when the client submits training",
        fontsize=16,
    )
    maximum = max(x["tokens"] for r in runs for x in r["batches"])
    for ax, run in zip(axes.flat, runs, strict=True):
        ax.scatter(
            [x["submitted"] / 60 for x in run["batches"]],
            [x["tokens"] / 1000 for x in run["batches"]],
            s=15,
            alpha=0.4,
            color="#27898e",
        )
        bins = defaultdict(list)
        for batch in run["batches"]:
            bins[int(batch["submitted"] // 60)].append(batch["tokens"])
        for minute, values in bins.items():
            ax.hlines(
                statistics.mean(values) / 1000,
                minute,
                min(minute + 1, run["wall_seconds"] / 60),
                color="#172333",
                lw=2.5,
            )
        ax.axhline(TOKEN_BUDGET / 1000, color="#c27119", ls="--")
        mean = run["mean_client_batch_tokens"]
        ax.set_title(
            f"{run['clients']} clients · mean {mean:,.0f} tokens "
            f"({mean / TOKEN_BUDGET:.1%} of budget)",
            loc="left",
            fontsize=11,
        )
        ax.set(
            xlim=(0, run["wall_seconds"] / 60),
            ylim=(0, maximum / 1000 * 1.1),
            xlabel="Minutes after warmup",
            ylabel="Tokens per client batch (thousands)",
        )
        ax.grid(alpha=0.15)
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                ls="",
                color="#27898e",
                label="Individual client batch",
            ),
            Line2D(
                [0],
                [0],
                color="#172333",
                lw=2.5,
                label="Mean of batches submitted in each minute",
            ),
            Line2D(
                [0],
                [0],
                color="#c27119",
                ls="--",
                label="114,688-token microbatch budget",
            ),
        ],
        loc="outside lower center",
        ncol=3,
    )
    fig.savefig(output / "trainer-batch-tokens.png", dpi=180)
    fig.savefig(output / "trainer-batch-tokens.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="*", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/assets/dapo-client-scaling")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.directories:
        runs = []
        for directory in args.directories:
            result = extract(
                json.loads((directory / "report.json").read_text()),
                (directory / "launch.log").read_text(),
            )
            runs.append(result)
        runs.sort(key=lambda x: x["clients"])
        (args.output / "trainer-activity.json").write_text(
            json.dumps(runs, indent=2) + "\n"
        )
    else:
        runs = json.loads((args.output / "trainer-activity.json").read_text())
    if [run["clients"] for run in runs] != [1, 2, 4, 8, 16, 32]:
        raise ValueError("Expected all six completed sweep points")
    render(runs, args.output)
    for run in runs:
        print(
            run["clients"],
            round(run["mean_client_batch_tokens"]),
            round(run["call_occupancy"] * 100, 1),
            run["operation_seconds"],
        )


if __name__ == "__main__":
    main()
