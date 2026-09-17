"""Snapshot and plot codegolf metrics without joining different trainer attempts."""

from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent / "results/multilora-codegolf"


def snapshot(run, out):
    import modal

    volume = modal.Volume.from_name("lilo-multilora-codegolf")

    def read(path):
        return json.loads(b"".join(volume.read_file(path)))

    result = {
        "run": run,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "clients": [],
    }
    with ThreadPoolExecutor(max_workers=12) as pool:
        for client in range(4):
            base = f"{run}/{run}-client-{client}"
            data = {"client": client, "live": read(base + "/live.json")}
            for folder in ("events", "metrics", "eval"):
                paths = sorted(
                    e.path
                    for e in volume.listdir(base + "/" + folder)
                    if e.path.endswith(".json")
                )
                data[folder] = list(pool.map(read, paths))
            result["clients"].append(data)
    # Log retains overwritten metrics from previous attempts. JSON files carry
    # publication timings that are written after the STEP log line.
    metrics = {}
    for line in (out / "supervisor.log").read_text(errors="replace").splitlines():
        if "INFO STEP " not in line:
            continue
        try:
            row = json.loads(line.split("INFO STEP ", 1)[1])
            row["logged_at"] = line[:23]
            metrics[row["model_id"], row["step"]] = row
        except (ValueError, KeyError):
            continue
    for data in result["clients"]:
        for row in data["metrics"]:
            metrics.setdefault((row["model_id"], row["step"]), {}).update(row)
        models = [e["model_id"] for e in data["events"] if e["kind"] == "model_bound"]
        models = list(dict.fromkeys(models))
        data["attempts"] = [
            {
                "model_id": mid,
                "rows": sorted(
                    (r for (m, _), r in metrics.items() if m == mid),
                    key=lambda r: r["step"],
                ),
            }
            for mid in models
        ]
    (out / "plot-snapshot.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def smooth(y):
    return [statistics.fmean(y[max(0, j - 9) : j + 1]) for j in range(len(y))]


PANELS = [
    ("reward", "Reward"),
    ("pass_rate", "Pass rate"),
    ("seconds", "Step time · seconds (log scale)"),
    ("completion_tokens", "Mean generated tokens (log scale)"),
]


def draw(ax, data, key, title):
    current = data["live"]["model_id"]
    color = plt.get_cmap("tab10")(data["client"])
    for attempt in data["attempts"]:
        rows = attempt["rows"]
        if not rows:
            continue
        is_current = attempt["model_id"] == current
        x, y = [r["step"] for r in rows], [r[key] for r in rows]
        if is_current:
            ax.plot(x, y, color=color, alpha=0.25, lw=1)
            ax.plot(
                x,
                smooth(y),
                color=color,
                lw=1.8,
                label="Current attempt · 10-step mean",
            )
        else:
            ax.plot(x, y, color="0.6", alpha=0.25, lw=0.8)
            ax.plot(x, smooth(y), color="0.55", alpha=0.7, lw=1, ls="--")
    if key in ("reward", "pass_rate"):
        es = data["eval"]
        ax.plot(
            [e["step"] for e in es],
            [e[key] for e in es],
            "o--",
            ms=4,
            color="black",
            lw=1,
            label="Latest saved held-out eval",
        )
    if key in ("seconds", "completion_tokens"):
        ax.set_yscale("log")
    if key == "pass_rate":
        ax.set_ylim(0, 1)
    if data["live"]["step"]:
        ax.axvline(data["live"]["step"], color="tab:red", ls=":", lw=1)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Trainer step")
    ax.grid(alpha=0.18)


FOOT = (
    "Faint: individual updates · Solid: trailing 10 updates · Gray dashed: earlier attempts (separate histories)\n"
    "Black: latest saved evaluation at each step · Red dotted: restored checkpoint · Step time excludes publication/evaluation; includes rollout waiting\n"
    "Only completed updates are plotted: ongoing waits are not yet reflected. Saved evaluations can precede the current attempt."
)


def render(data, out):
    fig, axs = plt.subplots(4, 4, figsize=(19, 13))
    for i, c in enumerate(data["clients"]):
        current = next(
            a for a in c["attempts"] if a["model_id"] == c["live"]["model_id"]
        )
        step = max((r["step"] for r in current["rows"]), default=c["live"]["step"])
        for j, (key, title) in enumerate(PANELS):
            draw(axs[i, j], c, key, f"Client {i} · {title}")
        axs[i, 0].set_ylabel(f"Current attempt: step {step}")
    axs[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle(
        f"Four-client multi-LoRA codegolf · {data['captured_at'][:19]} UTC", fontsize=15
    )
    fig.text(0.5, 0.012, FOOT, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"client-curves.{ext}", dpi=160)
    plt.close(fig)
    for c in data["clients"]:
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        for ax, (key, title) in zip(axs.flat, PANELS, strict=True):
            draw(ax, c, key, title)
        axs[0, 0].legend(fontsize=8)
        fig.suptitle(
            f"Codegolf multi-LoRA · client {c['client']} · current and previous attempts"
        )
        fig.text(0.5, 0.01, FOOT, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.09, 1, 0.95))
        for ext in ("png", "pdf"):
            fig.savefig(out / f"client-{c['client']}-curves.{ext}", dpi=160)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="tailrl-hero-v1")
    p.add_argument("--offline", action="store_true")
    args = p.parse_args()
    out = ROOT / args.run
    data = (
        json.loads((out / "plot-snapshot.json").read_text())
        if args.offline
        else snapshot(args.run, out)
    )
    render(data, out)
    print(out / "client-curves.png")


if __name__ == "__main__":
    main()
