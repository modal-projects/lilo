# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.11.1", "numpy>=2,<3"]
# ///
"""Render the checked-in LoRA validation snapshots without network access."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parent
COLORS = ["#2878b5", "#e27a32", "#269b79", "#8962b2"]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.labelcolor": "#344054",
        "text.color": "#172b4d",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def load(name):
    return json.loads((ROOT / name).read_text())


def smooth(values, window=20):
    values = np.array([np.nan if v is None else v for v in values], dtype=float)
    return [
        np.nanmean(values[max(0, i - window + 1) : i + 1]) for i in range(len(values))
    ]


def finish(fig, axes, title, subtitle, filename):
    fig.suptitle(title + "\n" + subtitle, fontsize=15, fontweight="medium")
    for ax in axes.flat:
        ax.grid(alpha=0.16)
        ax.set_axisbelow(True)
        ax.set_xlabel("Optimizer update")
    fig.savefig(ROOT / filename, dpi=170)
    plt.close(fig)


def parity():
    data = load("deterministic-parity.json")
    assert data["verification"]["passed"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    x = np.arange(1, 31)
    for row, (key, name) in enumerate((("gsm8k", "GSM8K"), ("dapo", "DAPO Math"))):
        d = data["datasets"][key]
        left, right = axes[row]
        left.plot(x, d["baseline"], color="#24374a", lw=2.5, label="Isolated baseline")
        for j, c in enumerate(d["clients"]):
            assert c["reward"] == d["baseline"]
            left.plot(
                x,
                c["reward"],
                color=COLORS[j],
                ls=("--", ":", "-.")[j],
                marker=("o", "s", "^")[j],
                markevery=(j, 3),
                mfc="none",
                ms=6,
                lw=1.2,
                label=f"Shared client {c['client']}",
            )
        left.set(
            title=f"{name}: matching reward trajectories",
            ylabel="Training reward",
            ylim=(-0.03, 1.03),
        )
        left.legend(fontsize=9, loc="best")
        for field, label, color, marker in (
            ("rollout_max_abs", "Rollout logprobs", COLORS[0], "o"),
            ("trainer_max_abs", "Trainer logprobs", COLORS[1], "x"),
        ):
            assert max(d[field]) == 0
            right.plot(
                x, d[field], color=color, marker=marker, mfc="none", ms=4, label=label
            )
        right.set(
            title=f"{name}: shared versus isolated",
            ylabel="Max absolute difference (nats)",
            ylim=(-1e-8, 1e-8),
            yticks=[0],
        )
        right.text(
            0.5,
            0.65,
            "Exact equality across all 30 updates",
            transform=right.transAxes,
            ha="center",
            fontsize=12,
        )
        right.legend(fontsize=9)
    finish(
        fig,
        axes,
        "Qwen3.5-9B-Base · six-client numerical parity",
        "30 updates · 8 prompts × 8 samples per client · experimental deterministic FA3",
        "qwen3-5-9b-parity.png",
    )


def async_math():
    data = load("async-math.json")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for row, dataset in enumerate(("GSM8K", "DAPO-Math")):
        clients = [c for c in data["clients"] if c["dataset"] == dataset]
        for j, c in enumerate(clients):
            rows = c["steps"]
            assert len(rows) == 30
            x = [s["step"] + 1 for s in rows]
            for col, key in enumerate(("reward_mean", "step_seconds")):
                y = [s[key] for s in rows]
                axes[row, col].plot(x, y, color=COLORS[j], alpha=0.2, lw=0.8)
                axes[row, col].plot(
                    x,
                    smooth(y, 5),
                    color=COLORS[j],
                    lw=2,
                    label=f"Client {c['client']}",
                )
        axes[row, 0].set(
            title=f"{dataset}: independent training",
            ylabel="Recorded training reward",
            ylim=(-0.02, 1.02),
        )
        axes[row, 1].set(
            title=f"{dataset}: publication-to-publication time",
            ylabel="Seconds",
            yscale="log",
        )
        for ax in axes[row]:
            ax.legend(fontsize=9)
    finish(
        fig,
        axes,
        "Qwen3.5-9B-Base · six asynchronous LoRA clients",
        "Shared 4×H100 trainer · 30 updates each · bold: trailing 5-update mean; faint: raw",
        "qwen3-5-9b-async-math.png",
    )


def codegolf():
    data = load("codegolf.json")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    timing, timing_axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for c in data["clients"]:
        rows, evaluations = c["metrics"], c["eval"]
        assert [r["step"] for r in rows] == list(range(1, 501))
        x = [r["step"] for r in rows]
        color, label = COLORS[c["client"]], f"Client {c['client']}"
        for ax, key in ((axes[0, 0], "reward"), (axes[1, 0], "pass_rate")):
            y = [r[key] for r in rows]
            ax.plot(x, y, color=color, alpha=0.09, lw=0.6)
            ax.plot(x, smooth(y), color=color, lw=2, label=label)
        for ax, key in ((axes[0, 1], "pass_rate"), (axes[1, 1], "passing_bytes")):
            ax.plot(
                [r["step"] for r in evaluations],
                [r[key] for r in evaluations],
                color=color,
                lw=1.8,
                marker="o",
                ms=3,
                label=label,
            )
        for ax, key in (
            (timing_axes[0, 0], "update_seconds"),
            (timing_axes[0, 1], "publish_seconds"),
            (timing_axes[1, 0], "rollout_wait_seconds"),
        ):
            y = [r["pipeline"][key] for r in rows]
            assert all(v is not None and v >= 0 for v in y)
            ax.plot(x, y, color=color, alpha=0.10, lw=0.6)
            ax.plot(x, smooth(y), color=color, lw=2, label=label)
        timing_axes[1, 1].plot(
            x,
            smooth([r["completion_tokens"] for r in rows]),
            color=color,
            lw=2,
            label=label,
        )
    axes[0, 0].set(title="Training reward", ylabel="Reward")
    axes[1, 0].set(title="Training pass rate", ylabel="Passing samples", ylim=(0, 1))
    axes[0, 1].set(
        title="Evaluation pass rate · fixed 16 problems",
        ylabel="Passing samples",
        ylim=(0, 1),
    )
    axes[1, 1].set(
        title="Evaluation program size · passing samples only",
        ylabel="Mean bytes",
        ylim=(0, None),
    )
    for ax in (axes[0, 1], axes[1, 0]):
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    titles = (
        "Training update RPC · includes queueing",
        "Weight publication RPC",
        "Wait for next rollout batch · includes persistence",
        "Generated tokens per consumed training sample",
    )
    for ax, title in zip(timing_axes.flat, titles):
        ax.set_title(title)
        ax.set_ylabel("Tokens" if ax is timing_axes[1, 1] else "Seconds")
        ax.set_ylim(bottom=0)
    for ax in (*axes.flat, *timing_axes.flat):
        ax.legend(fontsize=9, ncol=2)
    finish(
        fig,
        axes,
        "Codeforces codegolf · four LoRA clients × 500 updates",
        "Qwen3.5-9B · TailRL · training: trailing 20-update mean + faint raw values; evaluation: unsmoothed",
        "qwen3-5-9b-codegolf-learning.png",
    )
    finish(
        timing,
        timing_axes,
        "Codeforces codegolf · client-observed training operations",
        "Shared 8×H200 trainer · trailing 20-update means · timings are not GPU execution time",
        "qwen3-5-9b-codegolf-timing.png",
    )


if __name__ == "__main__":
    parity()
    async_math()
    codegolf()
