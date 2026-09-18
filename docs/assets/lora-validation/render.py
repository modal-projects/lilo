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
    source, native = load("async-math.json"), load("native-async-math.json")
    assert source["status"] == native["status"] == "passed"
    assert source["dataset_sha256"] == native["dataset_sha256"]
    for key, value in native["matched_configuration"].items():
        assert source["configuration"][key] == value
    systems = (("Lilo + Miles", source, COLORS[0]), ("Native Miles", native, COLORS[1]))
    for _, data, _ in systems:
        assert [c["client"] for c in data["clients"]] == list(range(6))
        for c in data["clients"]:
            assert [s["step"] for s in c["steps"]] == list(range(30))
            assert all(
                s["sequences"] == 64 and s["policy_lag_updates"] <= 2
                for s in c["steps"]
            )
            assert c["pipeline_start_time"] < c["steps"][0]["completion_time"]
            assert all(
                np.isfinite(s[k])
                for s in c["steps"]
                for k in ("reward_mean", "step_seconds", "completion_time")
            )
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for row, dataset in enumerate(("GSM8K", "DAPO-Math")):
        for label, data, color in systems:
            clients = [c for c in data["clients"] if c["dataset"] == dataset]
            assert len(clients) == 3
            for col, key in enumerate(("reward_mean", "step_seconds")):
                values = np.array([[s[key] for s in c["steps"]] for c in clients])
                for y in values:
                    axes[row, col].plot(range(1, 31), y, color=color, alpha=0.2, lw=0.8)
                axes[row, col].plot(
                    range(1, 31),
                    smooth(values.mean(axis=0), 5),
                    color=color,
                    lw=2,
                    label=label,
                )
        axes[row, 0].set(
            title=f"{dataset}: training reward",
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
        "Qwen3.5-9B-Base · six asynchronous clients · Lilo versus native Miles",
        "4×H100 training + 8×H200 rollout · faint: each client; bold: trailing 5-update mean across 3 clients/dataset",
        "qwen3-5-9b-async-math.png",
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    for label, data, color in systems:
        start = min(c["pipeline_start_time"] for c in data["clients"])
        for ax, dataset in zip(axes, ("GSM8K", "DAPO-Math"), strict=True):
            clients = [c for c in data["clients"] if c["dataset"] == dataset]
            for i, client in enumerate(clients):
                minutes = [(s["completion_time"] - start) / 60 for s in client["steps"]]
                ax.plot(
                    minutes,
                    smooth([s["reward_mean"] for s in client["steps"]], 5),
                    color=color,
                    alpha=0.65,
                    lw=1.3,
                    label=label if i == 0 else None,
                )
            ax.set(
                title=dataset,
                xlabel="Minutes since first client pipeline started",
                ylabel="Recorded training reward",
                ylim=(-0.02, 1.02),
            )
            ax.grid(alpha=0.16)
            ax.set_axisbelow(True)
            ax.legend(fontsize=9)
    fig.suptitle(
        "Training reward versus elapsed time · same GPU allocation\n"
        "Each line: one client, trailing 5-update mean; startup and initial warmup excluded",
        fontsize=14,
    )
    fig.savefig(ROOT / "qwen3-5-9b-async-math-walltime.png", dpi=170)
    plt.close(fig)


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


def cost_estimate():
    data = load("dapo-cost-estimate.json")
    rows = {row["key"]: row for row in data["costs"]}
    fig, axes = plt.subplots(
        1, 3, figsize=(14, 6), gridspec_kw={"width_ratios": [1.6, 1, 1]}
    )
    panels = (
        (["training", "sampling", "total"], "Workload cost", "USD"),
        (["per_client"], "Average per client", "USD / client"),
        (
            ["per_million_generated"],
            "Per million generated tokens",
            "USD / million generated tokens",
        ),
    )
    for ax, (keys, title, unit) in zip(axes, panels):
        positions = np.arange(len(keys))
        for offset, field, label, color, hatch in (
            (-0.18, "lilo", "Lilo · averaged GPU cost", COLORS[0], None),
            (
                0.18,
                "tinker_estimate",
                "Tinker · token-price estimate",
                COLORS[1],
                "///",
            ),
        ):
            values = [rows[key][field] for key in keys]
            bars = ax.barh(
                positions + offset,
                values,
                height=0.30,
                color=color,
                label=label,
                hatch=hatch,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.bar_label(
                bars, labels=[f"${v:.2f}" for v in values], padding=6, fontsize=12
            )
        ax.set_yticks(
            positions, [rows[k]["label"] if len(keys) > 1 else "" for k in keys]
        )
        ax.invert_yaxis()
        ax.set_xlim(0, max(rows[k]["tinker_estimate"] for k in keys) * 1.32)
        ax.set_xlabel(unit)
        ax.set_title(
            title
            + ("\nIncludes training" if keys == ["per_million_generated"] else ""),
            fontsize=12,
            pad=14,
        )
        ax.grid(axis="x", alpha=0.16)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        if len(keys) == 1:
            ax.set_ylim(0.7, -0.7)
    fig.suptitle(
        "Optimistic cost estimate · Qwen3.5-9B on DAPO Math\n"
        "12 clients share one 4×H100 trainer + 2–6 H200 inference GPUs",
        fontsize=16,
        y=0.97,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.84),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(left=0.085, right=0.97, top=0.65, bottom=0.28, wspace=0.32)
    fig.text(
        0.5,
        0.09,
        "Equal client workloads · saturated async trainer · concurrent persistence + adapter-aware inference routing\n"
        "Tinker cache assumption: 1 uncached + 7 cached prompt copies per 8-sample group. Matched Tinker run still pending.",
        ha="center",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.025,
        "Supplied rounded estimates; Lilo components sum to $9.09 versus the supplied $9.08 total.",
        ha="center",
        fontsize=9,
        color="#667085",
        parse_math=False,
    )
    fig.savefig(ROOT / "qwen3-5-9b-dapo-cost-estimate.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parity()
    async_math()
    codegolf()
    cost_estimate()
