"""Render the recorded replay-validation metrics; run beside metrics.json."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


root = Path(__file__).resolve().parent
data = json.loads((root / "metrics.json").read_text())
colors = {"replay_off": "#c65b3d", "replay_on": "#257a70"}

for name, title, ylabel in (
    (
        "logprob-difference",
        "Router replay lowers sampler–trainer logprob error",
        "Mean absolute logprob difference (nats)",
    ),
    (
        "paired-comparison",
        "Replay improves all 20 paired comparisons",
        "Replayed error / native-routing error",
    ),
    ("reward", "DAPO math: answer reward over 10 updates", "Correct answer fraction"),
):
    fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
    for mode, run in data["runs"].items():
        rows = run["steps"]
        if name == "paired-comparison":
            values = [
                r["paired/replay_on_mean_abs"] / r["paired/replay_off_mean_abs"]
                for r in rows
            ]
            label = mode.replace("_", " ").capitalize() + " run’s samples"
        else:
            key = "reward/mean" if name == "reward" else "logprob_diff/mean_abs"
            values = [r[key] for r in rows]
            label = mode.replace("_", " ").capitalize()
        ax.plot(
            [r["step"] for r in rows],
            values,
            "o-",
            color=colors[mode],
            label=label,
            linewidth=2,
        )
    if name == "paired-comparison":
        ax.axhline(1, color="#777777", linestyle="--", linewidth=1)
        ax.set_ylim(0.55, 1.04)
    elif name == "reward":
        ax.set_ylim(-0.03, 1.03)
    else:
        ax.set_ylim(bottom=0)
    ax.set(title=title, xlabel="Optimizer update", ylabel=ylabel)
    ax.set_xticks(range(1, 11))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(root / f"{name}.png", dpi=180)
    plt.close(fig)
