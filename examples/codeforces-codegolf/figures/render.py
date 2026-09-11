import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path(__file__).resolve().parent
snapshot = json.loads((out / "metrics.json").read_text())
rows = snapshot["training"]
evals = snapshot["evaluation"]
diag = snapshot["diagnostics"]
entropy_evals = snapshot["evaluation_entropy"]
last = snapshot["through_step"]


def plot(ax, data, key, label, color="tab:blue", split=False):
    chunks = (
        [[r for r in data if r["step"] <= 150], [r for r in data if r["step"] > 150]]
        if split
        else [data]
    )
    for i, chunk in enumerate(chunks):
        x = [r["step"] for r in chunk]
        y = [r[key] for r in chunk]
        ax.plot(x, y, color=color, alpha=0.18)
        smooth = [
            statistics.fmean(v for v in y[max(0, j - 9) : j + 1] if v is not None)
            for j in range(len(y))
        ]
        ax.plot(x, smooth, color=color, label=label if i == 0 else None)


def marks(axes):
    for ax in axes.flat:
        ax.axvline(50, color="tab:red", ls="--", alpha=0.6)
        ax.axvline(150, color="tab:purple", ls="--", alpha=0.8)
        ax.axvline(350, color="gray", ls=":", alpha=0.8)
        ax.set_xlim(0, last + 2)
        ax.grid(alpha=0.18)
        ax.set_xlabel("Trainer step")


def finish(fig, axes, title, file):
    marks(axes)
    fig.suptitle(title)
    fig.text(
        0.5,
        0.012,
        "Faint: individual steps · Solid: trailing 10-step mean · Dots: held-out evaluation\nRed: 16K output limit at 50 · Purple: reward change at 150 · Gray: restored checkpoint 350",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.96))
    fig.savefig(out / file, dpi=150)
    plt.close(fig)


fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, key, title in zip(
    axes.flat,
    ["reward", "pass_rate", "passing_bytes", "completion_tokens"],
    [
        "Reward (definition changes at 150)",
        "Correctness",
        "Extracted code length — passing solutions",
        "Full response length — including prose",
    ],
    strict=True,
):
    plot(ax, rows, key, "Training", split=key == "reward")
    # Reward definitions differ: avoid a connecting segment across the boundary.
    chunks = (
        [[e for e in evals if e["step"] < 150], [e for e in evals if e["step"] >= 150]]
        if key == "reward"
        else [evals]
    )
    for i, c in enumerate(chunks):
        ax.plot(
            [e["step"] for e in c],
            [e[key] for e in c],
            "o--",
            color="tab:orange",
            label="Held-out" if i == 0 else None,
            ms=4,
        )
    ax.set_title(title)
    ax.legend(fontsize=8)
    if key == "pass_rate":
        ax.set_ylim(0, 1.05)
finish(
    fig,
    axes,
    f"Codeforces codegolf · full history, steps 0–{last}",
    f"reward-0-{last}.png",
)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
a = axes[0, 0]
plot(a, diag, "sampled_entropy_nats_per_token", "Token weighted")
plot(a, diag, "sequence_mean_surprisal", "Equal weight per response", "tab:orange")
a.plot(
    [e["step"] for e in entropy_evals],
    [e["sampled_entropy_nats_per_token"] for e in entropy_evals],
    "o--",
    color="tab:green",
    ms=4,
    label="Held-out",
)
a.set_title("Sampled entropy estimate (nats/token)")
a.legend(fontsize=8)
a = axes[0, 1]
plot(a, diag, "unique_program_fraction", "Distinct programs / samples")
a.set_title("Within-problem code diversity")
a.set_ylim(0, 1.05)
a = axes[1, 0]
plot(a, diag, "truncation_fraction", "Fraction hitting output limit", "tab:red")
a.set_title("Response truncation")
a.set_ylim(0, 1)
a = axes[1, 1]
plot(a, diag, "grad_norm", "Gradient norm", "tab:purple")
a.set_yscale("log")
a.set_title("Optimizer gradient norm (log scale)")
finish(
    fig,
    axes,
    f"Codeforces codegolf · diagnostics, steps 0–{last}",
    f"diagnostics-0-{last}.png",
)
print("Wrote full-history plots through", last)
