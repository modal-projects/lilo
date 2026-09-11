"""Plot per-adapter progress from a complete or partial multi-LoRA E2E report."""

import argparse
import json
from pathlib import Path


def plot_report(path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads(Path(path).read_text())
    steps = report.get("steps", [])
    if not steps:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    xs = [s["step"] + 1 for s in steps]
    for i, label in enumerate(["GSM8K · rank 16", "DAPO · rank 32"]):
        rows = [s["adapters"][i] for s in steps]
        for ax, values in zip(
            axes.flat,
            [
                [r["reward_mean"] for r in rows],
                [r["nonzero_advantage_groups"] for r in rows],
                [r["optimizer"].get("grad_norm:mean", float("nan")) for r in rows],
                [r["forward"].get("loss:mean", float("nan")) for r in rows],
            ],
        ):
            ax.plot(xs, values, marker="o", label=label)
    for ax, title in zip(
        axes.flat,
        [
            "Rollout reward (training batches)",
            "Groups with nonzero advantages",
            "Gradient norm",
            "Importance-sampling loss",
        ],
    ):
        ax.set(title=title, xlabel="RL step")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0, 0].set_ylim(-0.02, 1.02)
    fig.suptitle(
        f"{report['definition_id']} · {report['settings']['groups']} prompts × {report['settings']['samples']} samples per adapter"
    )
    fig.savefig(Path(path).with_suffix(".png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    plot_report(parser.parse_args().report)
