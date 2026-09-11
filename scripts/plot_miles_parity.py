"""Plot saved trainer/inference parity diagnostics without rerunning a model."""

import argparse
import json
from pathlib import Path


def plot(path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads(path.read_text())
    records = report.get("diagnostic_parity", [])
    latest = report.get("final_parity") or (records[-1] if records else None)
    if latest is None:
        raise ValueError("This report does not contain saved parity arrays")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for i, label in enumerate(["Adapter A · rank 16", "Adapter B · rank 32"]):
        trainer = latest["trainer_logprobs"][i]
        inference = latest["inference_logprobs"][i]
        differences = [a - b for a, b in zip(trainer, inference, strict=True)]
        axes[0].plot(range(1, len(differences) + 1), differences, label=label)
        if records:
            axes[1].plot(
                [r["supervised_updates"] for r in records],
                [r["max_abs_errors"][i] for r in records],
                "o-",
                label=label,
            )
    axes[0].set(
        title="Latest fixed-input parity comparison",
        xlabel="Target-token position",
        ylabel="Trainer − inference logprob (nats)",
    )
    axes[1].set(
        title="Controlled-update parity",
        xlabel="Supervised updates per adapter",
        ylabel="Maximum absolute logprob difference (nats)",
    )
    axes[1].axhline(0.2, color="gray", linestyle=":", label="Original gate: 0.2")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("Miles trainer / SGLang inference · same tokens, published adapter versions")
    output = path.with_suffix(".parity.png")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    plot(parser.parse_args().report)
