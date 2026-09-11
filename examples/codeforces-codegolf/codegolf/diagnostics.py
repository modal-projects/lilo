"""Plot rollout uncertainty, diversity, response limits, and optimizer diagnostics."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def diagnostics(root: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        json.loads(p.read_text()) for p in sorted((root / "metrics").glob("*.json"))
    ]
    rows = []
    for metric in metrics:
        step = metric["step"]
        records = [
            json.loads(p.read_text())
            for p in sorted((root / "rollouts" / f"step-{step:04d}").glob("*.json"))
        ]
        samples = [r for g in records for r in g["rows"]]
        if len(samples) != metric["samples"]:
            continue
        logprobs = [lp for r in samples for lp in r["logprobs"]]
        rows.append(
            {
                "step": step,
                "sampled_entropy_nats_per_token": -statistics.fmean(logprobs),
                "sequence_mean_surprisal": statistics.fmean(
                    -statistics.fmean(r["logprobs"]) for r in samples
                ),
                "unique_program_fraction": statistics.fmean(
                    len({r["code"] for r in g["rows"]}) / len(g["rows"])
                    for g in records
                ),
                "informative_group_fraction": metric["informative_groups"]
                / len(records),
                "completion_tokens": metric["completion_tokens"],
                "truncation_fraction": metric["truncated"] / metric["samples"],
                "grad_norm": metric["optimizer"]["grad_norm:mean"],
            }
        )
    if not rows:
        raise ValueError("Download training rollouts before plotting diagnostics")
    evaluation = []
    for path in sorted((root / "eval").glob("*.json")):
        metric = json.loads(path.read_text())
        samples = [
            row
            for source in (root / "rollouts" / f"eval-{metric['step']:04d}").glob(
                "*.json"
            )
            for row in json.loads(source.read_text())["rows"]
        ]
        if len(samples) == metric["samples"]:
            evaluation.append(
                {
                    "step": metric["step"],
                    "sampled_entropy_nats_per_token": -statistics.fmean(
                        lp for row in samples for lp in row["logprobs"]
                    ),
                    "sequence_mean_surprisal": statistics.fmean(
                        -statistics.fmean(row["logprobs"]) for row in samples
                    ),
                }
            )
    x = [r["step"] for r in rows]

    def draw(ax, key, label, color):
        y = [r[key] for r in rows]
        ax.plot(x, y, color=color, alpha=0.25)
        smooth = [statistics.fmean(y[max(0, i - 9) : i + 1]) for i in range(len(y))]
        ax.plot(x, smooth, color=color, label=label)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    ax = axes[0, 0]
    draw(ax, "sampled_entropy_nats_per_token", "Token-weighted estimate", "tab:blue")
    draw(ax, "sequence_mean_surprisal", "Equal weight per response", "tab:orange")
    if evaluation:
        ax.plot(
            [r["step"] for r in evaluation],
            [r["sampled_entropy_nats_per_token"] for r in evaluation],
            "o--",
            color="tab:green",
            label="Held-out, token-weighted",
        )
    ax.set_title("Sampled token uncertainty")
    ax.set_ylabel("Mean −log p(token) · nats")
    ax.legend(fontsize=9)
    ax = axes[0, 1]
    draw(ax, "unique_program_fraction", "Distinct programs / 8 samples", "tab:blue")
    draw(
        ax, "informative_group_fraction", "Groups with differing rewards", "tab:orange"
    )
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Within-problem diversity")
    ax.set_ylabel("Fraction")
    ax.legend(fontsize=9)
    ax = axes[1, 0]
    draw(ax, "completion_tokens", "Completion tokens", "tab:blue")
    ax.set_ylabel("Mean completion tokens", color="tab:blue")
    other = ax.twinx()
    draw(other, "truncation_fraction", "At response limit", "tab:red")
    other.set_ylim(0, 1)
    other.set_ylabel("Fraction truncated", color="tab:red")
    ax.set_title("Response length and truncation")
    ax = axes[1, 1]
    draw(ax, "grad_norm", "Reported gradient norm", "tab:purple")
    ax.set_yscale("log")
    ax.set_ylabel("Gradient norm · log scale")
    ax.set_title("Optimizer diagnostic")
    restored = sorted(
        {
            e["step"]
            for p in (root / "events").glob("*.json")
            if (e := json.loads(p.read_text())).get("kind") == "trainer_restored"
        }
    )
    for ax in axes.flat:
        ax.grid(alpha=0.2)
        for step in restored:
            ax.axvline(step, color=".4", linestyle=":", alpha=0.5)
    for path in (root / "events").glob("*.json"):
        event = json.loads(path.read_text())
        if event.get("kind") == "response_budget_increased":
            for ax in axes.flat:
                ax.axvline(
                    event["after_step"], color="tab:red", linestyle="--", alpha=0.6
                )
            axes[0, 0].plot(
                [],
                [],
                color="tab:red",
                linestyle="--",
                label=f"Limit → {event['max_tokens']:,}",
            )
            axes[0, 0].legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("Trainer step")
    fig.suptitle(f"Qwen3.5-9B GRPO diagnostics · through step {x[-1]}")
    fig.text(
        0.5,
        0.01,
        "Faint: per-step values. Solid: trailing 10-step mean. Dotted: checkpoint restoration.\nEntropy is a Monte Carlo estimate from sampled log probabilities; prompts and response lengths vary.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.97))
    fig.savefig(root / "diagnostics.png", dpi=160)
    (root / "diagnostics.json").write_text(json.dumps(rows, indent=2))
    (root / "diagnostics-eval.json").write_text(json.dumps(evaluation, indent=2))
    print(
        json.dumps(
            {
                "steps": len(rows),
                "last": rows[-1],
                "first10": {
                    k: statistics.fmean(r[k] for r in rows[:10])
                    for k in rows[0]
                    if k != "step"
                },
                "last10": {
                    k: statistics.fmean(r[k] for r in rows[-10:])
                    for k in rows[0]
                    if k != "step"
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    diagnostics(parser.parse_args().root)
