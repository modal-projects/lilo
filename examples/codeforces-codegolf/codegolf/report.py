from __future__ import annotations

import argparse
import json
from pathlib import Path


def report(root: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = (
        json.loads((root / "spec.json").read_text())["config"]
        if (root / "spec.json").exists()
        else {}
    )
    estimator = config.get("advantage_estimator", "grpo")
    estimator_label = "TailRL" if estimator == "tailrl" else estimator.upper()
    rows = [
        json.loads(p.read_text()) for p in sorted((root / "metrics").glob("*.json"))
    ]
    evaluation = [
        json.loads(p.read_text()) for p in sorted((root / "eval").glob("*.json"))
    ]
    if not rows:
        raise ValueError("No completed trainer steps yet")
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, key, label in zip(
        axes,
        ["reward", "pass_rate", "passing_bytes"],
        ["Mean reward", "Pass rate", "Bytes (passing only)"],
        strict=True,
    ):
        ax.plot(
            [r["step"] for r in rows],
            [r[key] for r in rows],
            alpha=0.4,
            label="Training groups",
        )
        window = 10
        smooth = [
            sum(v) / len(v)
            if (
                v := [
                    r[key]
                    for r in rows[max(0, i - window + 1) : i + 1]
                    if r[key] is not None
                ]
            )
            else float("nan")
            for i in range(len(rows))
        ]
        ax.plot([r["step"] for r in rows], smooth, label="10-step mean")
        if evaluation:
            ax.plot(
                [r["step"] for r in evaluation],
                [r[key] for r in evaluation],
                "o--",
                label="Fixed held-out problems",
            )
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    restored_steps = sorted(
        {
            event["step"]
            for path in (root / "events").glob("*.json")
            if (event := json.loads(path.read_text())).get("kind") == "trainer_restored"
        }
    )
    for ax in axes:
        for index, step in enumerate(restored_steps):
            ax.axvline(
                step,
                color="0.4",
                linestyle=":",
                alpha=0.6,
                label="Checkpoint restored" if index == 0 else None,
            )
    for path in (root / "events").glob("*.json"):
        event = json.loads(path.read_text())
        if event.get("kind") == "response_budget_increased":
            for ax in axes:
                ax.axvline(
                    event["after_step"], color="tab:red", linestyle="--", alpha=0.6
                )
            axes[0].plot(
                [],
                [],
                color="tab:red",
                linestyle="--",
                label=f"Response limit → {event['max_tokens']:,}",
            )
    axes[0].legend()
    axes[-1].set_xlabel("Trainer step")
    fig.suptitle(f"Qwen3.5-9B · Codeforces codegolf · {estimator_label}")
    fig.tight_layout()
    fig.savefig(root / "reward.png", dpi=160)
    plt.close(fig)
    if any(r.get("eval_samples", 1) > 1 for r in evaluation):
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        for ax, key, metric_label in zip(
            axes,
            ["pass_at_k", "best_of_k"],
            ["Pass@k", "Best-of-k reward"],
            strict=True,
        ):
            budgets = sorted({int(k) for row in evaluation for k in row.get(key, {})})
            for k in budgets:
                points = [r for r in evaluation if str(k) in r.get(key, {})]
                ax.plot(
                    [r["step"] for r in points],
                    [r[key][str(k)] for r in points],
                    "o-",
                    label=f"k={k}",
                )
            ax.set_ylabel(metric_label)
            ax.grid(alpha=0.2)
            ax.legend()
        axes[0].set_ylim(-0.02, 1.02)
        axes[-1].set_xlabel("Trainer step")
        fig.suptitle(
            f"Codeforces codegolf · {estimator_label} · Held-out sampling budgets"
        )
        fig.tight_layout()
        fig.savefig(root / "sampling.png", dpi=160)
        plt.close(fig)
    (root / "metrics.json").write_text(json.dumps(rows, indent=2))
    summary = {
        "steps": rows[-1]["step"],
        "recorded_updates": len(rows),
        "target_steps": config.get("steps"),
        "advantage_estimator": estimator,
        "first": rows[0],
        "last": rows[-1],
        "evaluation": evaluation,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    report(p.parse_args().root)
