"""Aggregate benchmark evidence and render a comparison without mixing boundaries."""

import json
import math
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "scripts/results/lora-admission"
ARTIFACTS = ROOT / "docs/assets/lora-inference"


def distribution(values):
    values = sorted(values)
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "p50_s": statistics.median(values),
        "p95_s": values[math.ceil(0.95 * len(values)) - 1],
        "mean_s": statistics.mean(values),
        "max_s": values[-1],
    }


def summarize():
    def read(name):
        return json.loads((RESULTS / name).read_text())

    benchmark = read("benchmark-v3.json")
    summary = {
        "date": "2026-09-18",
        "config": benchmark["config"],
        "notes": [
            "Sidecar comparison uses idle-first eviction and guarded reload in BOTH arms; baseline callback retains its global lock.",
            "TTFT is measured at the sidecar's SGLang transport, not the public client.",
            "Connection benchmark uses a fixed CPU response through Flash, not a GPU.",
            "Eviction is one constructed paired stress case, not a latency distribution.",
        ],
        "admission": {},
        "connections": {},
        "idle_eviction": read("idle-eviction.json"),
        "prefix_probe": read("prefix-cache.json"),
    }
    # Source run identifiers are not needed to interpret aggregate measurements.
    summary["config"].pop("source_run", None)
    for phase in benchmark["phases"]:
        rows = list(phase["records"].values())
        main = [r for r in rows if not r.get("revisit")]
        good = [
            r
            for r in main
            if r["status"] == 200
            and r.get("meta", {}).get("completion_tokens") == 64
            and "ttft_s" in r
        ]
        revisit = [r for r in rows if r.get("revisit")]
        item = {
            "requests": len(main),
            "completed": len(good),
            "failures": dict(
                Counter(
                    str(r.get("meta", {}).get("finish_reason", r.get("error")))
                    for r in main
                    if r not in good
                )
            ),
            "revisits": len(revisit),
            "revisits_completed": sum(
                r.get("meta", {}).get("completion_tokens") == 64 for r in revisit
            ),
            "revisit_ttft": distribution(
                r["first_token"] - r["start"] for r in revisit if "first_token" in r
            ),
            "wall_s": phase["wall_s"],
            "max_cpu_adapters": max(phase["cpu_residency"]),
            "local_retries": sum(
                max(0, r.get("generate_attempts", 1) - 1) for r in main
            ),
            "refresh": distribution(phase["refresh_s"]),
            "validation": distribution(phase["resolve_s"]),
            "explicit_load": distribution(x["seconds"] for x in phase["loads"]),
        }
        for repeat, label in ((False, "first_use"), (True, "warm")):
            selected = [r for r in good if r["repeat"] == repeat]
            values = {
                key: distribution(r[key] for r in selected)
                for key in ("admission_s", "ttft_s", "engine_ttft_s", "e2e_s")
            }
            values["cached_tokens"] = sum(
                r["meta"].get("cached_tokens", 0) for r in selected
            )
            values["prompt_tokens"] = sum(r["meta"]["prompt_tokens"] for r in selected)
            values["token_cache_hit_fraction"] = (
                values["cached_tokens"] / values["prompt_tokens"]
                if values["prompt_tokens"]
                else None
            )
            item[label] = values
        summary["admission"][phase["label"]] = item
    connections = read("connections.json")
    for label in sorted({r["phase"] for r in connections} - {"warmup"}):
        rows = [r for r in connections if r["phase"] == label]
        summary["connections"][label] = {
            **distribution(r["elapsed_s"] for r in rows),
            "new_tcp_connections": sum(r["connections"] for r in rows),
            "physical_attempts": sum(r["attempts"] for r in rows),
        }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "measurements.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def render(summary):
    phases = summary["admission"]
    if set(phases) != {"baseline-a", "baseline-b", "optimized-a", "optimized-b"}:
        raise ValueError("All four comparison phases must finish before plotting")
    if any(
        phase["completed"] != phase["requests"]
        or phase["revisits_completed"] != phase["revisits"]
        for phase in phases.values()
    ):
        raise ValueError("Incomplete generation invalidates the latency comparison")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    colors = ["#8796ad", "#299b82"]

    def chart(ax, title, values, labels, ylabel="Seconds", logarithmic=False):
        bars = ax.bar(labels, values, color=colors, width=0.5)
        ax.set_title(title, fontsize=11, pad=14)
        ax.set_ylabel(ylabel)
        if logarithmic:
            ax.set_yscale("log")
            ax.set_ylim(0.1, 160)
        else:
            ax.set_ylim(0, max(values) * 1.25)
        for bar, value in zip(bars, values, strict=True):
            ax.annotate(
                f"{value:.3f} s",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=10,
            )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)

    connections = summary["connections"]
    chart(
        axes[0, 0],
        "Flash transport: median eight-request call\nFixed CPU response; excludes GPU generation",
        [
            statistics.mean(
                v["p50_s"] for k, v in connections.items() if k.startswith(prefix)
            )
            for prefix in ("new-client", "pooled")
        ],
        ["Fresh client", "Pooled client"],
    )
    for axis, group, title in (
        (axes[0, 1], "first_use", "GPU TTFT after publication"),
        (axes[1, 0], "warm", "GPU TTFT on warm repeats"),
    ):
        chart(
            axis,
            title + "\nMean of phase medians; excludes Flash gateway",
            [
                statistics.mean(
                    v[group]["ttft_s"]["p50_s"]
                    for k, v in summary["admission"].items()
                    if k.startswith(prefix)
                )
                for prefix in ("baseline", "optimized")
            ],
            ["Before", "After"],
        )
    chart(
        axes[1, 1],
        "GPU TTFT with a busy CPU-eviction victim\nOne constructed stress case; logarithmic axis",
        [v["admission_to_first_token_s"] for v in summary["idle_eviction"]],
        ["Strict LRU", "Idle-first LRU"],
        logarithmic=True,
    )
    fig.suptitle(
        "LoRA inference latency — separate measured components", fontsize=16, y=1.01
    )
    fig.text(
        0.5,
        -0.015,
        "Qwen3.5-9B · H200 · unchanged 64 CPU adapters / 4 GPU slots / 32 running / 8 queued\nThese measurements have different boundaries and must not be added together.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(pad=2)
    fig.savefig(ARTIFACTS / "latency.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    data = summarize()
    render(data)
    print(ARTIFACTS)
