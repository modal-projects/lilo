"""Time a scoped trainer's cold start and first training steps.

Runs the engine through ``lilo.run`` on a fresh single-use container and records
how long the trainer takes to become ready, how long the first (compile-heavy)
training steps take, and how the kernel-cache Volume changed. Run once from
``main`` (baseline), once with an empty ``lilo-kernel-cache`` Volume (cold), and
once more against the populated Volume (warm).

    MODAL_ENVIRONMENT=<env> python scripts/kernel_cache_bench.py --label warm --steps 5

The trainer's stdout is streamed by ``lilo.run``; pipe through a timestamping
filter to correlate trainer-side log lines with the client-side marks.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import modal
import tinker
from tinker import types

import lilo
from lilo.engines import qwen3_5_4b_full_64k

KERNEL_CACHE_VOLUME = "lilo-kernel-cache"


def volume_stats() -> dict[str, int] | None:
    try:
        volume = modal.Volume.from_name(KERNEL_CACHE_VOLUME)
        entries = [
            entry
            for entry in volume.iterdir("/", recursive=True)
            if entry.type == modal.volume.FileEntryType.FILE
        ]
    except modal.exception.NotFoundError:
        return None
    return {"files": len(entries), "bytes": sum(entry.size for entry in entries)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    engine = qwen3_5_4b_full_64k()
    marks: dict[str, float] = {}
    steps: list[dict[str, float]] = []
    before = volume_stats()

    def mark(name: str) -> None:
        marks[name] = time.time()
        print(f"[bench] {name} t+{marks[name] - marks['start']:.1f}s", flush=True)

    marks["start"] = time.time()
    mark("start")
    with (
        modal.enable_output(),
        lilo.run(
            engine=engine,
            warm=True,
            name="lilo-kcbench",
            latest=lilo.Pool(min_containers=0, max_containers=1),
        ) as (url, api_key),
    ):
        mark("trainer_running")
        service = tinker.ServiceClient(base_url=url, api_key=api_key)
        trainer = lilo.create_full_training_client(service, engine.model)
        mark("training_client_created")
        tokenizer = trainer.get_tokenizer()
        tokens = tokenizer.encode(
            "The capital of France is Paris. The capital of Italy is Rome. " * 8
        )
        datum = types.Datum(
            model_input=types.ModelInput.from_ints(tokens[:-1]),
            loss_fn_inputs={
                "target_tokens": tokens[1:],
                "weights": [1.0] * (len(tokens) - 1),
            },
        )
        for index in range(args.steps):
            started = time.time()
            trainer.forward_backward([datum], "cross_entropy").result(timeout=3600)
            backward_done = time.time()
            trainer.optim_step(types.AdamParams(learning_rate=1e-6)).result(
                timeout=3600
            )
            finished = time.time()
            steps.append(
                {
                    "step": index + 1,
                    "forward_backward_s": backward_done - started,
                    "optim_step_s": finished - backward_done,
                }
            )
            print(f"[bench] step {index + 1}: {json.dumps(steps[-1])}", flush=True)
        mark("steps_done")
    mark("run_closed")

    result = {
        "label": args.label,
        "engine": engine.name,
        "trainer_gpu": engine.trainer_gpu,
        "seconds": {
            "start_to_trainer_running": marks["trainer_running"] - marks["start"],
            "trainer_running_to_client": marks["training_client_created"]
            - marks["trainer_running"],
            "steps_total": marks["steps_done"] - marks["training_client_created"],
        },
        "steps": steps,
        "volume_before": before,
        "volume_after": volume_stats(),
    }
    print(json.dumps(result, indent=2), flush=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    sys.exit(main())
