"""Seed a new run from a committed checkpoint without mixing reward histories."""

import argparse
import dataclasses
import hashlib
import io
import json
import os
import time

import modal

from codegolf.config import VOLUME_NAME, config_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--variant", default="async-v5")
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    if not os.getenv("MODAL_ENVIRONMENT"):
        parser.error("Set MODAL_ENVIRONMENT")
    for name in [args.source, args.destination]:
        if not name or "/" in name or name in {".", ".."}:
            parser.error("Invalid run name")
    if not 0 < args.step < args.steps:
        parser.error("Fork step must precede target")
    volume = modal.Volume.from_name(VOLUME_NAME)

    def read(name):
        return json.loads(b"".join(volume.read_file(f"{args.source}/{name}")))

    try:
        volume.listdir(args.destination)
    except (modal.exception.NotFoundError, FileNotFoundError):
        pass
    else:
        raise ValueError("Destination already exists; refusing to overwrite")
    checkpoint = read("checkpoint.json")
    if checkpoint["step"] != args.step or not checkpoint["path"]:
        raise ValueError("Source checkpoint is not the requested committed step")
    source_spec = read("spec.json")
    dataset = b"".join(volume.read_file("problems.json"))
    if hashlib.sha256(dataset).hexdigest() != source_spec["dataset_sha256"]:
        raise ValueError("Dataset changed since source run")
    config = dataclasses.asdict(config_for(args.variant, args.steps))
    allowed = {
        "reward_bonus",
        "reward_scale",
        "output_token_penalty",
        "output_token_scale",
        "steps",
        "async_rollouts",
        "max_policy_lag",
        "rollout_workers",
        "buffer_batches",
        "prefill_batches",
        "rollout_min_replicas",
        "rollout_max_replicas",
        "judge_concurrency",
    }
    old = {k: v for k, v in source_spec["config"].items() if k not in allowed}
    new = {k: v for k, v in config.items() if k not in allowed}
    if old != new:
        raise ValueError(
            "Only reward/pipeline settings and target may change on this fork"
        )
    documents = {
        "spec.json": {
            "config": config,
            "dataset_sha256": source_spec["dataset_sha256"],
        },
        "checkpoint.json": checkpoint,
        "validated.json": read("validated.json"),
        "split.json": read("split.json"),
        "lineage.json": {
            "source_run": args.source,
            **checkpoint,
            "source_spec": source_spec,
            "created_at": time.time(),
            "variant": args.variant,
        },
    }
    with volume.batch_upload() as batch:
        for name, document in documents.items():
            batch.put_file(
                io.BytesIO(json.dumps(document, indent=2).encode()),
                f"{args.destination}/{name}",
            )
    print(
        json.dumps(
            {"run": args.destination, "checkpoint": checkpoint, "config": config},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
