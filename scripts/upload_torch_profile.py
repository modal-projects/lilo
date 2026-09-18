# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "wandb",
# ]
# ///

"""Upload a torch.profiler trace directory to W&B as an artifact."""

from __future__ import annotations

import argparse
import time

import wandb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        required=True,
        help="Local directory containing *.trace.json.gz / *.key_averages.txt",
    )
    parser.add_argument("--entity", default=None, help="W&B entity (e.g. modal-labs)")
    parser.add_argument(
        "--project", default="miles-lora-longcontext", help="W&B project"
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Attach the artifact to an existing run (resume=allow)",
    )
    parser.add_argument("--name", default="torch-profile", help="Artifact name")
    parser.add_argument("--group", default=None, help="W&B run group")
    args = parser.parse_args()

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        id=args.run_id,
        resume="allow" if args.run_id else None,
        group=args.group,
        job_type="torch-profile-upload",
    )
    artifact = wandb.Artifact(args.name, type="torch-profile")
    artifact.add_dir(args.dir)
    started = time.perf_counter()
    logged = run.log_artifact(artifact)
    logged.wait()
    print(
        f"uploaded artifact: {logged.qualified_name} in {time.perf_counter() - started:.1f}s"
    )
    run.finish()


if __name__ == "__main__":
    main()
