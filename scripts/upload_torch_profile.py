# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "wandb",
# ]
# ///

"""Upload a torch.profiler trace to a W&B run's Files tab as profiler/<trace>."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time

import wandb

FILES_DIR = "profiler"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        required=True,
        help="Local directory containing the trace files written by the trainer",
    )
    parser.add_argument(
        "--trace",
        default="rank0.trace.json.gz",
        help="Trace file inside --dir to upload",
    )
    parser.add_argument("--entity", default=None, help="W&B entity (e.g. modal-labs)")
    parser.add_argument(
        "--project", default="miles-lora-longcontext", help="W&B project"
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Attach the trace to an existing run (resume=allow)",
    )
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
    src = os.path.join(os.path.abspath(args.dir), args.trace)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        staged = os.path.join(tmp, FILES_DIR, args.trace)
        os.makedirs(os.path.dirname(staged))
        shutil.copyfile(src, staged)
        run.save(staged, base_path=tmp, policy="now")
        run.finish()
    print(
        f"uploaded to {run.url}/files/{FILES_DIR}/{args.trace} "
        f"in {time.perf_counter() - started:.1f}s"
    )


if __name__ == "__main__":
    main()
