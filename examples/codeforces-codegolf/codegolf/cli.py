from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal
from modal.volume import FileEntryType

from codegolf.config import (
    APP_NAME,
    DEFAULT_RUN,
    DEFAULT_STEPS,
    DEFAULT_VARIANT,
    VARIANTS,
    VOLUME_NAME,
    config_for,
)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["config", "launch", "status", "fetch"])
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--variant", choices=VARIANTS, default=DEFAULT_VARIANT)
    parser.add_argument(
        "--rollouts",
        action="store_true",
        help="Fetch raw rollouts too, for entropy and diversity diagnostics",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "config":
        print(
            json.dumps(
                dataclasses.asdict(config_for(args.variant, args.steps)), indent=2
            )
        )
        return
    if not os.environ.get("MODAL_ENVIRONMENT"):
        parser.error("Set MODAL_ENVIRONMENT to the deployment environment")
    if not args.run or "/" in args.run or args.run in {".", ".."}:
        parser.error("Invalid run name")
    handle = Path("artifacts/handles") / f"{args.run}-handle.json"
    legacy = Path(f"{args.run}-handle.json")
    if not handle.exists() and legacy.exists():
        handle = legacy
    handle.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "launch":
        if handle.exists():
            previous_handle = json.loads(handle.read_text())
            previous = modal.FunctionCall.from_id(previous_handle["call_id"])
            try:
                result = previous.get(timeout=0)
            except TimeoutError:
                print(
                    json.dumps({"state": "running", **json.loads(handle.read_text())})
                )
                return
            except Exception as exc:
                # Require an explicit, inspectable decision after a terminal exception;
                # unknown RPC errors must never cause a duplicate launch.
                raise RuntimeError(
                    "Inspect the previous call before replacing its handle"
                ) from exc
            if not isinstance(result, dict) or not isinstance(result.get("step"), int):
                raise RuntimeError("Cannot verify the previous call's completed step")
            if result["step"] >= args.steps:
                print(json.dumps({"state": "completed", "result": result}))
                return
            history = (
                Path("artifacts/handles/history")
                / f"{args.run}-{previous_handle['call_id']}.json"
            )
            history.parent.mkdir(parents=True, exist_ok=True)
            history.write_text(json.dumps(previous_handle, indent=2))
        call = modal.Function.from_name(APP_NAME, "run").spawn(
            args.run, args.steps, args.variant
        )
        value = {
            "call_id": call.object_id,
            "run": args.run,
            "steps": args.steps,
            "variant": args.variant,
        }
        handle.write_text(json.dumps(value, indent=2))
        print(json.dumps(value))
        return
    volume = modal.Volume.from_name(VOLUME_NAME)
    if args.command == "fetch":
        root = Path("runs") / args.run
        # Avoid streaming every raw rollout just to download the small report files.
        entries = volume.listdir(args.run)
        for directory in list(entries):
            if directory.type != FileEntryType.DIRECTORY:
                continue
            if Path(directory.path).name == "rollouts":
                if args.rollouts:
                    groups = volume.listdir(directory.path)
                    # Listing one group at a time avoids one large recursive stream.
                    for group in groups:
                        if group.type == FileEntryType.DIRECTORY:
                            entries.extend(volume.listdir(group.path))
            else:
                entries.extend(volume.listdir(directory.path))

        def download(entry):
            relative = Path(entry.path).relative_to(args.run)
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(5):
                try:
                    data = b"".join(volume.read_file(entry.path))
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(2**attempt)
            target.write_bytes(data)
            return target

        files = [
            entry
            for entry in entries
            if entry.type == FileEntryType.FILE
            and entry.path.endswith(".json")
            and (
                args.rollouts
                or Path(entry.path).relative_to(args.run).parts[0] != "rollouts"
            )
        ]
        with ThreadPoolExecutor(max_workers=4 if args.rollouts else 8) as pool:
            fetched = set(pool.map(download, files))
        # A recovery can remove remote metrics. Do not leave old local copies
        # in the curve when refreshing a previously downloaded run.
        for folder in ("metrics", "eval"):
            for target in (root / folder).glob("*.json"):
                if target not in fetched:
                    target.unlink()
        # Extension retires the previous completion marker remotely.
        completed = root / "complete.json"
        if completed.exists() and completed not in fetched:
            completed.unlink()
        print(root)
        return
    result = {}
    if handle.exists():
        call = modal.FunctionCall.from_id(json.loads(handle.read_text())["call_id"])
        try:
            result["call_result"] = call.get(timeout=0)
        except TimeoutError:
            result["call_state"] = "running"
    for filename in ["checkpoint.json", "live.json", "complete.json"]:
        try:
            result[filename] = json.loads(
                b"".join(volume.read_file(f"{args.run}/{filename}"))
            )
        except (modal.exception.NotFoundError, FileNotFoundError):
            pass
    try:
        entries = sorted(volume.listdir(f"{args.run}/metrics"), key=lambda e: e.path)
        if entries:
            result["latest_metric"] = json.loads(
                b"".join(volume.read_file(entries[-1].path))
            )
    except (modal.exception.NotFoundError, FileNotFoundError):
        pass
    try:
        events = sorted(volume.listdir(f"{args.run}/events"), key=lambda e: e.path)
        if events:
            result["latest_event"] = json.loads(
                b"".join(volume.read_file(events[-1].path))
            )
    except (modal.exception.NotFoundError, FileNotFoundError):
        pass
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
