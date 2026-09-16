from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(value, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


class Store:
    def __init__(self, root: Path, commit=None):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.commit = commit
        self.lock = asyncio.Lock()

    def read(self, name, default=None):
        path = self.root / name
        return json.loads(path.read_text()) if path.exists() else default

    async def write(self, name, value):
        async with self.lock:
            atomic_json(self.root / name, value)
            if self.commit:
                await self.commit()

    async def event(self, kind, **fields):
        await self.write(
            f"events/{time.time_ns()}-{uuid.uuid4().hex[:6]}.json",
            {"kind": kind, "time": time.time(), **fields},
        )

    async def prepare(self, spec):
        """Allow target extension or a checkpointed response-budget increase."""
        target = spec["config"]["steps"]
        if target <= 0:
            raise ValueError("The training step target must be positive")
        old = self.read("spec.json")
        if old is not None and old != spec:
            previous_target = old["config"]["steps"]
            old_tokens = old["config"].get("max_tokens")
            new_tokens = spec["config"].get("max_tokens")
            if old_tokens != new_tokens:
                expected = {
                    **old,
                    "config": {**old["config"], "max_tokens": new_tokens},
                }
                checkpoint = self.read("checkpoint.json")
                if (
                    expected != spec
                    or not isinstance(old_tokens, int)
                    or not isinstance(new_tokens, int)
                    or not old_tokens < new_tokens <= 65536
                    or not checkpoint
                    or not checkpoint.get("path")
                ):
                    raise ValueError(
                        "Resume configuration requires a checkpointed response-budget increase only"
                    )
                await self.write(
                    f"spec_history/tokens-{old_tokens}-to-{new_tokens}-{time.time_ns()}.json",
                    old,
                )
                await self.event(
                    "response_budget_increased",
                    previous_max_tokens=old_tokens,
                    max_tokens=new_tokens,
                    after_step=checkpoint["step"],
                )
            else:
                previous_inputs = {**old, "config": {**old["config"], "steps": target}}
                if previous_inputs != spec or target <= previous_target:
                    raise ValueError(
                        "Resume configuration or dataset differs from the saved run"
                    )
                await self.write(f"spec_history/{previous_target:04d}.json", old)
                await self.event(
                    "run_extended", previous_target=previous_target, target=target
                )
        await self.write("spec.json", spec)
        # This also repairs a restart between persisting the larger target and
        # retiring the old completion marker. The checkpoint is never removed.
        completed = self.read("complete.json")
        if completed is not None and completed["step"] < target:
            async with self.lock:
                destination = (
                    self.root / "completions" / f"{completed['step']:04d}.json"
                )
                destination.parent.mkdir(exist_ok=True)
                (self.root / "complete.json").replace(destination)
                if self.commit:
                    await self.commit()

    async def resume(self):
        state = self.read("checkpoint.json", {"step": 0, "path": None})
        # Exclude uncheckpointed steps from the canonical curve on rollback.
        for path in (self.root / "metrics").glob("*.json"):
            if int(path.stem) > state["step"]:
                dest = self.root / "rolled_back" / f"{time.time_ns()}-{path.name}"
                dest.parent.mkdir(exist_ok=True)
                path.replace(dest)
        # Evaluations beyond the restored policy version are stale too.
        for path in (self.root / "eval").glob("*.json"):
            if int(path.stem) > state["step"]:
                dest = self.root / "rolled_back" / f"eval-{time.time_ns()}-{path.name}"
                dest.parent.mkdir(exist_ok=True)
                path.replace(dest)
        if self.commit:
            await self.commit()
        return state
