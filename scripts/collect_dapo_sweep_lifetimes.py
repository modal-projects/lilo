"""Save final provider timestamps after each sweep app has stopped."""

import argparse
import asyncio
import json
from pathlib import Path

from modal.client import _Client
from modal_proto import api_pb2


async def collect(directories):
    client = await _Client.from_env()
    for directory in directories:
        state = json.loads((directory / "supervisor.json").read_text())
        if not state.get("drained"):
            raise ValueError(f"{directory} is not drained")
        known = {}
        for line in (directory / "resources.jsonl").read_text().splitlines():
            for task in json.loads(line).get("tasks", []):
                if task["gpu_count"]:
                    known[task["task_id"]] = task
        for task_id, task in known.items():
            info = (
                await client.stub.TaskGetInfo(
                    api_pb2.TaskGetInfoRequest(task_id=task_id)
                )
            ).info
            task.update(started_at=info.started_at, finished_at=info.finished_at)
            if not task["finished_at"]:
                raise ValueError(f"{task_id} has no final stop timestamp")
        (directory / "gpu-lifetimes.json").write_text(
            json.dumps(list(known.values()), indent=2) + "\n"
        )
        print(directory.name, len(known), "stopped GPU containers")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    asyncio.run(collect(parser.parse_args().runs))
