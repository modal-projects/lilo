"""Read Modal container placement in a separate process/event loop."""

import asyncio
import json
import sys

from modal.client import _Client
from modal_proto import api_pb2


async def snapshot(app_id, pool_name):
    client = await _Client.from_env()
    result = await client.stub.TaskList(
        api_pb2.TaskListRequest(environment_name="kailash-dev")
    )
    output = []
    for task in result.tasks:
        if task.app_id != app_id and task.app_description != pool_name:
            continue
        info = (
            await client.stub.TaskGetInfo(
                api_pb2.TaskGetInfoRequest(task_id=task.task_id)
            )
        ).info
        output.append(
            dict(
                task_id=task.task_id,
                app_id=task.app_id,
                app_name=task.app_description,
                started_at=info.started_at,
                finished_at=info.finished_at,
                gpu_type=info.gpu_type,
                gpu_count=info.gpu_config.count,
            )
        )
    return output


if __name__ == "__main__":
    print(json.dumps(asyncio.run(snapshot(*sys.argv[1:]))))
