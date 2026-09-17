"""CPU-only reproduction against installed SGLang source; does not alter a server.

The normal finished-output path releases one reference. Its abort response helper
must not release that same reference again. Exercise the actual installed helper
and counter, extracted with AST to avoid initializing a tokenizer/model.
"""

import argparse
import ast
import asyncio
from http import HTTPStatus
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional


def extract(path, name):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.unparse(node)
    raise ValueError(name)


async def main(root):
    counter_source = extract(root / "utils/common.py", "ConcurrentCounter")
    abort_source = extract(
        root / "managers/tokenizer_manager.py", "_handle_abort_finish_reason"
    )
    import fastapi

    ns = dict(
        asyncio=asyncio,
        Callable=Callable,
        Optional=Optional,
        HTTPStatus=HTTPStatus,
        fastapi=fastapi,
        CLIENT_CLOSED_REQUEST=499,
        ReqState=object,
    )
    exec(counter_source, ns)
    exec(abort_source, ns)
    results = []
    for status in (503, 500, 499):
        counter = ns["ConcurrentCounter"]()
        await counter.increment()
        # _handle_batch_output's common state.finished branch owns this release.
        await counter.decrement()

        async def release(_):
            await counter.decrement()

        fake = SimpleNamespace(
            rid_to_state={},
            enable_lora=True,
            lora_registry=SimpleNamespace(release=release),
        )
        state = SimpleNamespace(
            obj=SimpleNamespace(rid="request", lora_path="adapter", lora_id="id")
        )
        output = {
            "meta_info": {
                "finish_reason": {
                    "type": "abort",
                    "status_code": status,
                    "message": "test abort",
                }
            }
        }
        try:
            await ns["_handle_abort_finish_reason"](fake, output, state, False)
        except fastapi.HTTPException as e:
            assert e.status_code == status
        try:
            await asyncio.wait_for(counter.wait_for_zero(), timeout=0.05)
            blocked = False
        except TimeoutError:
            blocked = True
        results.append(
            {
                "status_code": status,
                "counter_after_one_finished_request": counter.value(),
                "eviction_wait_blocks": blocked,
            }
        )
    # Control: exactly one completion release makes the eviction wait finish.
    control = ns["ConcurrentCounter"]()
    await control.increment()
    await control.decrement()
    await asyncio.wait_for(control.wait_for_zero(), timeout=0.05)
    print(
        json.dumps(
            {
                "source_root": str(root),
                "cases": results,
                "single_release_control": "passed",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-root",
        type=Path,
        default=Path("/sgl-workspace/sglang/python/sglang/srt"),
    )
    args = p.parse_args()
    asyncio.run(main(args.source_root))
