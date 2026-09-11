from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import modal

from codegolf.config import (
    APP_NAME,
    DEFAULT_RUN,
    DEFAULT_STEPS,
    DEFAULT_VARIANT,
    VOLUME_NAME,
    config_for,
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "modal>=1.5.3",
        "tinker>=0.24.1,<0.25",
        "transformers==5.16.1",
        "httpx",
        "jinja2==3.1.6",
        "matplotlib",
    )
    .add_local_python_source("codegolf", "lilo")
)


@app.function(
    image=image,
    volumes={"/runs": volume},
    secrets=[modal.Secret.from_name("lilo-api")],
    timeout=24 * 3600,
    memory=8192,
    env={"TINKER_BASE_URL": os.environ.get("TINKER_BASE_URL", "")},
    retries=modal.Retries(max_retries=10, initial_delay=10, max_delay=60),
    max_containers=1,
)
def run(
    run_name: str = DEFAULT_RUN,
    steps: int = DEFAULT_STEPS,
    variant: str = DEFAULT_VARIANT,
):
    from codegolf.train import train

    if not run_name or "/" in run_name or run_name in {".", ".."}:
        raise ValueError("Invalid run name")
    if not os.environ.get("TINKER_BASE_URL"):
        raise ValueError("Set TINKER_BASE_URL before deploying the example")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    cfg = config_for(variant, steps)
    volume.reload()
    return asyncio.run(
        train(
            Path("/runs") / run_name,
            Path("/runs/problems.json"),
            app,
            cfg,
            volume.commit.aio,
        )
    )


@app.function(image=image, timeout=600)
async def judge_smoke():
    from transformers import AutoTokenizer

    from codegolf.judge import judge

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Print 1 in Python"}],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert prompt and all(isinstance(token, int) for token in prompt)

    tests = [{"input": "2 3\n", "output": "5\n"}, {"input": "-4 8\n", "output": "4\n"}]
    correct = await judge("print(sum(map(int,input().split())))", tests, app)
    wrong = await judge("print(0)", tests, app)
    timeout = await judge("while True: pass", tests * 2, app)
    assert correct["passed"] and not wrong["passed"] and not timeout["passed"]
    assert timeout["tests_run"] == 1 and timeout["tests_total"] == 4
    excessive = await judge("while True: print('x'*1000)", tests, app)
    large_valid = await judge(
        "print('x'*1000000)", [{"input": "", "output": "x" * 1000000 + "\n"}], app
    )
    assert not excessive["passed"] and excessive["tests_run"] == 1
    assert large_valid["passed"]
    return {
        "correct": correct,
        "wrong": wrong,
        "timeout": timeout,
        "excessive": excessive,
        "large_valid": large_valid,
    }


@app.local_entrypoint()
def main(
    run_name: str = DEFAULT_RUN,
    steps: int = DEFAULT_STEPS,
    smoke: bool = False,
    variant: str = DEFAULT_VARIANT,
):
    if smoke:
        print(judge_smoke.remote())
        return
    print(run.remote(run_name, steps, variant))


@app.function(image=image, timeout=600)
async def judge_transport_smoke():
    from codegolf.judge import judge

    code = "#" + "padding" * 16000 + "\nprint(len(input()))"
    tests = [{"input": "a" * 90000 + "\n", "output": "90000\n"}]
    result = await judge(code, tests, app)
    assert result["passed"]
    return {
        "passed": True,
        "source_bytes": len(code.encode()),
        "input_bytes": len(tests[0]["input"].encode()),
    }
