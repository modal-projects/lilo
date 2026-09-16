from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path

import modal
from lilo.providers.modal.scoped import control_image

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
image = control_image(
    "transformers==5.16.1",
    "jinja2==3.1.6",
    "matplotlib",
).add_local_python_source("codegolf")


@app.function(
    image=image,
    volumes={"/runs": volume},
    secrets=[modal.Secret.from_name("lilo-api")],
    timeout=24 * 3600,
    memory=8192,
    env={"MODAL_ENVIRONMENT": os.environ.get("MODAL_ENVIRONMENT", "")},
    retries=modal.Retries(max_retries=10, initial_delay=10, max_delay=60),
    max_containers=1,
)
def run(
    run_name: str = DEFAULT_RUN,
    steps: int = DEFAULT_STEPS,
    variant: str = DEFAULT_VARIANT,
    eval_samples: int | None = None,
):
    from codegolf.train import train

    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", run_name) or run_name in {".", ".."}:
        raise ValueError("Invalid run name")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    cfg = config_for(variant, steps, eval_samples=eval_samples)
    volume.reload()
    import lilo
    from lilo.engines import qwen3_5_9b_full_64k

    # The remote CPU controller owns the ephemeral deployment. A retry opens a
    # fresh scope; train() restores only its last committed full checkpoint.
    telemetry_env = {
        key: value for key, value in os.environ.items() if key.startswith("OTEL_")
    }
    resource_attrs = telemetry_env.get("OTEL_RESOURCE_ATTRIBUTES", "")
    telemetry_env["OTEL_RESOURCE_ATTRIBUTES"] = ",".join(
        item for item in (resource_attrs, f"lilo.run_id={run_name}") if item
    )
    with lilo.run(
        engine=qwen3_5_9b_full_64k(),
        warm=False,
        name=f"codegolf-{run_name}",
        telemetry_secret=modal.Secret.from_dict(telemetry_env),
        latest=lilo.Pool(
            min_containers=getattr(cfg, "rollout_min_replicas", 1),
            max_containers=getattr(cfg, "rollout_max_replicas", 2),
            scaledown_window=1200,
        ),
    ) as (url, api_key):
        os.environ["TINKER_BASE_URL"] = url
        os.environ["TINKER_API_KEY"] = api_key
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
    eval_samples: int | None = None,
):
    if smoke:
        print(judge_smoke.remote())
        return
    print(run.remote(run_name, steps, variant, eval_samples=eval_samples))


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
