"""Small real-GPU training/publication/sampling check for a Python-configured deployment.

Run from an authenticated operator environment. Results are written to --output;
credentials are read from TINKER_API_KEY and never included in the report.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import httpx
import modal
import tinker
from tinker import types

from lilo.providers.modal.deployment_records import deployed_manifest
from lilo.client import create_full_training_client
from lilo.providers.modal.kv import app_store_name


def write(path, report):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2) + "\n")
    temp.replace(path)


def trainer_record(app_id, model_id):
    placement = modal.Dict.from_name(app_store_name("lilo-models", app_id)).get(
        f"placement:{model_id}"
    )
    if not placement:
        raise RuntimeError(f"No trainer placement for {model_id}")
    row = modal.Dict.from_name(app_store_name("lilo-engines", app_id)).get(
        f"engine_instance:{placement['engine_instance_id']}"
    )
    return {
        key: row.get(key)
        for key in (
            "instance_id",
            "definition_id",
            "state",
            "boot_id",
            "call_id",
            "revision",
        )
    }


def step(training, tokens):
    datum = types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tokens[1:],
            "weights": [1.0] * (len(tokens) - 1),
        },
    )
    start = time.monotonic()
    result = training.forward_backward([datum], "cross_entropy").result(timeout=3600)
    if len(result.loss_fn_outputs[0]["logprobs"].data) != len(tokens) - 1:
        raise AssertionError("wrong forward/backward output length")
    if not all(
        math.isfinite(float(x)) for x in result.loss_fn_outputs[0]["logprobs"].data
    ):
        raise AssertionError("non-finite trainer logprobs")
    train_seconds = time.monotonic() - start
    optim = training.optim_step(types.AdamParams(learning_rate=1e-4)).result(
        timeout=3600
    )
    sample = training.save_weights_and_get_sampling_client()
    published = time.monotonic()
    output = sample.sample(
        prompt=types.ModelInput.from_ints(tokens[:16]),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=16, temperature=0.0, seed=42),
    ).result(timeout=3600)
    sequence = output.sequences[0]
    assert len(sequence.tokens) > 0
    assert sequence.logprobs is not None and len(sequence.logprobs) == len(
        sequence.tokens
    )
    assert all(math.isfinite(float(v)) for v in sequence.logprobs)
    return {
        "train_seconds": train_seconds,
        "step_seconds": time.monotonic() - start,
        "sample_seconds": time.monotonic() - published,
        "train_metrics": result.metrics,
        "optim_metrics": optim.metrics,
        "generated_tokens": len(sequence.tokens),
        "sample_tokens": sequence.tokens,
        "sample_logprobs": sequence.logprobs,
    }


def run(args):
    app = modal.App.lookup(args.frontend)
    server = modal.Function.from_name(args.frontend, "server")
    url = server.get_web_url()
    headers = {"X-API-Key": os.environ["TINKER_API_KEY"]}
    rows = deployed_manifest(args.frontend)
    row = next(r for r in rows if r["active"] and r["spec"]["name"] == args.name)
    definition_id = f"deployment_{args.name}_{row['generation'][:16]}"
    report = {
        "frontend": args.frontend,
        "app_id": app.app_id,
        "name": args.name,
        "generation": row["generation"],
        "spec": row["spec"],
        "status": "creating",
        "steps": [],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, report)
    service = tinker.ServiceClient(base_url=url, api_key=os.environ["TINKER_API_KEY"])
    training = None
    try:
        start = time.monotonic()
        training = (
            service.create_lora_training_client(base_model=definition_id, rank=32)
            if row["spec"]["model"]["parameterization"] == "lora"
            else create_full_training_client(service, definition_id)
        )
        report.update(
            model_id=training.model_id,
            create_seconds=time.monotonic() - start,
            trainer_before=trainer_record(app.app_id, training.model_id),
            status="training",
        )
        write(output, report)
        tokenizer = training.get_tokenizer()
        text = "The sum of one and one is two. The sum of two and two is four.\n" * 40
        tokens = tokenizer.encode(text, add_special_tokens=True)[:513]
        for index in range(args.steps):
            report["steps"].append(step(training, tokens))
            write(output, report)
            print(f"{args.name}: completed step {index + 1}", flush=True)
        if args.continue_file:
            report["status"] = "waiting_for_redeploy"
            write(output, report)
            deadline = time.monotonic() + 3600
            while not Path(args.continue_file).exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("redeploy gate did not open")
                time.sleep(5)
            report["after_redeploy_step"] = step(training, tokens)
            report["trainer_after"] = trainer_record(app.app_id, training.model_id)
            assert (
                report["trainer_before"]["boot_id"]
                == report["trainer_after"]["boot_id"]
            ), "unchanged trainer restarted"
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write(output, report)
        if training is not None:
            try:
                with httpx.Client(base_url=url, headers=headers, timeout=60) as client:
                    response = client.post(
                        "/api/v1/unload_model", json={"model_id": training.model_id}
                    )
                    response.raise_for_status()
                    request = response.json()["request_id"]
                    deadline = time.monotonic() + 180
                    while time.monotonic() < deadline:
                        response = client.post(
                            "/api/v1/retrieve_future", json={"request_id": request}
                        )
                        if response.status_code != 408:
                            response.raise_for_status()
                            break
            except Exception as exc:
                report["cleanup_error"] = str(exc)
                write(output, report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", required=True)
    parser.add_argument("--continue-file")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
