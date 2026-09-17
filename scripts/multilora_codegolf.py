"""Four-client codegolf on a dedicated Miles deployment; never deploys over lilo.

Run from this checkout with .venv/bin/python scripts/multilora_codegolf.py --help.
All controller code is mounted from this checkout. Results are persisted to a
separate Modal volume and mirrored locally by the status command.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "examples/codeforces-codegolf"))
os.environ.setdefault("LILO_TRAINER_MAX_CONTAINERS", "1")
os.environ.setdefault("MODAL_ENVIRONMENT", "kailash-dev")
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]

import modal
from codegolf.config import config_for
from codegolf.reward import datum
from lilo.providers.modal.app import app, server
from lilo.providers.modal.scoped import control_image

DEFINITION = "qwen3_5_9b_miles_lora_64k"
VOLUME = "lilo-multilora-codegolf"
CLIENTS = 4
RANK = 32
volume = modal.Volume.from_name(VOLUME, create_if_missing=True, version=2)
controller_image = control_image(
    "transformers==5.16.1", "jinja2==3.1.6", "numpy"
).add_local_python_source("codegolf", "multilora_codegolf")


def weighted_ppo_datum(prompt, tokens, logprobs, advantage, sequence_weight=1.0):
    """Miles sums PPO losses; fold the reference's nonnegative mask into A.

    w * min(r*A, clip(r)*A) == min(r*(w*A), clip(r)*(w*A)).
    Keep prompt advantages zero. Do not divide again by tokens or client count.
    """
    if not math.isfinite(sequence_weight) or sequence_weight < 0:
        raise ValueError("PPO sequence weights must be finite and nonnegative")
    result = datum(prompt, tokens, logprobs, advantage * sequence_weight)
    return type(result)(
        model_input=result.model_input,
        loss_fn_inputs={
            k: v for k, v in result.loss_fn_inputs.items() if k != "weights"
        },
    )


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


async def placement(model_ids, app_id):
    from lilo.providers.modal.kv import ModalKeyValueStore, app_store_name, STORE_NAMES
    from lilo.control_plane.keys import placement_key

    kv = ModalKeyValueStore(
        modal.Dict.from_name(app_store_name(STORE_NAMES["models"], app_id))
    )
    records = [await kv.get(placement_key(mid)) for mid in model_ids]
    assert all(r and r["engine_definition_id"] == DEFINITION for r in records), records
    identities = {(r["engine_instance_id"], r["engine_boot_id"]) for r in records}
    assert len(identities) == 1, identities
    return {
        "engine_instance_id": records[0]["engine_instance_id"],
        "engine_boot_id": records[0]["engine_boot_id"],
        "model_ids": model_ids,
    }


@app.function(
    image=controller_image,
    volumes={"/runs": volume},
    secrets=[modal.Secret.from_name("lilo-api")],
    timeout=86400,
    memory=16384,
    retries=0,
)
async def controller(url, parent_app_id, run_name, steps, phase):
    import tinker
    from tinker import types
    from codegolf.train import train, release

    os.environ["TINKER_BASE_URL"] = url
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    volume.reload()
    root = Path("/runs") / run_name
    root.mkdir(parents=True, exist_ok=True)
    lock = asyncio.Lock()

    async def commit():
        async with lock:
            await volume.commit.aio()

    async def report(name, value):
        write_json(root / name, value)
        await commit()

    await report(
        "controller.json",
        {
            "phase": phase,
            "steps": steps,
            "started_at": time.time(),
            "status": "running",
        },
    )
    clients = []
    try:
        service = tinker.ServiceClient(
            base_url=url, api_key=os.environ["TINKER_API_KEY"]
        )
        if phase == "capacity":
            for i in range(CLIENTS):
                clients.append(
                    await service.create_lora_training_client_async(
                        base_model=DEFINITION,
                        rank=RANK,
                        user_metadata={
                            "run_id": run_name,
                            "attempt_id": f"capacity-{i}",
                        },
                    )
                )
            tok = await asyncio.to_thread(clients[0].get_tokenizer)
            chunk = tok.encode(
                "def solve():\n    x = int(input())\n    print(x + 1)\n",
                add_special_tokens=False,
            )
            tokens = (chunk * (65536 // len(chunk) + 1))[:65536]
            item = types.Datum(
                model_input=types.ModelInput.from_ints(tokens[:-1]),
                loss_fn_inputs={
                    "target_tokens": tokens[1:],
                    "weights": [1.0 / 65535] * 65535,
                },
            )
            short = types.Datum(
                model_input=types.ModelInput.from_ints(tokens[:255]),
                loss_fn_inputs={"target_tokens": tokens[1:256], "weights": [1.0] * 255},
            )

            async def forward(c):
                out = await (
                    await c.forward_async([short], "cross_entropy")
                ).result_async()
                return list(out.loss_fn_outputs[0]["logprobs"].data)

            before = await asyncio.gather(*(forward(c) for c in clients))
            results = {
                "placement": await placement(
                    [c.model_id for c in clients], parent_app_id
                ),
                "context_tokens": 65536,
                "clients": CLIENTS,
                "rank": RANK,
                "updates": [],
            }
            await report("capacity.json", results)

            async def update(c):
                started = time.monotonic()
                fb = await c.forward_backward_async([item], "cross_entropy")
                opt = await c.optim_step_async(types.AdamParams(learning_rate=1e-6))
                output, optimizer = await asyncio.gather(
                    fb.result_async(), opt.result_async()
                )
                lp = output.loss_fn_outputs[0]["logprobs"].data
                assert len(lp) == 65535 and all(math.isfinite(x) for x in lp)
                metrics = {**output.metrics, **optimizer.metrics}
                assert all(
                    math.isfinite(v)
                    for v in metrics.values()
                    if isinstance(v, (float, int))
                )
                return {"seconds": time.monotonic() - started, "metrics": metrics}

            for step in range(2):
                results["updates"].append(
                    await asyncio.gather(*(update(c) for c in clients))
                )
                await report("capacity.json", results)
            after = await asyncio.gather(*(forward(c) for c in clients))
            results["update_logprob_max_abs"] = [
                max(abs(a - b) for a, b in zip(x, y, strict=True))
                for x, y in zip(before, after, strict=True)
            ]
            assert all(x > 0 for x in results["update_logprob_max_abs"])
            # Validate native optimizer checkpoints and exported adapters for every slot.
            results["roundtrips"] = []
            for c, expected in zip(clients, after, strict=True):
                sampler = await c.save_weights_and_get_sampling_client_async()
                infer_before = await sampler.compute_logprobs_async(
                    types.ModelInput.from_ints(tokens[:256])
                )
                saved = await (
                    await c.save_state_async("capacity-roundtrip")
                ).result_async()
                await update(c)
                await (
                    await c.load_state_with_optimizer_async(saved.path)
                ).result_async()
                restored = await forward(c)
                sampler = await c.save_weights_and_get_sampling_client_async()
                infer_after = await sampler.compute_logprobs_async(
                    types.ModelInput.from_ints(tokens[:256])
                )
                trainer_diff = max(
                    abs(a - b) for a, b in zip(expected, restored, strict=True)
                )
                inference_diff = max(
                    abs(a - b)
                    for a, b in zip(infer_before, infer_after, strict=True)
                    if a is not None and b is not None
                )
                assert trainer_diff <= 1e-5 and inference_diff <= 1e-5, (
                    trainer_diff,
                    inference_diff,
                )
                results["roundtrips"].append(
                    {
                        "model_id": c.model_id,
                        "checkpoint": saved.path,
                        "trainer_reload_max_abs": trainer_diff,
                        "inference_reload_max_abs": inference_diff,
                    }
                )
                await report("capacity.json", results)
            results["status"] = "passed"
            await report("capacity.json", results)
        else:
            cfg = config_for("tailrl", steps)
            config = dataclasses.asdict(cfg)
            await report(
                "manifest.json",
                {
                    "config": config,
                    "definition": DEFINITION,
                    "clients": CLIENTS,
                    "rank": RANK,
                    "lora_alpha": 32,
                    "dataset_sha256": hashlib.sha256(
                        Path("/runs/problems.json").read_bytes()
                    ).hexdigest(),
                    "trainer_gpu": "H200:8",
                    "tp": 8,
                    "cp": 1,
                    "dp": 1,
                    "context": 65536,
                    "reference_variant": "tailrl",
                    "reference_split": json.loads(
                        Path("/runs/reference-split.json").read_text()
                    ),
                    "loss_translation": "nonnegative sequence weights folded into PPO advantages",
                },
            )
            bound = {}

            async def create_client(svc, model, *, user_metadata):
                assert model == "Qwen/Qwen3.5-9B"
                c = await svc.create_lora_training_client_async(
                    base_model=DEFINITION, rank=RANK, user_metadata=user_metadata
                )
                bound[user_metadata["run_id"]] = c.model_id
                if len(bound) == CLIENTS:
                    # Forward barrier ensures all placements exist before asserting sharing.
                    await report(
                        "clients.json",
                        {
                            "clients": dict(bound),
                            "placement": await placement(
                                list(bound.values()), parent_app_id
                            ),
                        },
                    )
                return c

            semaphore = asyncio.Semaphore(cfg.judge_concurrency)
            # Reference solutions were sandbox-validated by the original FFT run.
            validated = json.loads(Path("/runs/reference-validated.json").read_text())
            for i in range(CLIENTS):
                sub = root / f"{run_name}-client-{i}"
                if not (sub / "validated.json").exists():
                    write_json(sub / "validated.json", validated)
            await commit()
            async with asyncio.TaskGroup() as group:
                tasks = [
                    group.create_task(
                        train(
                            root / f"{run_name}-client-{i}",
                            Path("/runs/problems.json"),
                            app,
                            cfg,
                            commit,
                            create_training=create_client,
                            make_datum=weighted_ppo_datum,
                            recovery_limit=1 if phase == "smoke" else 30,
                            judge_semaphore=semaphore,
                        )
                    )
                    for i in range(CLIENTS)
                ]
            await report("complete.json", {"clients": [t.result() for t in tasks]})
        await report(
            "controller.json",
            {
                "phase": phase,
                "steps": steps,
                "status": "passed",
                "finished_at": time.time(),
            },
        )
        return {"status": "passed", "phase": phase, "steps": steps}
    except BaseException as exc:
        await report(
            "controller.json",
            {
                "phase": phase,
                "status": "failed",
                "error": repr(exc),
                "finished_at": time.time(),
            },
        )
        raise
    finally:
        for c in clients:
            await release(c)


def supervise(args):
    from lilo.providers.modal.lora_pool import LoraPoolSpec, stop_pool
    from lilo.providers.modal.miles_image import MILES_COMMIT

    spec = LoraPoolSpec(DEFINITION)
    state = {
        "status": "building",
        "run": args.run,
        "phase": args.phase,
        "steps": args.steps,
        "miles_revision": MILES_COMMIT,
        "pool": spec.as_dict(),
    }

    def save():
        write_json(args.output / "supervisor.json", state)

    save()
    try:
        with (
            modal.enable_output(),
            app.run(name="multilora-codegolf-" + args.run, detach=True),
        ):
            state.update(
                app_id=app.app_id, base_url=server.get_web_url(), status="running"
            )
            save()
            call = controller.spawn(
                state["base_url"], app.app_id, args.run, args.steps, args.phase
            )
            state["call_id"] = call.object_id
            save()
            state["result"] = call.get()
            state["status"] = "passed"
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        raise
    finally:
        try:
            stop_pool(spec)
        except Exception as exc:
            state["cleanup_error"] = repr(exc)
        # Detached apps need explicit stop after their controller finishes.
        if state.get("app_id"):
            subprocess.run(
                [sys.executable, "-m", "modal", "app", "stop", "-y", state["app_id"]],
                check=False,
            )
        state["finished_at"] = time.time()
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["launch", "supervise", "status"])
    parser.add_argument("--run", required=True)
    parser.add_argument(
        "--phase", choices=["capacity", "smoke", "hero"], default="smoke"
    )
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    import re

    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.run):
        parser.error("Invalid run name")
    args.output = REPO / "scripts/results/multilora-codegolf" / args.run
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "supervise":
        return supervise(args)
    if args.command == "launch":
        if (args.output / "supervisor.pid").exists():
            raise RuntimeError("Run name already launched")
        with (args.output / "supervisor.log").open("w") as log:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "supervise",
                    "--run",
                    args.run,
                    "--phase",
                    args.phase,
                    "--steps",
                    str(args.steps),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        (args.output / "supervisor.pid").write_text(str(proc.pid) + "\n")
        print(json.dumps({"pid": proc.pid, "output": str(args.output)}))
        return
    result = {"supervisor": json.loads((args.output / "supervisor.json").read_text())}
    remote = modal.Volume.from_name(VOLUME)
    for name in [
        "controller.json",
        "capacity.json",
        "manifest.json",
        "complete.json",
        "clients.json",
    ]:
        try:
            value = json.loads(b"".join(remote.read_file(args.run + "/" + name)))
            write_json(args.output / name, value)
            result[name] = value
        except (FileNotFoundError, modal.exception.NotFoundError):
            pass
    for i in range(CLIENTS):
        sub = f"{args.run}/{args.run}-client-{i}"
        client = {}
        for name in [
            "checkpoint.json",
            "complete.json",
            "live.json",
            "spec.json",
            "split.json",
        ]:
            try:
                value = json.loads(b"".join(remote.read_file(sub + "/" + name)))
                write_json(args.output / f"client-{i}" / name, value)
                if name not in ("spec.json", "split.json"):
                    client[name] = value
            except (FileNotFoundError, modal.exception.NotFoundError):
                pass
        for folder in ["metrics", "eval", "events"]:
            try:
                entries = sorted(
                    remote.listdir(sub + "/" + folder), key=lambda e: e.path
                )
                if entries:
                    value = json.loads(b"".join(remote.read_file(entries[-1].path)))
                    write_json(
                        args.output
                        / f"client-{i}"
                        / folder
                        / Path(entries[-1].path).name,
                        value,
                    )
                    client["latest_" + folder] = value
            except (FileNotFoundError, modal.exception.NotFoundError):
                pass
        if client:
            result[f"client-{i}"] = client
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
