"""One isolated, checkpoint-free DAPO client-count measurement on Modal."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("MODAL_ENVIRONMENT", "kailash-dev")
os.environ["LILO_TRAINER_MAX_CONTAINERS"] = "1"
os.environ.setdefault("LILO_MILES_COMMIT", "ef3807c0ef659d7c6d8494c4933bd7ee0332700f")
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]

import modal
from lilo.providers.modal.app import app, ensure_lora_pool, server
from lilo.providers.modal.lora_pool import LoraPoolSpec, stop_pool
from lilo.providers.modal.scoped import control_image

ROOT = Path(__file__).resolve().parent
DEFINITION = "qwen3_5_9b_dapo_sweep"
volume = modal.Volume.from_name(
    "lilo-dapo-client-sweep", create_if_missing=True, version=2
)
image = (
    control_image("transformers==5.16.1", "jinja2==3.1.6", "numpy")
    .add_local_python_source("dapo_client_sweep_point", "dapo_sweep_workload")
    .add_local_file(ROOT / "data/dapo-sweep/dapo.jsonl", "/root/dapo.jsonl")
)


def prompts_for(rows, tokenizer):
    prompts = [
        tokenizer.encode(
            r["prompt"] + "\nGive the final answer in \\boxed{}.\nAnswer:",
            add_special_tokens=True,
        )
        for r in rows
    ]
    assert all(isinstance(p, list) and p and len(p) + 4096 < 16384 for p in prompts)
    return prompts


@app.function(
    image=image,
    volumes={"/assets": modal.Volume.from_name("lilo-model-assets")},
    timeout=300,
)
def preflight():
    from transformers import AutoTokenizer
    from dapo_sweep_workload import load_rows, make_datum

    tokenizer = AutoTokenizer.from_pretrained("/assets/Qwen3.5-9B-Base")
    rows = load_rows(Path("/root/dapo.jsonl"))
    prompts = prompts_for(rows, tokenizer)
    make_datum(prompts[0], [tokenizer.eos_token_id], logprobs=[-0.1], advantage=1.0)
    return {
        "status": "passed",
        "rows": len(rows),
        "max_prompt_tokens": max(map(len, prompts)),
    }


@app.function(
    image=image,
    volumes={"/runs": volume, "/assets": modal.Volume.from_name("lilo-model-assets")},
    secrets=[modal.Secret.from_name("lilo-api")],
    timeout=14400,
    cpu=16,
    memory=32768,
    retries=0,
    nonpreemptible=True,
)
def controller(url, app_id, run, count):
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import tinker
    from tinker import types
    from transformers import AutoTokenizer
    from dapo_sweep_workload import (
        PlacementProbe,
        load_rows,
        make_datum,
        numeric_answer,
        train_step,
        unload,
    )

    root = Path("/runs") / run
    root.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    report = dict(
        run=run,
        status="starting",
        started_at=time.time(),
        config=dict(
            clients=count,
            steps=8,
            warmup_steps=2,
            groups=8,
            samples_per_group=8,
            max_tokens=4096,
            context_length=16384,
            model="Qwen/Qwen3.5-9B-Base",
            rank=32,
            trainer_gpu="H200:8",
            tensor_parallel=8,
            trainer_slots=32,
            microbatch_token_budget=114688,
            inference_gpus=8,
            inference_running_requests_per_replica=128,
            inference_queued_requests_per_replica=256,
            inference_gpu_adapter_slots=16,
            inference_cpu_adapters=128,
            learning_rate=1e-5,
            loss="importance_sampling",
            max_policy_lag=2,
            prefetch_batches_per_client=1,
            checkpoints=False,
            dataset_sha256=hashlib.sha256(
                Path("/root/dapo.jsonl").read_bytes()
            ).hexdigest(),
        ),
        clients=[dict(index=i, steps=[], status="starting") for i in range(count)],
    )

    def persist():
        with lock:
            content = json.dumps(report, indent=2, allow_nan=False)
        (root / "report.tmp").write_text(content)
        (root / "report.tmp").replace(root / "report.json")
        volume.commit()

    stop_reporter = threading.Event()

    def periodic_report():
        while not stop_reporter.wait(30):
            try:
                persist()
            except Exception as exc:
                print("REPORT_WARNING", repr(exc), flush=True)

    reporter = threading.Thread(target=periodic_report, daemon=True)
    reporter.start()
    rows = load_rows(Path("/root/dapo.jsonl"))
    tokenizer = AutoTokenizer.from_pretrained("/assets/Qwen3.5-9B-Base")
    prompts = prompts_for(rows, tokenizer)
    clients = []
    service = tinker.ServiceClient(base_url=url, api_key=os.environ["TINKER_API_KEY"])

    def mark_start():
        report["measurement_start"] = time.time()
        report["status"] = "measuring"
        print("MEASUREMENT_START", report["measurement_start"], flush=True)

    barrier = threading.Barrier(count, action=mark_start, timeout=7200)
    start_barrier = threading.Barrier(count, timeout=7200)
    try:
        for i in range(count):
            c = service.create_lora_training_client(
                base_model=DEFINITION,
                rank=32,
                user_metadata={"run_id": f"{run}-client-{i}"},
            )
            clients.append(c)
            report["clients"][i]["model_id"] = c.model_id
            print("CLIENT_CREATED", i, c.model_id, flush=True)
        probe = PlacementProbe(app_id, DEFINITION)
        report["placement"] = probe.shared_engine([c.model_id for c in clients])
        report["status"] = "warming"
        persist()

        def client_loop(i):
            c = clients[i]
            own = report["clients"][i]

            def publish(update):
                t = time.monotonic()
                saved = c.save_weights_for_sampler(f"{run}-c{i}-u{update}").result(
                    timeout=3600
                )
                sampler = c.create_sampling_client(saved.path)
                return dict(sampler=sampler, policy_update=update), time.monotonic() - t

            def rollout(step, snapshot):
                t = time.monotonic()
                started = time.time()

                def group(g):
                    index = (i * 97 + (step - 1) * 8 + g) % len(rows)
                    prompt = prompts[index]
                    result = (
                        snapshot["sampler"]
                        .sample(
                            prompt=types.ModelInput.from_ints(prompt),
                            num_samples=8,
                            sampling_params=types.SamplingParams(
                                max_tokens=4096,
                                temperature=1.0,
                                seed=7341 + i * 10000 + step * 8 + g,
                            ),
                        )
                        .result(timeout=3600)
                    )
                    assert len(result.sequences) == 8
                    target = numeric_answer(str(rows[index]["answer"]))
                    rewards = [
                        float(numeric_answer(tokenizer.decode(s.tokens)) == target)
                        for s in result.sequences
                    ]
                    mean = sum(rewards) / 8
                    data = [
                        make_datum(
                            prompt,
                            list(s.tokens),
                            logprobs=list(s.logprobs),
                            advantage=r - mean,
                        )
                        for s, r in zip(result.sequences, rewards, strict=True)
                    ]
                    return dict(
                        data=data,
                        rewards=rewards,
                        prompt_tokens=len(prompt) * 8,
                        output_tokens=sum(len(s.tokens) for s in result.sequences),
                        truncated=sum(
                            "length" in str(s.stop_reason).lower()
                            for s in result.sequences
                        ),
                        nonzero_groups=int(any(r != mean for r in rewards)),
                    )

                with ThreadPoolExecutor(max_workers=8) as groups:
                    results = list(groups.map(group, range(8)))
                return dict(
                    data=[d for g in results for d in g["data"]],
                    policy_update=snapshot["policy_update"],
                    rollout_start=started,
                    rollout_end=time.time(),
                    rollout_seconds=time.monotonic() - t,
                    prompt_tokens=sum(g["prompt_tokens"] for g in results),
                    output_tokens=sum(g["output_tokens"] for g in results),
                    reward=sum(sum(g["rewards"]) for g in results) / 64,
                    truncated=sum(g["truncated"] for g in results),
                    nonzero_groups=sum(g["nonzero_groups"] for g in results),
                )

            snapshot, own["initial_publish_seconds"] = publish(0)
            start_barrier.wait()

            def run_updates(first, last, snapshot):
                previous = time.time()
                with ThreadPoolExecutor(max_workers=1) as prefetch:
                    pending = prefetch.submit(rollout, first, snapshot)
                    for step in range(first, last + 1):
                        wait = time.monotonic()
                        batch = pending.result()
                        ready_wait = time.monotonic() - wait
                        lag = step - 1 - batch["policy_update"]
                        assert 0 <= lag <= 2
                        if step < last:
                            pending = prefetch.submit(rollout, step + 1, snapshot)
                        trained = train_step(
                            c, batch["data"], "importance_sampling", 1e-5
                        )
                        assert trained["optimizer"].get("update_successful:mean") == 1
                        assert (
                            trained["forward"]["tokens:sum"]
                            == batch["prompt_tokens"] + batch["output_tokens"] - 64
                        )
                        # Publish every update except the terminal update, which has no future sampler.
                        publication = 0.0
                        if step < 8:
                            snapshot, publication = publish(step)
                        now = time.time()
                        entry = dict(
                            step=step,
                            step_seconds=now - previous,
                            completed_at=now,
                            **{k: v for k, v in batch.items() if k != "data"},
                            training=trained["forward"],
                            optimizer=trained["optimizer"],
                            train_seconds=trained["seconds"],
                            publish_seconds=publication,
                            rollout_wait_seconds=ready_wait,
                            policy_lag=lag,
                        )
                        with lock:
                            own["steps"].append(entry)
                        print(
                            "DAPO_STEP", json.dumps(dict(client=i, **entry)), flush=True
                        )
                        previous = now
                return snapshot

            try:
                snapshot = run_updates(1, 2, snapshot)
                own["status"] = "warm"
                barrier.wait()
                own["status"] = "measuring"
                run_updates(3, 8, snapshot)
                own["completed_at"] = time.time()
                own["status"] = "completed"
            except BaseException:
                barrier.abort()
                start_barrier.abort()
                own["status"] = "failed"
                raise

        with ThreadPoolExecutor(max_workers=count) as workers:
            futures = [workers.submit(client_loop, i) for i in range(count)]
            for f in as_completed(futures):
                f.result()
        report["measurement_end"] = max(c["completed_at"] for c in report["clients"])
        report["placement_final"] = probe.shared_engine([c.model_id for c in clients])
        report["status"] = "completed"
        return report
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        report["finished_at"] = time.time()
        stop_reporter.set()
        reporter.join(timeout=60)
        persist()
        # Every rollout future has joined before successful completion; no speculative leftovers.
        for c in clients:
            try:
                unload(url, c.model_id)
            except Exception as exc:
                print("UNLOAD_WARNING", repr(exc), flush=True)


def task_snapshot(app_id, pool_name):
    # Modal's synchronous API owns its event loop. A fresh process avoids sharing
    # that cached async client with the monitor thread or the shutdown checker.
    result = subprocess.run(
        [sys.executable, str(ROOT / "dapo_sweep_resources.py"), app_id, pool_name],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return json.loads(result.stdout)


def main():
    import threading

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clients", type=int, required=True, choices=[1, 2, 4, 8, 16, 32]
    )
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    output = ROOT / "results/dapo-client-sweep" / args.run
    output.mkdir(parents=True, exist_ok=True)
    spec = LoraPoolSpec(DEFINITION)
    state = dict(
        run=args.run,
        clients=args.clients,
        status="building",
        pool=spec.as_dict(),
        pool_app=spec.app_name,
    )

    def save():
        (output / "supervisor.json").write_text(json.dumps(state, indent=2))

    stop = threading.Event()
    observer = None

    def observe():
        with (output / "resources.jsonl").open("a", buffering=1) as stream:
            while not stop.is_set():
                try:
                    stream.write(
                        json.dumps(
                            dict(
                                time=time.time(),
                                tasks=task_snapshot(state["app_id"], spec.app_name),
                            )
                        )
                        + "\n"
                    )
                except Exception as exc:
                    stream.write(
                        json.dumps(dict(time=time.time(), error=repr(exc))) + "\n"
                    )
                stop.wait(10)

    save()
    try:
        with modal.enable_output(), app.run(name="lilo-" + args.run, detach=True):
            state.update(app_id=app.app_id, url=server.get_web_url(), status="starting")
            save()
            observer = threading.Thread(target=observe, daemon=True)
            observer.start()
            state["preflight"] = preflight.remote()
            save()
            ensure_lora_pool.remote(spec.as_dict())
            # Start the workload only once all eight fixed inference GPUs exist.
            # Model loading then overlaps trainer startup and the two warmup steps.
            deadline = time.monotonic() + 1800
            while time.monotonic() < deadline:
                tasks = task_snapshot(app.app_id, spec.app_name)
                replicas = [
                    t
                    for t in tasks
                    if t["app_name"] == spec.app_name
                    and t["gpu_type"] == "H200"
                    and t["started_at"] > 0
                ]
                if len(replicas) == 8 and sum(t["gpu_count"] for t in replicas) == 8:
                    state["inference_allocated_at"] = time.time()
                    save()
                    break
                time.sleep(10)
            else:
                raise TimeoutError(
                    "Eight inference GPUs did not start within 30 minutes"
                )
            call = controller.spawn(state["url"], app.app_id, args.run, args.clients)
            state.update(call_id=call.object_id, status="running")
            save()
            result = call.get()
            (output / "report.json").write_text(json.dumps(result, indent=2))
            state["status"] = result["status"]
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        raise
    finally:
        errors = []
        try:
            stop_pool(spec)
        except Exception as exc:
            errors.append(repr(exc))
        if state.get("app_id"):
            result = subprocess.run(
                [sys.executable, "-m", "modal", "app", "stop", "-y", state["app_id"]],
                check=False,
            )
            if result.returncode:
                errors.append("app stop failed")
            deadline = time.monotonic() + 600
            empty = 0
            while time.monotonic() < deadline:
                remaining = task_snapshot(state["app_id"], spec.app_name)
                empty = empty + 1 if not remaining else 0
                if empty >= 2:
                    break
                time.sleep(5)
            else:
                errors.append("tasks remained after stop timeout")
            state["remaining_tasks"] = remaining
            state["drained"] = empty >= 2
        stop.set()
        if observer:
            observer.join(timeout=30)
        state.update(finished_at=time.time(), cleanup_errors=errors)
        save()
        if errors:
            raise RuntimeError("Cleanup failed: " + str(errors))


if __name__ == "__main__":
    main()
