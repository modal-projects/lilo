"""Isolated PR-21 GPU and owner-lifecycle E2E tests.

Run with Python 3.12, PYTHONPATH=src and MODAL_ENVIRONMENT set explicitly.
Creates only apps prefixed lilo-e2e-. Never targets pre-existing apps.
Reports API limitations as failed checks and continues independent tests.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import httpx
import modal
from modal.client import _Client
from modal._utils.async_utils import synchronize_api
from modal_proto import api_pb2
import tinker
from tinker import types

import lilo
from lilo.engines import qwen3_5_4b_full_64k


class ScenarioComplete(Exception):
    pass


class Report:
    def __init__(self, path, mode):
        self.lock = threading.RLock()
        self.path = Path(path)
        self.data = dict(
            mode=mode, started_at=time.time(), events=[], checks=[], apps=[]
        )
        self.save()

    def save(self):
        with self.lock:
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.data, indent=2, default=str))
            temporary.replace(self.path)

    def event(self, event_name, **values):
        with self.lock:
            value = dict(event=event_name, at=time.time(), **values)
            self.data["events"].append(value)
            self.save()
            print(json.dumps(value, default=str), flush=True)

    def check(self, name, condition, **values):
        with self.lock:
            value = dict(name=name, passed=bool(condition), **values)
            self.data["checks"].append(value)
            self.event("check", **value)

    def app(self, app_id):
        with self.lock:
            if app_id and app_id not in self.data["apps"]:
                self.data["apps"].append(app_id)
                self.save()


def until(fn, timeout=180, interval=2):
    deadline = time.monotonic() + timeout
    while True:
        value = fn()
        if value:
            return value
        if time.monotonic() >= deadline:
            raise TimeoutError(f"condition did not hold within {timeout}s")
        time.sleep(interval)


@synchronize_api
async def rpc(method, request):
    client = await _Client.from_env()
    return await getattr(client.stub, method)(request)


def task_ids(app_id):
    response = rpc(
        "TaskList",
        api_pb2.TaskListRequest(
            environment_name=os.environ["MODAL_ENVIRONMENT"], app_id=app_id
        ),
    )
    return [task.task_id for task in response.tasks]


def app_states():
    response = rpc(
        "AppList",
        api_pb2.AppListRequest(environment_name=os.environ["MODAL_ENVIRONMENT"]),
    )
    return {
        app.app_id: dict(state=int(app.state), containers=app.n_running_tasks)
        for app in response.apps
    }


def wait_cleanup(report, timeout=180):
    ids = report.data["apps"]
    deadline = time.monotonic() + timeout
    while True:
        states = app_states()
        owned = {a: states.get(a) for a in ids}
        if all(
            v is None
            or (v["state"] == api_pb2.APP_STATE_STOPPED and not v["containers"])
            for v in owned.values()
        ):
            report.check("all_owned_apps_stopped", True, states=owned)
            return
        if time.monotonic() >= deadline:
            report.check("all_owned_apps_stopped", False, states=owned)
            return
        time.sleep(3)


def create_bounded(service, model, timeout=1800):
    # Public helper has no timeout; its underlying future does.
    from lilo.client import _create_full_training_client_submit

    return _create_full_training_client_submit(service, model, None, None).result(
        timeout=timeout
    )


def forward_values(result):
    values = []
    for output in result.loss_fn_outputs:
        data = output["logprobs"]
        values.extend(data.data if hasattr(data, "data") else data["data"])
    assert values and all(math.isfinite(x) for x in values)
    return values


def compare(report, name, left, right, tolerance=0.003):
    same_length = len(left) == len(right)
    error = max((abs(a - b) for a, b in zip(left, right)), default=float("inf"))
    report.check(
        name,
        same_length and error <= tolerance,
        max_abs_error=error,
        tolerance=tolerance,
        count=len(left),
    )


def run_worker(args):
    report = Report(args.report, args.mode)
    import lilo.providers.modal.scoped as scoped

    original_build = scoped.build_app
    resources = {}
    services = []

    def capture(*a, **kw):
        result = original_build(*a, **kw)
        resources.update(
            app=result[0], registry_name=a[2], manage=result[2], servers=result[3]
        )
        return result

    scoped.build_app = capture
    engine = qwen3_5_4b_full_64k()
    if args.asset_path:
        engine = replace(
            engine, training=replace(engine.training, hf_checkpoint=args.asset_path)
        )
    name = "lilo-e2e-" + args.mode + "-" + uuid.uuid4().hex[:6]
    report.data.update(
        name=name,
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    )
    report.save()
    try:
        with (
            modal.enable_output(),
            lilo.run(
                engine=engine,
                name=name,
                warm=args.mode == "gpu",
                latest=lilo.Pool(
                    min_containers=1, max_containers=1, scaledown_window=60
                ),
            ) as (url, api_key),
        ):
            app_id = resources["app"].app_id
            report.app(app_id)
            report.event("ready", app_id=app_id, url=url)
            registry = modal.Dict.from_name(resources["registry_name"])
            manage = resources["manage"]
            if args.mode == "isolation":
                with httpx.Client(timeout=90, trust_env=False) as http:
                    auth_a = {"X-Api-Key": api_key}
                    session_a = http.post(
                        url + "/api/v1/create_session", headers=auth_a, json={}
                    )
                    session_a.raise_for_status()
                    session_id = session_a.json()["session_id"]
                    with lilo.run(engine=engine, name=name, warm=False) as (
                        url_b,
                        key_b,
                    ):
                        report.app(resources["app"].app_id)
                        auth_b = {"X-Api-Key": key_b}
                        response = http.get(
                            url_b + "/api/v1/get_server_capabilities", headers=auth_b
                        )
                        report.check(
                            "second_scoped_api_works", response.status_code == 200
                        )
                        response = http.get(
                            url_b + "/api/v1/get_server_capabilities", headers=auth_a
                        )
                        report.check(
                            "credentials_isolated", response.status_code == 401
                        )
                        response = http.post(
                            url_b + "/api/v1/session_heartbeat",
                            headers=auth_b,
                            json={"session_id": session_id},
                        )
                        report.check(
                            "session_state_isolated",
                            response.status_code == 404,
                            status=response.status_code,
                            body=response.text[:300],
                        )
                        for owned in report.data["apps"]:
                            records = modal.Dict.from_name(
                                owned + "-engines", create_if_missing=True
                            )
                            report.check(
                                "warm_false_starts_no_trainer:" + owned,
                                not any(
                                    k.startswith("engine_instance:")
                                    for k, v in records.items()
                                ),
                            )
                    response = http.post(
                        url + "/api/v1/session_heartbeat",
                        headers=auth_a,
                        json={"session_id": session_id},
                    )
                    report.check(
                        "outer_run_survives_inner_exit", response.status_code == 200
                    )
                report.event("body_complete")
                raise ScenarioComplete()
            if args.mode in ("hard-owner", "exception"):
                # Exercise real pinned-app ownership without starting GPU replicas.
                manage.remote("pinned", "lifecycle-probe", 0)
                pin_key = "pinned:lifecycle-probe:0"
                route = until(lambda: registry.get(pin_key), timeout=180)
                report.app(route["app_id"])
                report.check("child_app_owned", route["app_id"] != app_id, route=route)
                # Active leases must survive a synthetic idle timestamp.
                manage.remote("pinned", "lifecycle-probe", 0, "probe-lease")
                records = registry.get("pin_records")
                records[pin_key]["last_used"] = time.time() - 700
                registry.put("pin_records", records)
                demand = manage.remote("pin_demand")
                report.check("active_lease_prevents_idle_eviction", pin_key in demand)
                manage.remote("release_pinned", "lifecycle-probe", 0, "probe-lease")
                if args.mode == "exception":
                    report.event("intentional_body_exception")
                    raise RuntimeError("intentional-e2e-body-exception")
                report.event("ready_for_owner_kill")
                while True:
                    time.sleep(1)

            models = modal.Dict.from_name(app_id + "-models", create_if_missing=True)
            report.check("warm_does_not_accept_model", not list(models.items()))
            engines = modal.Dict.from_name(app_id + "-engines")

            def engine_records():
                return [
                    v for k, v in engines.items() if k.startswith("engine_instance:")
                ]

            report.check(
                "warm_has_running_trainer",
                any(r["state"] == "running" for r in engine_records()),
            )
            with httpx.Client(base_url=url, timeout=90, trust_env=False) as http:
                bad = http.get(
                    "/api/v1/get_server_capabilities", headers={"X-Api-Key": "wrong"}
                )
                report.check(
                    "wrong_api_key_rejected",
                    bad.status_code in (401, 403),
                    status=bad.status_code,
                )
            service = tinker.ServiceClient(base_url=url, api_key=api_key)
            peer = tinker.ServiceClient(base_url=url, api_key=api_key)
            services.extend([service, peer])
            trainer = create_bounded(service, engine.model)
            report.event("training_client", model_id=trainer.model_id)
            tokenizer = trainer.get_tokenizer()
            tokens = tokenizer.encode(
                "The capital of France is Paris. The capital of Italy is Rome."
            )
            datum = types.Datum(
                model_input=types.ModelInput.from_ints(tokens[:-1]),
                loss_fn_inputs={
                    "target_tokens": tokens[1:],
                    "weights": [1.0] * (len(tokens) - 1),
                },
            )
            prompt = types.ModelInput.from_ints(
                tokenizer.encode("The capital of France is")
            )

            def sample(client, label):
                start = time.time()
                result = client.sample(
                    prompt=prompt,
                    num_samples=1,
                    sampling_params=types.SamplingParams(max_tokens=12, temperature=0),
                ).result(timeout=1800)
                seq = result.sequences[0]
                assert seq.tokens and len(seq.logprobs) == len(seq.tokens)
                assert all(math.isfinite(x) for x in seq.logprobs)
                value = {"tokens": seq.tokens, "logprobs": seq.logprobs}
                report.event(label, seconds=time.time() - start, **value)
                return value

            def update(client, label):
                result = client.forward_backward([datum], "cross_entropy").result(
                    timeout=1800
                )
                assert all(
                    math.isfinite(v)
                    for v in result.metrics.values()
                    if isinstance(v, (float, int))
                )
                optim = client.optim_step(types.AdamParams(learning_rate=5e-5)).result(
                    timeout=1800
                )
                report.check(
                    label,
                    optim.metrics.get("update_successful:mean") == 1.0,
                    forward=result.metrics,
                    optimizer=optim.metrics,
                )
                return result

            base = peer.create_sampling_client(base_model=engine.model)
            base_before = sample(base, "base_initial")
            update(trainer, "first_optimizer_step")
            checkpoint = trainer.save_state("e2e-optimizer").result(timeout=1800)
            report.event("checkpoint_saved", path=checkpoint.path)
            checkpoint_values = forward_values(
                trainer.forward([datum], "cross_entropy").result(timeout=1800)
            )
            latest = trainer.save_weights_and_get_sampling_client()
            sample(latest, "latest_first")
            pinned_path = (
                trainer.save_weights_for_sampler("e2e-pin").result(timeout=1800).path
            )
            pinned = peer.create_sampling_client(model_path=pinned_path)
            pinned_before = sample(pinned, "pinned_first")
            pin_key = next(
                k
                for k in (registry.get("pin_records") or {})
                if k.startswith("pinned:" + trainer.model_id + ":")
            )
            pin_route = registry.get(pin_key)
            report.app(pin_route["app_id"])
            # A second client must not displace a healthy training model.
            try:
                create_bounded(peer, engine.model, timeout=60)
            except Exception as exc:
                report.check(
                    "second_live_trainer_rejected",
                    "already active" in str(exc),
                    error=str(exc)[:500],
                )
            else:
                report.check("second_live_trainer_rejected", False)
            # Capture uninterrupted continuation for an optimizer-state comparison.
            update(trainer, "uninterrupted_second_step")
            continuation_values = forward_values(
                trainer.forward([datum], "cross_entropy").result(timeout=1800)
            )
            latest_new = trainer.save_weights_and_get_sampling_client()
            with ThreadPoolExecutor(max_workers=3) as pool:
                jobs = [
                    pool.submit(sample, c, label)
                    for c, label in (
                        (base, "base_after_update"),
                        (pinned, "pinned_after_update"),
                        (latest_new, "latest_after_update"),
                    )
                ]
                base_after, pinned_after, _ = [j.result(timeout=1800) for j in jobs]
            report.check(
                "base_unchanged", base_before["tokens"] == base_after["tokens"]
            )
            compare(
                report,
                "base_logprobs_unchanged",
                base_before["logprobs"],
                base_after["logprobs"],
            )
            report.check(
                "pinned_tokens_unchanged",
                pinned_before["tokens"] == pinned_after["tokens"],
            )
            compare(
                report,
                "pinned_logprobs_unchanged",
                pinned_before["logprobs"],
                pinned_after["logprobs"],
            )
            # Wait for actual child shutdown before requesting recreation.
            until(
                lambda: not registry.get("pin_records")[pin_key].get("leases"),
                timeout=60,
            )
            records = registry.get("pin_records")
            records[pin_key]["last_used"] = time.time() - 700
            registry.put("pin_records", records)
            manage.remote("pin_demand")
            until(
                lambda: app_states().get(pin_route["app_id"], {}).get("state")
                == api_pb2.APP_STATE_STOPPED,
                timeout=180,
            )
            report.check("pinned_idle_app_stopped", True, old_app=pin_route["app_id"])
            recreated = sample(pinned, "pinned_recreated")
            new_pin_route = registry.get(pin_key)
            report.app(new_pin_route["app_id"])
            report.check(
                "pinned_recreated_with_new_app",
                new_pin_route["app_id"] != pin_route["app_id"],
            )
            report.check(
                "pinned_recreated_same_tokens",
                recreated["tokens"] == pinned_before["tokens"],
            )
            compare(
                report,
                "pinned_recreated_same_logprobs",
                recreated["logprobs"],
                pinned_before["logprobs"],
            )
            # Standard checkpoint endpoints must work with this already-saved checkpoint.
            with httpx.Client(
                base_url=url,
                timeout=90,
                trust_env=False,
                headers={"X-Api-Key": api_key},
            ) as http:
                response = http.post(
                    "/api/v1/weights_info", json={"tinker_path": checkpoint.path}
                )
                report.check(
                    "checkpoint_metadata_endpoint",
                    response.status_code == 200,
                    status=response.status_code,
                    body=response.text[:500],
                )
                response = http.post(
                    "/api/v1/load_weights",
                    json={
                        "session_id": peer.holder.get_session_id(),
                        "model_seq_id": 1000,
                        "path": checkpoint.path,
                        "optimizer": True,
                    },
                )
                # With a live trainer, a working metadata path may reject slot occupancy.
                report.check(
                    "create_from_checkpoint_reads_metadata",
                    "metadata unavailable" not in response.text
                    and response.status_code in (200, 400),
                    status=response.status_code,
                    body=response.text[:500],
                )
            # Stop the container, preserving/rescheduling its invocation: real restart fault.
            current = next(r for r in engine_records() if r["state"] == "running")
            call = modal.FunctionCall.from_id(current["call_id"])
            graph = call.get_call_graph()

            def walk(nodes):
                for node in nodes:
                    yield node
                    yield from walk(node.children)

            live_tasks = set(task_ids(app_id))
            candidates = [
                n.task_id
                for n in walk(graph)
                if n.function_call_id == current["call_id"] and n.task_id in live_tasks
            ]
            if not candidates:
                # Some Modal call graphs omit task IDs even for running inputs.
                # This recipe has exactly one four-GPU trainer; samplers use one.
                assert engine.trainer_gpu == "H100:4" and engine.sampler_gpu == "H100:1"
                for task_id in live_tasks:
                    info = rpc(
                        "TaskGetInfo", api_pb2.TaskGetInfoRequest(task_id=task_id)
                    ).info
                    if (
                        info.gpu_config.count == 4
                        and info.gpu_config.gpu_type == "H100"
                    ):
                        candidates.append(task_id)
            assert len(set(candidates)) == 1, (
                f"ambiguous trainer containers: {candidates}"
            )
            container = candidates[0]
            assert container in task_ids(app_id), (
                "refuse fault injection outside our app"
            )
            report.event(
                "restarting_trainer_container",
                container=container,
                call_id=current["call_id"],
                boot=current["boot_id"],
            )
            rpc("ContainerStop", api_pb2.ContainerStopRequest(task_id=container))
            restarted = until(
                lambda: next(
                    (
                        r
                        for r in engine_records()
                        if r["instance_id"] == current["instance_id"]
                        and r["boot_id"] != current["boot_id"]
                        and r["state"] == "running"
                    ),
                    None,
                ),
                timeout=1800,
                interval=5,
            )
            report.check(
                "same_invocation_restarted",
                restarted["call_id"] == current["call_id"],
                old_boot=current["boot_id"],
                new_boot=restarted["boot_id"],
            )
            try:
                trainer.forward([datum], "cross_entropy").result(timeout=60)
            except Exception as exc:
                report.check(
                    "old_trainer_reports_loss",
                    getattr(exc, "status_code", None) == 410,
                    error=str(exc)[:500],
                )
            else:
                report.check("old_trainer_reports_loss", False)
            replacement = None
            try:
                replacement = create_bounded(service, engine.model, timeout=90)
            except Exception as exc:
                report.check(
                    "replacement_after_container_restart", False, error=str(exc)[:500]
                )
            else:
                report.check("replacement_after_container_restart", True)
            if replacement is None:
                # Continue the independent terminal-invocation recovery test.
                call.cancel(terminate_containers=True)

                def finished():
                    try:
                        call.get(timeout=0)
                    except TimeoutError:
                        return False
                    except modal.exception.Error:
                        return True
                    return True

                until(finished, timeout=120)
                replacement = create_bounded(service, engine.model)
                report.check(
                    "replacement_after_terminal_cancellation",
                    True,
                    model_id=replacement.model_id,
                )
            replacement.load_state_with_optimizer(checkpoint.path).result(timeout=1800)
            restored_values = forward_values(
                replacement.forward([datum], "cross_entropy").result(timeout=1800)
            )
            compare(
                report,
                "checkpoint_restores_forward_logprobs",
                checkpoint_values,
                restored_values,
            )
            update(replacement, "restored_second_step")
            resumed_values = forward_values(
                replacement.forward([datum], "cross_entropy").result(timeout=1800)
            )
            compare(
                report,
                "optimizer_continuation_matches",
                continuation_values,
                resumed_values,
            )
            replacement_latest = replacement.save_weights_and_get_sampling_client()
            sample(replacement_latest, "replacement_latest")
            try:
                sample(latest_new, "unexpected_retired_latest")
            except Exception as exc:
                report.check(
                    "retired_latest_returns_410",
                    getattr(exc, "status_code", None) == 410,
                    error=str(exc)[:500],
                )
            else:
                report.check("retired_latest_returns_410", False)
            pinned_final = sample(pinned, "old_pinned_after_trainer_replacement")
            # Recovery can exceed the idle threshold and create another child.
            report.app(registry.get(pin_key)["app_id"])
            report.check(
                "old_pinned_stays_immutable",
                pinned_final["tokens"] == pinned_before["tokens"],
            )
            compare(
                report,
                "old_pinned_logprobs_after_replacement",
                pinned_final["logprobs"],
                pinned_before["logprobs"],
            )
            report.event("body_complete")
    except ScenarioComplete:
        pass
    except RuntimeError as exc:
        if args.mode == "exception" and str(exc) == "intentional-e2e-body-exception":
            report.check("body_exception_propagated", True)
        else:
            report.event(
                "fatal_error", error=repr(exc), traceback=traceback.format_exc()
            )
            report.check("scenario_completed", False)
    except BaseException as exc:
        report.event("fatal_error", error=repr(exc), traceback=traceback.format_exc())
        report.check("scenario_completed", False)
    finally:
        scoped.build_app = original_build
        if resources.get("app") and resources["app"].app_id:
            report.app(resources["app"].app_id)
        for service in services:
            if service._session_holder:
                service._session_holder.close()
        report.event("context_exited")
        wait_cleanup(report)
        report.data["finished_at"] = time.time()
        report.data["passed"] = all(c["passed"] for c in report.data["checks"])
        report.save()
    return 0 if report.data["passed"] else 1


def hard_owner_parent(args):
    worker_path = Path(args.report).with_name("hard-owner-worker.json")
    log_path = worker_path.with_suffix(".log")
    report = Report(args.report, "hard-owner-controller")
    with log_path.open("w") as log:
        child = subprocess.Popen(
            [
                sys.executable,
                __file__,
                "--mode",
                "hard-owner",
                "--report",
                str(worker_path),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:

            def ready():
                if child.poll() is not None:
                    raise RuntimeError(
                        f"owner exited early: {child.returncode}; see {log_path}"
                    )
                if not worker_path.exists():
                    return False
                data = json.loads(worker_path.read_text())
                return (
                    data
                    if any(e["event"] == "ready_for_owner_kill" for e in data["events"])
                    else False
                )

            data = until(ready, timeout=600)
            for app_id in data["apps"]:
                report.app(app_id)
            report.data["checks"].extend(data["checks"])
            report.event("kill_owner_without_cleanup", pid=child.pid)
            child.kill()
            child.wait(timeout=15)
            wait_cleanup(report, timeout=240)
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=150)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=15)
            report.data["finished_at"] = time.time()
            report.data["passed"] = bool(report.data["checks"]) and all(
                c["passed"] for c in report.data["checks"]
            )
            report.save()
    return 0 if report.data["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "gpu",
            "exception",
            "hard-owner",
            "hard-owner-controller",
            "isolation",
        ),
        default="gpu",
    )
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--asset-path", help="Optional isolated mounted HF checkpoint directory"
    )
    args = parser.parse_args()
    if not os.environ.get("MODAL_ENVIRONMENT"):
        parser.error("set MODAL_ENVIRONMENT explicitly")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    sys.exit(
        hard_owner_parent(args)
        if args.mode == "hard-owner-controller"
        else run_worker(args)
    )
