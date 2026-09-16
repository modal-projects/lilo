"""Real same-scope trainer restart and SDK checkpoint recovery.

Python 3.12, PYTHONPATH=src:scripts, MODAL_ENVIRONMENT must be set.
Uses four H100s; targets only the app created here. No cancellation fallback.
"""

import argparse
import hashlib
import math
import os
from pathlib import Path
import subprocess
import time
import traceback
import uuid

import modal
import httpx
from modal_proto import api_pb2
import tinker
from tinker import types

import lilo
from lilo.engines import qwen3_5_4b_full_64k
from scoped_e2e import (
    Report,
    app_states,
    compare,
    forward_values,
    rpc,
    task_ids,
    until,
    wait_cleanup,
)


def main(output):
    import lilo.providers.modal.scoped as scoped

    report = Report(output, "same-scope-restart")
    report.data["revision"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    report.data["source_hashes"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path("src/lilo/providers/modal").glob("*.py")
    }
    report.save()
    resources = {}
    service = None
    original_build = scoped.build_app

    def capture(*args, **kwargs):
        result = original_build(*args, **kwargs)
        resources.update(app=result[0], registry_name=args[2], manage=result[2])
        return result

    scoped.build_app = capture

    def check(name, condition, **values):
        report.check(name, condition, **values)
        assert condition, name

    try:
        engine = qwen3_5_4b_full_64k()
        with (
            modal.enable_output(),
            lilo.run(
                engine=engine,
                name="lilo-e2e-release-" + uuid.uuid4().hex[:6],
                latest=lilo.Pool(min_containers=0, max_containers=1),
            ) as (url, api_key),
        ):
            app_id = resources["app"].app_id
            report.app(app_id)
            report.event("ready", app_id=app_id, url=url)
            registry = modal.Dict.from_name(resources["registry_name"])
            engines = modal.Dict.from_name(app_id + "-engines")

            def running():
                return [
                    v
                    for k, v in engines.items()
                    if k.startswith("engine_instance:") and v["state"] == "running"
                ]

            with httpx.Client(timeout=90) as http:
                denied = http.post(url + "/api/v1/create_session", headers={"x-api-key": "incorrect"}, json={})
                check("wrong_api_key_rejected", denied.status_code == 401)
            service = tinker.ServiceClient(base_url=url, api_key=api_key)
            rest = service.create_rest_client()
            first = lilo.create_full_training_client(service, engine.model)
            report.event("first_client", model_id=first.model_id)
            tokens = first.get_tokenizer().encode(
                "The capital of France is Paris. The capital of Italy is Rome."
            )
            datum = types.Datum(
                model_input=types.ModelInput.from_ints(tokens[:-1]),
                loss_fn_inputs={
                    "target_tokens": tokens[1:],
                    "weights": [1.0] * (len(tokens) - 1),
                },
            )

            def forward(client):
                return forward_values(
                    client.forward([datum], "cross_entropy").result(timeout=1800)
                )

            def update(client, name):
                result = client.forward_backward([datum], "cross_entropy").result(
                    timeout=1800
                )
                check(
                    name + "_finite_loss",
                    all(
                        math.isfinite(v)
                        for v in result.metrics.values()
                        if isinstance(v, (float, int))
                    ),
                    metrics=result.metrics,
                )
                optim = client.optim_step(types.AdamParams(learning_rate=1e-5)).result(
                    timeout=1800
                )
                check(
                    name + "_optimizer",
                    optim.metrics.get("update_successful:mean") == 1.0,
                    metrics=optim.metrics,
                )

            update(first, "first_step")
            checkpoint = first.save_state("same-scope-recovery").result(timeout=1800)
            report.event("checkpoint_saved", path=checkpoint.path)
            expected = forward(first)
            info = rest.get_weights_info_by_tinker_path(checkpoint.path).result(
                timeout=120
            )
            check(
                "sdk_checkpoint_metadata",
                info.base_model == engine.model and not info.is_lora,
            )
            entries = (
                rest.list_checkpoints(first.model_id).result(timeout=120).checkpoints
            )
            check(
                "sdk_checkpoint_listing",
                any(e.tinker_path == checkpoint.path for e in entries),
            )

            tokenizer = first.get_tokenizer()
            prompt = types.ModelInput.from_ints(tokenizer.encode("The capital of France is"))
            def sample(client, label):
                seq = client.sample(prompt=prompt, num_samples=1,
                    sampling_params=types.SamplingParams(max_tokens=16, temperature=0)).result(timeout=1800).sequences[0]
                check(label, bool(seq.tokens) and len(seq.logprobs) == len(seq.tokens)
                    and all(math.isfinite(x) for x in seq.logprobs), tokens=list(seq.tokens))
                return list(seq.tokens)
            sample(service.create_sampling_client(base_model=engine.model), "base_sampling")
            sample(first.save_weights_and_get_sampling_client(), "latest_sampling")
            saved = first.save_weights_for_sampler("release-pinned").result(timeout=1800)
            pinned = service.create_sampling_client(model_path=saved.path)
            pinned_tokens = sample(pinned, "pinned_sampling")
            pin_key = next(k for k, _ in registry.items() if k.startswith("pinned:" + first.model_id + ":"))
            route = registry.get(pin_key)
            report.app(route["app_id"])
            until(lambda: not registry.get("pin_records")[pin_key].get("leases"), timeout=60)
            pins = registry.get("pin_records")
            pins[pin_key]["last_used"] = time.time() - 700
            registry.put("pin_records", pins)
            resources["manage"].remote("pin_demand")
            until(lambda: app_states().get(route["app_id"], {}).get("state") == api_pb2.APP_STATE_STOPPED, timeout=180)
            check("pinned_idle_cleanup", True)
            check("pinned_recreation_preserves_tokens", sample(pinned, "pinned_recreated") == pinned_tokens)
            new_route = registry.get(pin_key)
            report.app(new_route["app_id"])
            check("pinned_recreated_new_app", new_route["app_id"] != route["app_id"])
            records = running()
            check("one_trainer_before_restart", len(records) == 1)
            original = records[0]
            # Select the one four-H100 trainer within this app, never another run.
            candidates = []
            for task_id in task_ids(app_id):
                task = rpc(
                    "TaskGetInfo", api_pb2.TaskGetInfoRequest(task_id=task_id)
                ).info
                if task.gpu_config.count == 4 and task.gpu_config.gpu_type == "H100":
                    candidates.append(task_id)
            check(
                "fault_target_unambiguous", len(candidates) == 1, containers=candidates
            )
            target = candidates[0]
            assert target in task_ids(app_id)
            report.event(
                "container_stop",
                container=target,
                call_id=original["call_id"],
                boot=original["boot_id"],
            )
            rpc("ContainerStop", api_pb2.ContainerStopRequest(task_id=target))
            restarted = until(
                lambda: next(
                    (
                        r
                        for r in running()
                        if r["instance_id"] == original["instance_id"]
                        and r["boot_id"] != original["boot_id"]
                    ),
                    None,
                ),
                timeout=1800,
                interval=5,
            )
            check(
                "same_invocation_new_boot",
                restarted["call_id"] == original["call_id"],
                old_boot=original["boot_id"],
                new_boot=restarted["boot_id"],
            )

            # Application recovery is triggered by the normal request's error.
            try:
                first.forward([datum], "cross_entropy").result(timeout=120)
            except tinker.APIStatusError as exc:
                body = exc.response.json()
                check(
                    "old_client_model_lost",
                    exc.status_code == 410 and body.get("error") == "model_lost",
                    status=exc.status_code,
                    body=body,
                )
                report.event("creating_second_client_from_checkpoint")
                second = service.create_training_client_from_state_with_optimizer(
                    checkpoint.path
                )
            else:
                raise AssertionError("old client unexpectedly survived container loss")

            check(
                "second_client_created_in_same_scope",
                second.model_id != first.model_id,
                model_id=second.model_id,
                app_id=app_id,
            )
            records = running()
            check(
                "reused_only_trainer",
                len(records) == 1
                and records[0]["call_id"] == original["call_id"]
                and records[0]["boot_id"] == restarted["boot_id"],
            )
            check(
                "slot_transferred",
                registry.get("slot:0") == second.model_id
                and registry.get("retired:" + first.model_id) is True,
            )
            restored = forward(second)
            compare(report, "checkpoint_restores_forward", expected, restored)
            update(second, "resumed_step_one")
            after = forward(second)
            check(
                "resumed_update_changes_model",
                any(abs(a - b) > 1e-6 for a, b in zip(restored, after)),
                max_abs_change=max(abs(a - b) for a, b in zip(restored, after)),
            )
            update(second, "resumed_step_two")
            check(
                "continued_forward_finite",
                all(math.isfinite(x) for x in forward(second)),
            )
            try:
                first.forward([datum], "cross_entropy").result(timeout=120)
            except tinker.APIStatusError as exc:
                check(
                    "old_client_stays_lost",
                    exc.status_code == 410,
                    body=exc.response.json(),
                )
            else:
                raise AssertionError("old client was resurrected")
            sample(second.save_weights_and_get_sampling_client(), "replacement_latest_sampling")
            check("old_pinned_survives_replacement", sample(pinned, "pinned_after_replacement") == pinned_tokens)
            report.app(registry.get(pin_key)["app_id"])
            # Exercise the public delete callback and avoid leaving an 82-GB checkpoint.
            rest.delete_checkpoint_from_tinker_path(checkpoint.path).result(timeout=180)
            entries = (
                rest.list_checkpoints(first.model_id).result(timeout=120).checkpoints
            )
            check(
                "sdk_checkpoint_deleted",
                all(e.tinker_path != checkpoint.path for e in entries),
            )
            report.event("body_complete")
    except BaseException as exc:
        report.event("fatal_error", error=repr(exc), traceback=traceback.format_exc())
        report.check("scenario_completed", False)
    finally:
        scoped.build_app = original_build
        if resources.get("app") and resources["app"].app_id:
            report.app(resources["app"].app_id)
        if service and service._session_holder:
            service._session_holder.close()
        report.event("context_exited")
        wait_cleanup(report)
        report.data["finished_at"] = time.time()
        report.data["passed"] = bool(report.data["checks"]) and all(
            c["passed"] for c in report.data["checks"]
        )
        report.save()
    return 0 if report.data["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if not os.environ.get("MODAL_ENVIRONMENT"):
        parser.error("set MODAL_ENVIRONMENT explicitly")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    raise SystemExit(main(args.report))
