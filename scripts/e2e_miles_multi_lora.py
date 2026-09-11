"""Two concurrent Tinker clients sharing one Miles trainer and Lilo publication path.

See docs/multi-lora-e2e.md. No direct Miles or SGLang calls perform training,
publication, or sampling; Modal state is read only to assert placement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from decimal import Decimal, InvalidOperation
from pathlib import Path

import httpx
import tinker
from tinker import types

DEFAULT_DEFINITION = "qwen3_5_9b_base_miles_lora_16k"
TIMEOUT = 15 * 60


def progress(phase, **details):
    print(json.dumps({"event": phase, "time": time.time(), **details}), flush=True)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parallel(functions):
    # Submit every operation before waiting; both clients stay alive throughout.
    with ThreadPoolExecutor(max_workers=len(functions)) as pool:
        futures = [pool.submit(fn) for fn in functions]
        return [future.result() for future in futures]


def load_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    require(bool(rows), f"empty dataset: {path}")
    for row in rows:
        require(
            isinstance(row.get("prompt"), str) and bool(row["prompt"]),
            "JSONL rows require a nonempty string prompt",
        )
        require(
            isinstance(row.get("answer"), (str, int)),
            "JSONL rows require a string or integer answer",
        )
        require(
            numeric_answer(str(row["answer"])) is not None,
            "This smoke test requires numeric answers; see the documented dataset conversion",
        )
    return rows


def numeric_answer(text):
    # Deliberately conservative exact numeric grading, not symbolic math grading.
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if matches:
        text = matches[-1]
    elif "####" in text:
        text = text.rsplit("####", 1)[1]
    else:
        text = text.strip()
    try:
        value = Decimal(text.strip().replace(",", "").strip("$"))
        return value if value.is_finite() else None
    except InvalidOperation:
        return None


def make_datum(prompt, completion, *, logprobs=None, advantage=None):
    require(bool(prompt) and bool(completion), "prompt and completion must be nonempty")
    tokens = prompt + completion
    prefix = len(prompt) - 1
    inputs = {"target_tokens": tokens[1:]}
    if logprobs is None:
        inputs["weights"] = [0.0] * prefix + [1.0] * len(completion)
    else:
        require(len(logprobs) == len(completion), "token/logprob length mismatch")
        require(all(math.isfinite(x) for x in logprobs), "nonfinite sampling logprobs")
        # Mask prompt loss with zero advantages, never dummy target IDs: Miles
        # validates the shifted target sequence even at masked positions.
        inputs["logprobs"] = [0.0] * prefix + list(logprobs)
        inputs["advantages"] = [0.0] * prefix + [advantage] * len(completion)
    return types.Datum(model_input=types.ModelInput.from_ints(tokens[:-1]), loss_fn_inputs=inputs)


def finite_metrics(result):
    values = {k: float(v) for k, v in result.metrics.items() if isinstance(v, (float, int))}
    require(
        bool(values) and all(math.isfinite(v) for v in values.values()),
        f"invalid metrics: {values}",
    )
    return values


def logprobs(training, datum):
    result = training.forward([datum], "cross_entropy").result(timeout=TIMEOUT)
    require(len(result.loss_fn_outputs) == 1, "missing forward result")
    values = list(result.loss_fn_outputs[0]["logprobs"].data)
    require(len(values) == len(datum.model_input.to_ints()), "forward length mismatch")
    require(all(math.isfinite(v) for v in values), "nonfinite trainer logprobs")
    return values


def max_error(left, right):
    require(len(left) == len(right) and bool(left), "logprob shape mismatch")
    require(all(math.isfinite(v) for v in [*left, *right]), "nonfinite logprobs")
    return max(abs(a - b) for a, b in zip(left, right, strict=True))


def train_step(training, data, loss_fn, learning_rate, *, response_lengths=None):
    start = time.monotonic()
    progress("train_start", model_id=training.model_id, sequences=len(data), loss_fn=loss_fn)
    forward = training.forward_backward(data, loss_fn)
    optimizer = training.optim_step(types.AdamParams(learning_rate=learning_rate))
    result = forward.result(timeout=TIMEOUT)
    require(len(result.loss_fn_outputs) == len(data), "missing batch outputs")
    for datum, output in zip(data, result.loss_fn_outputs, strict=True):
        values = output["logprobs"].data
        require(len(values) == len(datum.model_input.to_ints()), "training length mismatch")
        require(all(math.isfinite(v) for v in values), "nonfinite training logprobs")
    diagnostics = {}
    if response_lengths is not None:
        differences = []
        for datum, output, length in zip(data, result.loss_fn_outputs, response_lengths, strict=True):
            trainer = list(output["logprobs"].data)
            inference = list(datum.loss_fn_inputs["logprobs"].data)
            require(0 < length <= len(trainer), "invalid response length for parity")
            require(len(trainer) == len(inference), "parity token alignment mismatch")
            differences.extend(a - b for a, b in zip(trainer[-length:], inference[-length:], strict=True))
        require(bool(differences), "missing response logprobs for parity")
        absolute = sorted(abs(value) for value in differences)
        diagnostics["rollout_trainer_parity"] = {
            "tokens": len(differences),
            "mean_signed": math.fsum(differences) / len(differences),
            "mean_abs": math.fsum(absolute) / len(absolute),
            "p95_abs": absolute[math.ceil(0.95 * len(absolute)) - 1],
            "max_abs": absolute[-1],
        }
    return {
        **diagnostics,
        "forward": finite_metrics(result),
        "optimizer": finite_metrics(optimizer.result(timeout=TIMEOUT)),
        "seconds": time.monotonic() - start,
    }


class PlacementProbe:
    def __init__(self, app_id, definition_id):
        from lilo.providers.modal.kv import STORE_NAMES, app_store_name

        import modal

        self.definition_id = definition_id
        self.stores = {
            domain: modal.Dict.from_name(app_store_name(name, app_id)) for domain, name in STORE_NAMES.items()
        }
        self.identity = None

    def shared_engine(self, model_ids):
        from lilo.control_plane.keys import placement_key
        from lilo.providers.modal.engines import instance_key

        placements = [self.stores["models"].get(placement_key(mid)) for mid in model_ids]
        identity = validate_placements(placements, model_ids, self.definition_id)
        if self.identity is not None:
            require(
                identity == self.identity,
                "trainer instance or boot changed during test",
            )
        self.identity = identity
        record = self.stores["engines"].get(instance_key(identity[0]))
        require(record is not None and record["state"] == "running", "trainer not running")
        require(record["boot_id"] == identity[1], "stale engine placement")
        # Do not serialize the engine record: it contains a private URL and token.
        headers = {"Authorization": f"Bearer {record['token']}"}
        response = httpx.get(record["url"].rstrip("/") + "/api/v1/models", headers=headers, timeout=60)
        response.raise_for_status()
        resident = response.json()["model_ids"]
        require(set(model_ids) <= set(resident), "clients not resident in shared trainer")
        return {
            "engine_instance_id": identity[0],
            "engine_boot_id": identity[1],
            "model_ids": model_ids,
            "resident_model_ids": resident,
        }

    def artifact(self, path, model_id):
        from lilo.control_plane.keys import sampler_artifact_key

        record = self.stores["artifacts"].get(sampler_artifact_key(path))
        require(
            record is not None and record["model_id"] == model_id,
            "sampler artifact does not belong to requested adapter",
        )
        require(
            record["engine_definition_id"] == self.definition_id,
            "sampler artifact uses wrong definition",
        )
        return {
            "path": path,
            "publish_version": record["publish_version"],
            "model_id": model_id,
        }


def validate_placements(placements, model_ids, definition_id):
    require(len(model_ids) == len(set(model_ids)), "expected distinct model IDs")
    require(len(placements) == len(model_ids) and bool(placements), "missing placements")
    for placement, mid in zip(placements, model_ids, strict=True):
        require(
            placement is not None and placement["model_id"] == mid,
            f"missing placement for {mid}",
        )
        require(
            placement["engine_definition_id"] == definition_id,
            "wrong trainer definition",
        )
        require(bool(placement["engine_boot_id"]), "missing trainer boot identity")
    identities = {(p["engine_instance_id"], p["engine_boot_id"]) for p in placements}
    require(
        len(identities) == 1,
        "adapters were placed on different trainer instances/boots",
    )
    return next(iter(identities))


def publish(training, probe, name):
    progress("publish_start", model_id=training.model_id, name=name)
    saved = training.save_weights_for_sampler(name).result(timeout=TIMEOUT)
    artifact = probe.artifact(saved.path, training.model_id)
    progress("publish_complete", **artifact)
    return training.create_sampling_client(saved.path), artifact


def sampler_logprobs(sampling, tokens):
    values = sampling.compute_logprobs(types.ModelInput.from_ints(tokens)).result(timeout=TIMEOUT)
    require(len(values) == len(tokens), "sampler prompt logprob length mismatch")
    require(all(v is not None for v in values[1:]), "missing sampler prompt logprobs")
    return [float(v) for v in values[1:]]


def snapshot_files(module, artifact):
    from stitch.types import VersionRef

    import modal

    volume = modal.Volume.from_name(module.BULLETIN_VOLUME_NAME, version=2)
    ref = VersionRef(artifact["model_id"], artifact["publish_version"])
    manifest = json.loads(b"".join(volume.read_file(f"{ref.identity}/snapshot.json")))
    require(manifest["ref"] == ref.identity, "snapshot manifest identity mismatch")
    return manifest["files"]


def unload(base_url, model_id):
    with httpx.Client(
        base_url=base_url,
        headers={"X-API-Key": os.environ["TINKER_API_KEY"]},
        timeout=60,
    ) as client:
        response = client.post("/api/v1/unload_model", json={"model_id": model_id})
        response.raise_for_status()
        request_id = response.json()["request_id"]
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            response = client.post("/api/v1/retrieve_future", json={"request_id": request_id})
            if response.status_code != 408:
                response.raise_for_status()
                require("error" not in response.json(), "unload operation failed")
                return
            time.sleep(1)
        raise TimeoutError(f"unloading {model_id}")


def run(args, module, base_url, app_id, output):
    rows = [load_rows(args.gsm8k), load_rows(args.dapo)]
    report = {
        "status": "running",
        "definition_id": args.definition_id,
        "app_id": app_id,
        "ranks": [16, 32],
        "steps": [],
        "artifacts": [],
        "settings": {
            "groups": args.groups,
            "samples": args.samples,
            "max_new_tokens": args.max_new_tokens,
            "lr": args.lr,
        },
        "datasets": [
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in (args.gsm8k, args.dapo)
        ],
    }
    clients = []
    live = []
    probe = PlacementProbe(app_id, args.definition_id)
    try:
        service = tinker.ServiceClient(base_url=base_url, api_key=os.environ["TINKER_API_KEY"])
        # Creation is sequential to retain handles for cleanup if the second
        # fails. Both models remain live; rollout/training/publication run concurrently.
        write_report(output, report)
        for rank in [16, 32]:
            progress("create_client_start", rank=rank)
            client = service.create_lora_training_client(base_model=module.MODEL_NAME, rank=rank)
            clients.append(client)
            live.append(client.model_id)
            progress("create_client_complete", rank=rank, model_id=client.model_id)
        tokenizer = clients[0].get_tokenizer()

        def encode(row):
            return tokenizer.encode(
                row["prompt"] + "\nGive the final answer in \\boxed{}.\nAnswer:",
                add_special_tokens=True,
            )

        prompts = [[encode(row) for row in dataset] for dataset in rows]
        require(
            all(len(p) + args.max_new_tokens < module.MAX_CONTEXT_LENGTH for dataset in prompts for p in dataset),
            "dataset prompt plus generation exceeds context limit; prefilter long rows",
        )
        completion = tokenizer.encode(" 42", add_special_tokens=False)
        probe_tokens = prompts[0][0] + completion
        report["probe_tokens"] = probe_tokens
        datum = make_datum(prompts[0][0], completion)
        # Prime backward kernels before measuring isolation. The first backward
        # changed BF16 forward numerics despite byte-identical idle weights.
        # Zero loss and LR preserve parameters and optimizer moments; both
        # optimizer step counters advance once before the supervised warmups.
        zero_datum = types.Datum(
            model_input=datum.model_input,
            loss_fn_inputs={
                "target_tokens": datum.loss_fn_inputs["target_tokens"],
                "weights": [0.0] * len(datum.model_input.to_ints()),
            },
        )
        report["kernel_warmup"] = []
        for client in clients:
            report["kernel_warmup"].append(train_step(client, [zero_datum], "cross_entropy", 0.0))
            require(
                report["kernel_warmup"][-1]["optimizer"]["grad_norm:mean"] == 0.0,
                "zero-loss warmup produced nonzero gradients",
            )
            write_report(output, report)
        # A real forward establishes residency before reading placement records.
        progress("initial_forward_start")
        before = parallel([lambda c=c: logprobs(c, datum) for c in clients])
        repeated = parallel([lambda c=c: logprobs(c, datum) for c in clients])
        report["baseline_logprobs"] = before
        report["baseline_repeat_errors"] = [max_error(a, b) for a, b in zip(before, repeated, strict=True)]
        progress("baseline_repeat", errors=report["baseline_repeat_errors"])
        report["placement"] = probe.shared_engine(live.copy())
        progress("shared_trainer_verified", **report["placement"])
        write_report(output, report)
        initial = parallel([lambda c=c: publish(c, probe, "initial") for c in clients])
        report["artifacts"].append([item[1] for item in initial])
        progress("initial_sampler_start")
        initial_sampler = parallel([lambda s=s: sampler_logprobs(s, probe_tokens) for s, _ in initial])
        report["initial_sampler_logprobs"] = initial_sampler
        report["initial_parity_errors"] = [max_error(a, b) for a, b in zip(before, initial_sampler, strict=True)]
        progress("initial_parity", errors=report["initial_parity_errors"])
        write_report(output, report)
        require(
            all(e <= args.parity_atol for e in report["initial_parity_errors"]),
            "initial trainer/sampler parity failed",
        )

        # A-only update tests that B is unaffected and the published weights change.
        report["warmup_a"] = train_step(clients[0], [datum], "cross_entropy", args.lr)
        after = parallel([lambda c=c: logprobs(c, datum) for c in clients])
        change = max_error(before[0], after[0])
        isolation = max_error(before[1], after[1])
        report["after_a_logprobs"] = after
        report["isolation"] = {"a_change": change, "b_max_error": isolation}
        progress("isolation_check", **report["isolation"])
        write_report(output, report)
        diagnostic = publish(clients[1], probe, "isolation-diagnostic")
        report["isolation"]["b_files_before"] = snapshot_files(module, initial[1][1])
        report["isolation"]["b_files_after"] = snapshot_files(module, diagnostic[1])
        write_report(output, report)
        require(
            report["isolation"]["b_files_before"] == report["isolation"]["b_files_after"],
            "updating A changed B's exported adapter weights",
        )
        if isolation > args.isolation_atol:
            report["isolation"]["b_repeat_logprobs"] = logprobs(clients[1], datum)
            report["isolation"]["b_sampler_error"] = max_error(
                initial_sampler[1], sampler_logprobs(diagnostic[0], probe_tokens)
            )
            write_report(output, report)
        require(change > args.change_min, "A warmup did not produce a measurable update")
        require(isolation <= args.isolation_atol, "updating A changed idle B")
        report["warmup_b"] = train_step(clients[1], [datum], "cross_entropy", args.lr)
        report["diagnostic_parity"] = []
        for update in range(getattr(args, "diagnostic_updates", 0)):
            if update:
                parallel([lambda c=c: train_step(c, [datum], "cross_entropy", args.lr) for c in clients])
            publications = parallel([lambda c=c: publish(c, probe, f"diagnostic-{update}") for c in clients])
            trainer_lp = parallel([lambda c=c: logprobs(c, datum) for c in clients])
            inference_lp = parallel([lambda s=s: sampler_logprobs(s, probe_tokens) for s, _ in publications])
            errors = [max_error(a, b) for a, b in zip(trainer_lp, inference_lp, strict=True)]
            report["diagnostic_parity"].append(
                {
                    "supervised_updates": update + 1,
                    "artifacts": [artifact for _, artifact in publications],
                    "trainer_logprobs": trainer_lp,
                    "inference_logprobs": inference_lp,
                    "max_abs_errors": errors,
                }
            )
            write_report(output, report)
            progress("diagnostic_parity", supervised_updates=update + 1, errors=errors)
            require(
                all(e <= args.parity_atol for e in errors),
                "diagnostic trainer/sampler parity failed",
            )
        previous_versions = [item[1]["publish_version"] for item in initial]
        for step in range(args.steps):
            progress("rl_step_start", step=step)
            start = time.monotonic()
            publications = parallel([lambda c=c: publish(c, probe, f"step-{step}") for c in clients])
            artifacts = [item[1] for item in publications]
            for i, artifact in enumerate(artifacts):
                require(
                    artifact["publish_version"] > previous_versions[i],
                    "version did not advance",
                )
                previous_versions[i] = artifact["publish_version"]
            report["artifacts"].append(artifacts)

            def rollout(i):
                sampling = publications[i][0]

                def rollout_group(group):
                    index = (step * args.groups + group) % len(rows[i])
                    prompt = prompts[i][index]
                    progress("rollout_group_start", adapter=i, step=step, group=group)
                    result = sampling.sample(
                        prompt=types.ModelInput.from_ints(prompt),
                        num_samples=args.samples,
                        sampling_params=types.SamplingParams(max_tokens=args.max_new_tokens, temperature=1.0),
                    ).result(timeout=TIMEOUT)
                    require(len(result.sequences) == args.samples, "missing rollout samples")
                    target = numeric_answer(str(rows[i][index]["answer"]))
                    group_rewards = [
                        float(numeric_answer(tokenizer.decode(s.tokens)) == target) for s in result.sequences
                    ]
                    mean = sum(group_rewards) / len(group_rewards)
                    advantages = [r - mean for r in group_rewards]
                    data = []
                    for sequence, advantage in zip(result.sequences, advantages, strict=True):
                        data.append(
                            make_datum(
                                prompt,
                                list(sequence.tokens),
                                logprobs=list(sequence.logprobs or []),
                                advantage=advantage,
                            )
                        )
                    progress(
                        "rollout_group_complete",
                        adapter=i,
                        step=step,
                        group=group,
                        rewards=group_rewards,
                        lengths=[len(s.tokens) for s in result.sequences],
                        stop_reasons=[s.stop_reason for s in result.sequences],
                    )
                    return (
                        data,
                        group_rewards,
                        int(any(a != 0 for a in advantages)),
                        [len(s.tokens) for s in result.sequences],
                    )

                data, rewards, nonzero, response_lengths = [], [], 0, []
                # Keep multiple prompt groups in flight so one long response
                # does not leave the rollout pool waiting for the next group.
                with ThreadPoolExecutor(max_workers=min(4, args.groups)) as pool:
                    for group_data, group_rewards, active, lengths in pool.map(rollout_group, range(args.groups)):
                        data.extend(group_data)
                        rewards.extend(group_rewards)
                        nonzero += active
                        response_lengths.extend(lengths)
                return data, {
                    "reward_mean": sum(rewards) / len(rewards),
                    "nonzero_advantage_groups": nonzero,
                    "samples": len(rewards),
                    "response_lengths": response_lengths,
                }

            rollouts = parallel([lambda i=i: rollout(i) for i in range(2)])
            trained = parallel(
                [
                    lambda i=i: train_step(
                        clients[i],
                        rollouts[i][0],
                        "importance_sampling",
                        args.lr,
                        response_lengths=rollouts[i][1]["response_lengths"],
                    )
                    for i in range(2)
                ]
            )
            report["steps"].append(
                {
                    "step": step,
                    "seconds": time.monotonic() - start,
                    "adapters": [{**rollouts[i][1], **trained[i]} for i in range(2)],
                    "placement": probe.shared_engine(live.copy()),
                }
            )
            write_report(output, report)
            from plot_miles_multi_lora import plot_report

            plot_report(output)
            print(json.dumps(report["steps"][-1]), flush=True)

        final = parallel([lambda c=c: publish(c, probe, "final") for c in clients])
        report["artifacts"].append([item[1] for item in final])
        for i, (_, artifact) in enumerate(final):
            require(
                artifact["publish_version"] > previous_versions[i],
                "final publication version did not advance",
            )
        report["final_placement"] = probe.shared_engine(live.copy())
        trained_lp = parallel([lambda c=c: logprobs(c, datum) for c in clients])
        served_lp = parallel([lambda s=s: sampler_logprobs(s, probe_tokens) for s, _ in final])
        errors = [max_error(a, b) for a, b in zip(trained_lp, served_lp, strict=True)]
        report["final_parity"] = {
            "trainer_logprobs": trained_lp,
            "inference_logprobs": served_lp,
            "max_abs_errors": errors,
        }
        write_report(output, report)
        progress("final_parity", errors=errors)
        require(
            all(e <= args.parity_atol for e in errors),
            "final trainer/sampler parity failed",
        )
        for i in range(2):
            require(
                max_error(served_lp[i], initial_sampler[i]) > args.change_min,
                f"sampler {i} still appears to serve initial weights",
            )
        old_lp = sampler_logprobs(initial[0][0], probe_tokens)
        old_error = max_error(old_lp, initial_sampler[0])
        require(old_error <= args.isolation_atol, "immutable initial snapshot changed")
        report["publication_checks"] = {
            "trainer_sampler_errors": errors,
            "old_snapshot_error": old_error,
        }
        unload(base_url, clients[0].model_id)
        live.remove(clients[0].model_id)
        survivor_error = max_error(logprobs(clients[1], datum), trained_lp[1])
        require(survivor_error <= args.isolation_atol, "unloading A changed B")
        sampler_logprobs(final[1][0], probe_tokens)
        report["survivor"] = probe.shared_engine(live.copy())
        require(
            clients[0].model_id not in report["survivor"]["resident_model_ids"],
            "unloaded adapter remains resident in trainer",
        )
        report["status"] = "passed"
    except BaseException as exc:
        progress("run_failed", error=f"{type(exc).__name__}: {exc}")
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["cleanup_errors"] = []
        for mid in live:
            try:
                unload(base_url, mid)
            except Exception as exc:
                report["cleanup_errors"].append(f"{mid}: {type(exc).__name__}: {exc}")
        if report["cleanup_errors"] and report["status"] == "passed":
            report["status"] = "failed"
        write_report(output, report)
    require(report["status"] == "passed", "cleanup failed; see report")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsm8k", type=Path, required=True)
    parser.add_argument("--dapo", type=Path, required=True)
    parser.add_argument("--definition-id", default=DEFAULT_DEFINITION)
    parser.add_argument("--base-url")
    parser.add_argument("--app-id", help="Required with --base-url for placement assertions")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument(
        "--diagnostic-updates",
        type=int,
        default=0,
        help="Check parity after controlled supervised updates before RL",
    )
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--parity-atol", type=float, default=0.2)
    parser.add_argument("--isolation-atol", type=float, default=1e-5)
    parser.add_argument("--change-min", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, default=Path("scripts/results/miles_multi_lora.json"))
    args = parser.parse_args()
    if bool(args.base_url) != bool(args.app_id):
        parser.error("--base-url and --app-id must be provided together")
    if (
        args.steps < 0
        or args.diagnostic_updates < 0
        or (args.steps == 0 and args.diagnostic_updates == 0)
        or min(args.groups, args.samples, args.max_new_tokens) < 1
        or args.samples < 2
    ):
        parser.error("positive sizes and at least two samples per group are required")
    if not all(math.isfinite(v) and v > 0 for v in (args.lr, args.parity_atol, args.isolation_atol, args.change_min)):
        parser.error("learning rate and tolerances must be finite and positive")
    # Validate input files before launching any GPU infrastructure.
    load_rows(args.gsm8k)
    load_rows(args.dapo)
    require(bool(os.environ.get("TINKER_API_KEY")), "TINKER_API_KEY is required")
    if args.base_url is None:
        # Must be set before importing app/definitions (decorator-time limit).
        os.environ["LILO_TRAINER_MAX_CONTAINERS"] = "1"
    from lilo.providers.modal.app import app, module_for, server

    import modal

    module = module_for(args.definition_id)
    require(
        module.PARAMETERIZATION == "lora" and "miles" in module.DEFINITION_ID,
        "test requires a Miles LoRA definition",
    )
    require(
        module.MAX_LORA_SLOTS >= 2 and module.MAX_LORA_RANK >= 32,
        "definition must support two slots and rank 32",
    )
    output = args.output.with_name(f"{args.output.stem}.{uuid.uuid4().hex[:10]}.json")
    context = nullcontext() if args.base_url else app.run(name=f"multi-lora-e2e-{uuid.uuid4().hex[:8]}")
    with modal.enable_output(), context:
        base_url = args.base_url or server.get_web_url()
        app_id = args.app_id or app.app_id
        require(bool(base_url) and bool(app_id), "missing control-plane URL or app ID")
        try:
            progress("app_ready", app_id=app_id, base_url=base_url, output=str(output))
            report = run(args, module, base_url, app_id, output)
        finally:
            print(f"Report: {output}", flush=True)
    print(report["status"])


if __name__ == "__main__":
    main()
