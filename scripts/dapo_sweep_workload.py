"""DAPO sampling, grading, training validation and placement checks for the sweep."""

import json
import math
import os
import re
import time
from decimal import Decimal, InvalidOperation
import httpx
from tinker import types

TIMEOUT = 3600


def progress(phase, **details):
    print(json.dumps({"event": phase, "time": time.time(), **details}), flush=True)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


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
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]), loss_fn_inputs=inputs
    )


def finite_metrics(result):
    values = {
        k: float(v) for k, v in result.metrics.items() if isinstance(v, (float, int))
    }
    require(
        bool(values) and all(math.isfinite(v) for v in values.values()),
        f"invalid metrics: {values}",
    )
    return values


def train_step(training, data, loss_fn, learning_rate, *, response_lengths=None):
    start = time.monotonic()
    progress(
        "train_start", model_id=training.model_id, sequences=len(data), loss_fn=loss_fn
    )
    forward = training.forward_backward(data, loss_fn)
    optimizer = training.optim_step(types.AdamParams(learning_rate=learning_rate))
    result = forward.result(timeout=TIMEOUT)
    require(len(result.loss_fn_outputs) == len(data), "missing batch outputs")
    for datum, output in zip(data, result.loss_fn_outputs, strict=True):
        values = output["logprobs"].data
        require(
            len(values) == len(datum.model_input.to_ints()), "training length mismatch"
        )
        require(all(math.isfinite(v) for v in values), "nonfinite training logprobs")
    diagnostics = {}
    if response_lengths is not None:
        differences = []
        for datum, output, length in zip(
            data, result.loss_fn_outputs, response_lengths, strict=True
        ):
            trainer = list(output["logprobs"].data)
            inference = list(datum.loss_fn_inputs["logprobs"].data)
            require(0 < length <= len(trainer), "invalid response length for parity")
            require(len(trainer) == len(inference), "parity token alignment mismatch")
            differences.extend(
                a - b
                for a, b in zip(trainer[-length:], inference[-length:], strict=True)
            )
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
            domain: modal.Dict.from_name(app_store_name(name, app_id))
            for domain, name in STORE_NAMES.items()
        }
        self.identity = None

    def shared_engine(self, model_ids):
        from lilo.control_plane.keys import placement_key
        from lilo.providers.modal.engines import instance_key

        placements = [
            self.stores["models"].get(placement_key(mid)) for mid in model_ids
        ]
        identity = validate_placements(placements, model_ids, self.definition_id)
        if self.identity is not None:
            require(
                identity == self.identity,
                "trainer instance or boot changed during test",
            )
        self.identity = identity
        record = self.stores["engines"].get(instance_key(identity[0]))
        require(
            record is not None and record["state"] == "running", "trainer not running"
        )
        require(record["boot_id"] == identity[1], "stale engine placement")
        # Do not serialize the engine record: it contains a private URL and token.
        headers = {"Authorization": f"Bearer {record['token']}"}
        response = httpx.get(
            record["url"].rstrip("/") + "/api/v1/models", headers=headers, timeout=60
        )
        response.raise_for_status()
        resident = response.json()["model_ids"]
        require(
            set(model_ids) <= set(resident), "clients not resident in shared trainer"
        )
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
    require(
        len(placements) == len(model_ids) and bool(placements), "missing placements"
    )
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
            response = client.post(
                "/api/v1/retrieve_future", json={"request_id": request_id}
            )
            if response.status_code != 408:
                response.raise_for_status()
                require("error" not in response.json(), "unload operation failed")
                return
            time.sleep(1)
        raise TimeoutError(f"unloading {model_id}")
