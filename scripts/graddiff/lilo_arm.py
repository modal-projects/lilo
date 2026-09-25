# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "datasets",
#   "wandb",
#   "tinker>=0.24,<0.25",
#   "tinker-cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@c8ed9c764b59161391156f980102d82f05014765",
# ]
# ///
"""Lilo arm of the step-0 gradient diff: replay batch.json through a fresh
rank-32 LoRA on the lilo-graddiff server (which has LILO_GRADDIFF_DUMP_DIR
enabled) and persist the client-visible outputs."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

APP_NAME = "lilo-graddiff-client"
VOLUME_NAME = "lilo-graddiff"
VOLUME_ROOT = "/graddiff"

MODEL_ID = "qwen3_5_9b_miles_lora_16k_dp2"
BASE_URL = os.environ.get(
    "TINKER_BASE_URL",
    "https://modal-labs-micah-dev--lilo-graddiff-server.us-west.modal.run",
)
LOSS_FN_CONFIG = {"clip_low_threshold": 0.8, "clip_high_threshold": 1.28}
# "a:b" slice over batch.json datums; None = full batch. OUT_NAME selects the
# volume/local output filename (default "outputs" -> lilo_arm_outputs.json).
DATUM_RANGE = os.environ.get("DATUM_RANGE")
OUT_NAME = os.environ.get("OUT_NAME", "outputs")
ADAM = {
    "learning_rate": 1e-4,
    "beta1": 0.9,
    "beta2": 0.95,
    "eps": 1e-8,
    "weight_decay": 0.0,
    "grad_clip_norm": 0.0,
}

SCRIPTS_DIR = Path(__file__).resolve().parents[1]

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
secrets = [modal.Secret.from_name("lilo-api", required_keys=["TINKER_API_KEY"])]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "datasets",
        "wandb",
        "tinker>=0.24,<0.25",
        "tinker-cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@c8ed9c764b59161391156f980102d82f05014765",
        "torch",
    )
    .add_local_dir(str(SCRIPTS_DIR), remote_path="/root/scripts")
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    volumes={VOLUME_ROOT: volume},
    secrets=secrets,
    timeout=60 * 60,
    cpu=4,
    memory=32 * 1024,
)
def lilo_arm(datum_range: str | None = None, out_name: str = "outputs") -> dict:
    import tinker
    import torch

    volume.reload()
    batch_path = Path(VOLUME_ROOT) / "batch" / "batch.json"
    payload = json.loads(batch_path.read_text())

    entries = payload["datums"]
    if datum_range:
        lo, hi = (int(x) for x in datum_range.split(":"))
        entries = entries[lo:hi]
    datums = []
    for entry in entries:
        datums.append(
            tinker.Datum(
                model_input=tinker.ModelInput.from_ints(entry["input_tokens"]),
                # Cookbook strips "mask" before forward_backward; only
                # target_tokens/logprobs/advantages reach the server.
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData.from_torch(
                        torch.tensor(entry["target_tokens"], dtype=torch.int64)
                    ),
                    "logprobs": tinker.TensorData.from_torch(
                        torch.tensor(entry["sampled_logprobs"], dtype=torch.float32)
                    ),
                    "advantages": tinker.TensorData.from_torch(
                        torch.tensor(entry["advantages"], dtype=torch.float32)
                    ),
                },
            )
        )

    service = tinker.ServiceClient(base_url=BASE_URL)
    training_client = service.create_lora_training_client(
        base_model=MODEL_ID,
        rank=32,
        train_mlp=True,
        train_attn=True,
        train_unembed=False,
    )

    timings: dict[str, float] = {}
    t0 = time.time()
    fwd_bwd_result = training_client.forward_backward(
        datums, loss_fn="ppo", loss_fn_config=LOSS_FN_CONFIG
    ).result()
    timings["forward_backward_s"] = time.time() - t0

    loss_fn_outputs = []
    for output in fwd_bwd_result.loss_fn_outputs:
        row = {
            key: (
                list(value.to_torch().tolist()) if hasattr(value, "to_torch") else value
            )
            for key, value in output.items()
        }
        loss_fn_outputs.append(row)

    t0 = time.time()
    optim_result = training_client.optim_step(tinker.AdamParams(**ADAM)).result()
    timings["optim_step_s"] = time.time() - t0

    result = {
        "metrics": getattr(fwd_bwd_result, "metrics", None),
        "loss_fn_outputs": loss_fn_outputs,
        "optim_step": getattr(optim_result, "metrics", optim_result),
        "timings": timings,
        "model_id": MODEL_ID,
        "base_url": BASE_URL,
        "n_datums": len(datums),
        "adam": ADAM,
        "loss_fn_config": LOSS_FN_CONFIG,
    }

    out_path = Path(VOLUME_ROOT) / "lilo_arm" / f"{out_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result))
    volume.commit()
    print(
        json.dumps(
            {
                "metrics": result["metrics"],
                "optim_step": result["optim_step"],
                "timings": timings,
                "n_loss_fn_outputs": len(loss_fn_outputs),
            },
            indent=2,
            default=str,
        )
    )
    return result


@app.local_entrypoint()
def main() -> None:
    result = lilo_arm.remote(DATUM_RANGE, OUT_NAME)
    out_dir = Path.home() / "work/graddiff"
    out_dir.mkdir(parents=True, exist_ok=True)
    local_name = (
        "lilo_arm_outputs.json"
        if OUT_NAME == "outputs"
        else f"lilo_arm_outputs{OUT_NAME}.json"
    )
    (out_dir / local_name).write_text(json.dumps(result))
    print(f"wrote {out_dir / local_name}")
