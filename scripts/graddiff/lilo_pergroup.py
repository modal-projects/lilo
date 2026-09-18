# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "datasets",
#   "wandb",
#   "tinker>=0.24,<0.25",
#   "tinker-cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@c8ed9c764b59161391156f980102d82f05014765",
# ]
# ///

"""Per-group decomposition of the step-0 batch on ONE fresh LoRA client.

For k in 0..15: forward_backward(group k's 8 datums, advantages exactly as
stored in batch.json) then optim_step(lr=0) to flush/reset grads. After each
call the trainer's dump (written to /graddiff/lilo_dump/lilo on the shared
volume) is copied to /graddiff/lilo_pergroup/<tag>/ and the originals removed,
since the trainer overwrites the same filenames every call. A 17th call on all
128 datums produces lilo_pergroup/full/ (the A reference for sum-of-groups).
"""

from __future__ import annotations

import json
import os
import shutil
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
DUMP_SRC = Path(VOLUME_ROOT) / "lilo_dump" / "lilo"
PERGROUP_DIR = Path(VOLUME_ROOT) / "lilo_pergroup"
N_GROUPS = 16
GROUP_SIZE = 8
ADAM = {
    "learning_rate": 0.0,  # flush grads without changing weights
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


def _dump_slot_stats(path: Path) -> dict:
    """Which adapter slots have nonzero grads in a grads_reduced file."""
    import torch

    f = path / "rank0_tp0_dp0_grads_reduced.pt"
    if not f.exists():
        return {}
    g = torch.load(f, weights_only=False)
    stats: dict[str, dict] = {}
    for name, t in g.items():
        slot = name.split(".", 1)[0]
        s = stats.setdefault(slot, {"n": 0, "nonzero": 0})
        s["n"] += 1
        if t.abs().max().item() > 0:
            s["nonzero"] += 1
    return stats


@app.function(
    image=image,
    volumes={VOLUME_ROOT: volume},
    secrets=secrets,
    timeout=60 * 60,
    cpu=4,
    memory=32 * 1024,
)
def lilo_pergroup() -> dict:
    import tinker
    import torch

    volume.reload()
    payload = json.loads((Path(VOLUME_ROOT) / "batch" / "batch.json").read_text())

    datums = []
    for entry in payload["datums"]:
        datums.append(
            tinker.Datum(
                model_input=tinker.ModelInput.from_ints(entry["input_tokens"]),
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
    assert len(datums) == N_GROUPS * GROUP_SIZE

    service = tinker.ServiceClient(base_url=BASE_URL)
    training_client = service.create_lora_training_client(
        base_model=MODEL_ID,
        rank=32,
        train_mlp=True,
        train_attn=True,
        train_unembed=False,
    )

    adam = tinker.AdamParams(**ADAM)
    calls = [
        (f"g{k}", datums[k * GROUP_SIZE : (k + 1) * GROUP_SIZE])
        for k in range(N_GROUPS)
    ]
    calls.append(("full", datums))

    results: dict[str, dict] = {}
    slot_stats: dict[str, dict] = {}
    for tag, group_datums in calls:
        t0 = time.time()
        fwd_bwd_result = training_client.forward_backward(
            group_datums, loss_fn="ppo", loss_fn_config=LOSS_FN_CONFIG
        ).result()
        fb_s = time.time() - t0

        t0 = time.time()
        optim_result = training_client.optim_step(adam).result()
        opt_s = time.time() - t0

        loss_fn_outputs = []
        for output in fwd_bwd_result.loss_fn_outputs:
            loss_fn_outputs.append(
                {
                    key: (
                        list(value.to_torch().tolist())
                        if hasattr(value, "to_torch")
                        else value
                    )
                    for key, value in output.items()
                }
            )
        results[tag] = {
            "metrics": getattr(fwd_bwd_result, "metrics", None),
            "loss_fn_outputs": loss_fn_outputs,
            "optim_step": getattr(optim_result, "metrics", optim_result),
            "forward_backward_s": fb_s,
            "optim_step_s": opt_s,
            "n_datums": len(group_datums),
        }

        # Move the trainer-written dump aside before the next call overwrites it.
        volume.reload()
        dest = PERGROUP_DIR / tag
        dest.mkdir(parents=True, exist_ok=True)
        moved = 0
        for f in DUMP_SRC.iterdir():
            if f.is_file():
                shutil.copy2(f, dest / f.name)
                f.unlink()
                moved += 1
        volume.commit()
        slot_stats[tag] = _dump_slot_stats(dest)
        print(
            f"[{tag}] n={len(group_datums)} fb={fb_s:.1f}s opt={opt_s:.1f}s "
            f"files={moved} slots={slot_stats[tag]}",
            flush=True,
        )

    out = {
        "results": results,
        "slot_stats": slot_stats,
        "model_id": MODEL_ID,
        "base_url": BASE_URL,
        "adam": ADAM,
        "loss_fn_config": LOSS_FN_CONFIG,
    }
    (PERGROUP_DIR / "outputs_pergroup.json").write_text(json.dumps(out))
    volume.commit()
    return out


@app.local_entrypoint()
def main() -> None:
    out = lilo_pergroup.remote()
    out_dir = Path.home() / "work/graddiff"
    out_dir.mkdir(parents=True, exist_ok=True)
    slim = {
        k: {
            t: {kk: v[kk] for kk in v if kk != "loss_fn_outputs"}
            for t, v in out[k].items()
        }
        for k in ("results", "slot_stats")
    }
    print(json.dumps(slim, indent=2, default=str))
    (out_dir / "lilo_arm_outputs_pergroup.json").write_text(json.dumps(out))
    print(f"wrote {out_dir / 'lilo_arm_outputs_pergroup.json'}")
