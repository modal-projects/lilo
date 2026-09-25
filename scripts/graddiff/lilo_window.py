# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "tinker>=0.24,<0.25",
# ]
# ///

"""Position-window probe on ONE datum, Lilo only.

Same datum (default: g8 datum 67, prompt 11357 + response 2159 tokens) is sent
repeatedly with the advantage restricted to one window of response tokens
(constant 1/len(window) inside, 0 outside). If every response position
contributes to dB, the B-grad norm per window is of comparable magnitude; a
window whose gradient collapses to ~0 localizes a position cutoff.
Dumps land in /graddiff/lilo_window/<tag>/.
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
OUT_DIR = Path(VOLUME_ROOT) / "lilo_window"
ADAM = {
    "learning_rate": 0.0,
    "beta1": 0.9,
    "beta2": 0.95,
    "eps": 1e-8,
    "weight_decay": 0.0,
    "grad_clip_norm": 0.0,
}
DATUM = 67
WINDOWS = [
    (0, 256),
    (256, 512),
    (512, 768),
    (768, 931),
    (931, 1200),
    (1200, 1500),
    (1500, 1800),
    (1800, 2159),
]

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
secrets = [modal.Secret.from_name("lilo-api", required_keys=["TINKER_API_KEY"])]
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "tinker>=0.24,<0.25", "torch"
)
app = modal.App(APP_NAME)


def _bgrad_norms(path: Path) -> dict[str, float]:
    import torch

    files = sorted(path.glob("rank*_grads_reduced.pt"))
    seen: dict[str, dict[int, torch.Tensor]] = {}
    for f in files:
        # rank{r}_tp{t}_dp{d}_grads_reduced.pt ; reduced grads equal across dp
        parts = f.name.split("_")
        tp = int(parts[1][2:])
        dp = int(parts[2][2:])
        if dp != 0:
            continue
        for name, t in torch.load(f, weights_only=False).items():
            if "linear_out" in name and "language_model" in name:
                seen.setdefault(name, {})[tp] = t.double()
    total = 0.0
    per_type = {"fc1": 0.0, "fc2": 0.0, "qkv": 0.0, "proj": 0.0}
    for name, shards in seen.items():
        n2 = sum(float((s * s).sum()) for s in shards.values())
        total += n2
        for k in per_type:
            if f"linear_{k}" in name:
                per_type[k] += n2
    return {"total": total**0.5, **{k: v**0.5 for k, v in per_type.items()}}


@app.function(
    image=image, volumes={VOLUME_ROOT: volume}, secrets=secrets, timeout=3600, cpu=4
)
def lilo_window(
    datum_idx: int = DATUM, windows: list = WINDOWS, filler_idx: int = 71
) -> dict:
    import tinker
    import torch

    volume.reload()
    all_datums = json.loads((Path(VOLUME_ROOT) / "batch" / "batch.json").read_text())[
        "datums"
    ]
    entry = all_datums[datum_idx]

    def make_datum(e: dict, adv: torch.Tensor) -> tinker.Datum:
        return tinker.Datum(
            model_input=tinker.ModelInput.from_ints(e["input_tokens"]),
            loss_fn_inputs={
                "target_tokens": tinker.TensorData.from_torch(
                    torch.tensor(e["target_tokens"], dtype=torch.int64)
                ),
                "logprobs": tinker.TensorData.from_torch(
                    torch.tensor(e["sampled_logprobs"], dtype=torch.float32)
                ),
                "advantages": tinker.TensorData.from_torch(adv),
            },
        )

    # The trainer pads a batch to a DP multiple with a dummy that lacks
    # target_tokens, so a lone datum fails; pair it with a zero-advantage filler.
    filler_entry = all_datums[filler_idx]
    filler = make_datum(
        filler_entry, torch.zeros(len(filler_entry["mask"]), dtype=torch.float32)
    )
    mask = torch.tensor(entry["mask"], dtype=torch.bool)
    first = int(mask.nonzero()[0])
    n_resp = int(mask.sum())

    service = tinker.ServiceClient(base_url=BASE_URL)
    client = service.create_lora_training_client(
        base_model=MODEL_ID,
        rank=32,
        train_mlp=True,
        train_attn=True,
        train_unembed=False,
    )
    adam = tinker.AdamParams(**ADAM)
    results = {}
    for lo, hi in windows:
        hi = min(hi, n_resp)
        adv = torch.zeros(len(entry["mask"]), dtype=torch.float32)
        adv[first + lo : first + hi] = 1.0 / (hi - lo)
        datum = make_datum(entry, adv)
        fb = client.forward_backward(
            [datum, filler], loss_fn="ppo", loss_fn_config=LOSS_FN_CONFIG
        ).result()
        opt = client.optim_step(adam).result()
        tag = f"w{lo}_{hi}"
        dest = OUT_DIR / tag
        dest.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + 300
        while True:
            volume.reload()
            if (
                len(list(DUMP_SRC.glob("rank*_grads_reduced.pt"))) >= 8
                and len(list(DUMP_SRC.glob("rank*_delta.pt"))) >= 8
            ):
                break
            if time.time() > deadline:
                raise RuntimeError(f"[{tag}] dump incomplete")
            time.sleep(5)
        for f in DUMP_SRC.iterdir():
            if f.is_file():
                shutil.copy2(f, dest / f.name)
                f.unlink()
        volume.commit()
        norms = _bgrad_norms(dest)
        loss = fb.loss_fn_outputs[0]["loss:sum"].to_torch().item()
        results[tag] = {
            "abs_pos": (first + lo, first + hi),
            "loss_sum": loss,
            "grad_norm_reported": getattr(opt, "metrics", {}),
            **norms,
        }
        print(
            f"[{tag}] abs={first + lo}-{first + hi} loss={loss:.5g} {norms}", flush=True
        )
    (OUT_DIR / "results.json").write_text(json.dumps(results, default=str))
    volume.commit()
    return results


@app.local_entrypoint()
def main() -> None:
    out = lilo_window.remote()
    print(json.dumps(out, indent=2, default=str))
    p = Path.home() / "work/graddiff/lilo_window_results.json"
    p.write_text(json.dumps(out, default=str))
    print("wrote", p)
