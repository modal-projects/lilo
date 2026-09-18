# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["modal"]
# ///
"""Launch the LongRLVR Lilo client (scripts/run_longrlvr_lilo_lora.py) as a
detached Modal function against the deployed ``lilo-dp2`` server.

Reproduces the czzwpqau client container (pinned prompts at
/root/pinned/pinned_prompts.json, log under /root) with an optional
``--adv-weighting sample_mean`` switch for the graddiff e2e experiment.

    uv run scripts/graddiff/e2e_lilo_client.py --run-name lilo-9b-16k-samplemean-30 \
        --steps 30 --adv-weighting sample_mean --wandb-group graddiff-e2e
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "lilo-graddiff-e2e-client"
VOLUME_NAME = "lilo-graddiff"
VOLUME_ROOT = "/graddiff"
PINNED_PROMPT_PATH = "/root/pinned/pinned_prompts.json"
MODEL_ID = "qwen3_5_9b_miles_lora_16k_dp2"
BASE_URL = os.environ.get(
    "TINKER_BASE_URL",
    "https://modal-labs-micah-dev--lilo-dp2-server.us-west.modal.run",
)
SCRIPTS_DIR = Path(__file__).resolve().parents[1]

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
secrets = [
    modal.Secret.from_name("lilo-api", required_keys=["TINKER_API_KEY"]),
    modal.Secret.from_name("wandb-secret"),
    modal.Secret.from_name("huggingface-secret"),
]

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
    .add_local_file(
        str(Path.home() / "work/modal_dl/pinned_prompts.json"),
        PINNED_PROMPT_PATH,
    )
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    volumes={VOLUME_ROOT: volume},
    secrets=secrets,
    timeout=24 * 60 * 60,
    cpu=4,
    memory=32 * 1024,
)
def run_client(client_args: list[str], run_name: str) -> int:
    log_path = f"/root/{run_name}_metrics.json"
    command = [
        sys.executable,
        "/root/scripts/run_longrlvr_lilo_lora.py",
        *client_args,
        "--log-path",
        log_path,
    ]
    print("launching:", " ".join(command), flush=True)
    proc = subprocess.run(command, check=False)
    dest = Path(VOLUME_ROOT) / "e2e" / run_name
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "-r", log_path, str(dest)], check=False)
    volume.commit()
    return proc.returncode


@app.local_entrypoint()
def main(
    run_name: str,
    steps: int = 30,
    adv_weighting: str = "token_mean",
    wandb_group: str = "graddiff-e2e",
    base_url: str = BASE_URL,
    seed: int = 0,
    detach: bool = True,
) -> None:
    del detach
    client_args = [
        "--steps",
        str(steps),
        "--base-url",
        base_url,
        "--base-model",
        MODEL_ID,
        "--prompt-file",
        PINNED_PROMPT_PATH,
        "--wandb-group",
        wandb_group,
        "--run-name",
        run_name,
        "--trainer-gpus",
        "8",
        "--max-steps-off-policy",
        "1",
        "--seed",
        str(seed),
        "--std-normalize-advantages",
        "--per-token-loss-scale",
        "--adv-weighting",
        adv_weighting,
    ]
    call = run_client.spawn(client_args, run_name)
    print(f"spawned function call {call.object_id} for {run_name}")


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args([])
