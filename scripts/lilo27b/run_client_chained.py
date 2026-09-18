"""Chained LongRLVR client: auto-resumes from the latest checkpoint and spawns
its own successor before Modal's 24h function cap.

log_path (and therefore checkpoints.jsonl) lives on a Modal Volume so each
generation can pick up where the previous one stopped.
"""

import os
import subprocess
import threading
import time
from pathlib import Path

import modal

SCRIPTS_DIR = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "datasets",
        "wandb",
        "tinker>=0.24,<0.25",
        "git+https://github.com/thinking-machines-lab/tinker-cookbook.git@c8ed9c764b59161391156f980102d82f05014765",
    )
    .add_local_dir(
        str(SCRIPTS_DIR),
        remote_path="/root/scripts",
        copy=True,
    )
)

APP_NAME = "lilo27b-client-chained"
app = modal.App(APP_NAME, image=image)
runs_volume = modal.Volume.from_name("lilo-27b-runs", create_if_missing=True)

MAX_GENERATIONS = 8


@app.function(
    secrets=[
        modal.Secret.from_name("lilo-api"),
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret"),
    ],
    volumes={"/runs": runs_volume},
    timeout=24 * 60 * 60,
    retries=0,
)
def run(
    steps: int = 30,
    wandb_group: str = "lilo-27b",
    run_name: str = "lilo-27b-256k-pad-30",
    trainer_gpus: int = 24,
    max_steps_off_policy: int = 1,
    std_normalize_advantages: bool = False,
    per_token_loss_scale: bool = False,
    base_url: str = "https://modal-labs-micah-dev--lilo-27b-server.us-west.modal.run",
    base_model: str = "Qwen/Qwen3.8-27B",
    context_length: int | None = None,
    max_generation_tokens: int | None = None,
    engine_model: str | None = None,
    pad_to_tokens: int = 0,
    save_every: int = 2,
    generation: int = 0,
    deadline_hours: float = 22.0,
    wandb_run_id: str | None = None,
) -> None:
    log_path = f"/runs/{run_name}"
    os.makedirs(log_path, exist_ok=True)
    id_file = os.path.join(log_path, "wandb_run_id.txt")
    if os.path.exists(id_file):
        with open(id_file, encoding="utf-8") as handle:
            wandb_run_id = handle.read().strip() or None
    if wandb_run_id is None:
        wandb_run_id = f"{run_name}-{int(time.time())}"
    with open(id_file, "w", encoding="utf-8") as handle:
        handle.write(wandb_run_id)
    runs_volume.commit()

    env = {
        **os.environ,
        "TINKER_BASE_URL": base_url,
        "WANDB_ENTITY": "modal-labs",
        "WANDB_RUN_GROUP": wandb_group,
        "WANDB_RUN_ID": wandb_run_id,
        "WANDB_RESUME": "allow",
        "PYTHONPATH": "/root/scripts",
    }
    command = [
        "python",
        "/root/scripts/run_longrlvr_lilo_lora.py",
        "--steps",
        str(steps),
        "--base-url",
        base_url,
        "--base-model",
        base_model,
        "--wandb-group",
        wandb_group,
        "--run-name",
        run_name,
        "--trainer-gpus",
        str(trainer_gpus),
        "--max-steps-off-policy",
        str(max_steps_off_policy),
        "--save-every",
        str(save_every),
        "--log-path",
        log_path,
        *(["--engine-model", engine_model] if engine_model is not None else []),
        *(["--context-length", str(context_length)] if context_length is not None else []),
        *(
            ["--max-generation-tokens", str(max_generation_tokens)]
            if max_generation_tokens is not None
            else []
        ),
        *(["--std-normalize-advantages"] if std_normalize_advantages else []),
        *(["--per-token-loss-scale"] if per_token_loss_scale else []),
        *(["--pad-to-tokens", str(pad_to_tokens)] if pad_to_tokens else []),
    ]

    stop = threading.Event()

    def commit_loop() -> None:
        while not stop.wait(60):
            try:
                runs_volume.commit()
            except Exception as exc:  # pragma: no cover - operational
                print(f"volume commit failed: {exc}", flush=True)

    committer = threading.Thread(target=commit_loop, daemon=True)
    committer.start()

    def checkpoint_count() -> int:
        path = os.path.join(log_path, "checkpoints.jsonl")
        if not os.path.exists(path):
            return 0
        with open(path, encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    print(f"generation={generation} log_path={log_path} wandb_run_id={wandb_run_id}", flush=True)
    soft_deadline = time.time() + deadline_hours * 3600
    hard_deadline = time.time() + (deadline_hours + 1.5) * 3600
    proc = subprocess.Popen(command, env=env)
    hit_deadline = False
    checkpoints_at_soft_deadline: int | None = None
    while True:
        try:
            rc = proc.wait(timeout=60)
            break
        except subprocess.TimeoutExpired:
            now = time.time()
            if now >= soft_deadline and checkpoints_at_soft_deadline is None:
                checkpoints_at_soft_deadline = checkpoint_count()
                print(
                    f"soft deadline reached with {checkpoints_at_soft_deadline} checkpoints; "
                    "waiting for the next save before handing off",
                    flush=True,
                )
            past_save = (
                checkpoints_at_soft_deadline is not None
                and checkpoint_count() > checkpoints_at_soft_deadline
            )
            if past_save or now >= hard_deadline:
                print(
                    f"handing off for chained resume (past_save={past_save})",
                    flush=True,
                )
                hit_deadline = True
                proc.terminate()
                try:
                    rc = proc.wait(timeout=600)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    rc = proc.wait()
                break

    stop.set()
    try:
        runs_volume.commit()
    except Exception as exc:  # pragma: no cover - operational
        print(f"final volume commit failed: {exc}", flush=True)

    print(f"child exited rc={rc} hit_deadline={hit_deadline}", flush=True)
    if rc == 0 and not hit_deadline:
        print("training finished", flush=True)
        return
    if generation + 1 >= MAX_GENERATIONS:
        raise RuntimeError(f"max generations reached (rc={rc})")

    successor = modal.Function.from_name(APP_NAME, "run")
    call = successor.spawn(
        steps=steps,
        wandb_group=wandb_group,
        run_name=run_name,
        trainer_gpus=trainer_gpus,
        max_steps_off_policy=max_steps_off_policy,
        std_normalize_advantages=std_normalize_advantages,
        per_token_loss_scale=per_token_loss_scale,
        base_url=base_url,
        base_model=base_model,
        context_length=context_length,
        max_generation_tokens=max_generation_tokens,
        engine_model=engine_model,
        pad_to_tokens=pad_to_tokens,
        save_every=save_every,
        generation=generation + 1,
        deadline_hours=deadline_hours,
        wandb_run_id=None,
    )
    print(f"spawned generation {generation + 1}: {call.object_id}", flush=True)
