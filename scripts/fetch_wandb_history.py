"""Dump a W&B run's full history as JSON (runs on Modal so wandb-secret is available).

Usage:
    MODAL_ENVIRONMENT=micah-dev uv run --with modal modal run \
        scripts/fetch_wandb_history.py --run-id <id> --out logs/<id>_history.json
"""

import json

import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install("wandb")
app = modal.App("fetch-wandb-history", image=image)


@app.function(secrets=[modal.Secret.from_name("wandb-secret")], timeout=600)
def fetch(run_path: str) -> str:
    import wandb

    run = wandb.Api().run(run_path)
    rows = list(run.scan_history())
    return json.dumps({"summary": dict(run.summary), "history": rows}, default=str)


@app.local_entrypoint()
def main(
    run_id: str,
    out: str,
    entity: str = "modal-labs",
    project: str = "miles-lora-longcontext",
):
    data = fetch.remote(f"{entity}/{project}/{run_id}")
    with open(out, "w") as f:
        f.write(data)
    print(f"wrote {out} ({len(data)} bytes)")
