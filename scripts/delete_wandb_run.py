"""Delete a W&B run (runs on Modal so wandb-secret is available).

Usage:
    MODAL_ENVIRONMENT=micah-dev uv run --with modal modal run \
        scripts/delete_wandb_run.py --run-path modal-labs/miles-lora-longcontext/<id>
"""

import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install("wandb")
app = modal.App("delete-wandb-run", image=image)


@app.function(secrets=[modal.Secret.from_name("wandb-secret")], timeout=300)
def delete(run_path: str) -> str:
    import wandb

    run = wandb.Api().run(run_path)
    name = run.name
    run.delete()
    return f"deleted {run_path} ({name})"


@app.local_entrypoint()
def main(run_path: str):
    print(delete.remote(run_path))
