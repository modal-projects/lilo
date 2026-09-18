"""Upload graddiff-step0 result bundle as a W&B artifact.

Run: MODAL_SERVER_URL=https://api.modal.com MODAL_PROFILE= MODAL_ENVIRONMENT=micah-dev \
  uv run modal run scripts/graddiff/upload_artifact.py
"""

import modal

app = modal.App("graddiff-wandb-artifact")
image = (
    modal.Image.debian_slim()
    .pip_install("wandb")
    .add_local_dir("/home/ubuntu/work/graddiff/artifact_bundle", remote_path="/bundle")
)


@app.function(
    image=image, secrets=[modal.Secret.from_name("wandb-secret")], timeout=1800
)
def upload() -> str:
    import wandb

    run = wandb.init(
        project="miles-lora-longcontext",
        entity="modal-labs",
        group="graddiff-step0",
        name="graddiff-step0-artifact",
        job_type="artifact-upload",
    )
    art = wandb.Artifact(
        "graddiff-step0-dumps",
        type="graddiff-results",
        description=(
            "Lilo-vs-Miles step-0 gradient diff: batch, metrics jsons, comparison table, "
            "probe outputs. Full fp32 dumps remain on Modal volume lilo-graddiff "
            "(see manifest.json)."
        ),
    )
    art.add_dir("/bundle")
    logged = run.log_artifact(art)
    run.finish()
    return logged.id


@app.local_entrypoint()
def main() -> None:
    print("artifact_id:", upload.remote())
