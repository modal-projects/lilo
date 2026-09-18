import os

import modal

CLIENT_APP = os.environ.get("LILO_CLIENT_APP_NAME", "lilo27b-client-chained-b")
BASE_URL = os.environ.get(
    "LILO_BASE_URL",
    "https://modal-labs-micah-dev--lilo-27b-b-server.us-west.modal.run",
)

fn = modal.Function.from_name(CLIENT_APP, "run")
call = fn.spawn(
    steps=10,
    wandb_group="lilo-27b",
    run_name="lilo-27b-128k-pad-samplemean-10",
    trainer_gpus=8,
    max_steps_off_policy=1,
    std_normalize_advantages=True,
    per_token_loss_scale=False,
    sample_mean_advantages=True,
    base_url=BASE_URL,
    base_model="Qwen/Qwen3.8-27B",
    engine_model="qwen3_8_27b_miles_lora_128k",
    context_length=131072,
    max_generation_tokens=4096,
    pad_to_tokens=120000,
    save_every=2,
    deadline_hours=8.0,
)
print(f"CALL_ID={call.object_id}", flush=True)
