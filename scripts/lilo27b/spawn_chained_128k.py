import modal

fn = modal.Function.from_name("lilo27b-client-chained", "run")
call = fn.spawn(
    steps=30,
    wandb_group="lilo-27b",
    run_name="lilo-27b-128k-pad-30-r2",
    trainer_gpus=8,
    max_steps_off_policy=1,
    std_normalize_advantages=True,
    per_token_loss_scale=True,
    base_url="https://modal-labs-micah-dev--lilo-27b-server.us-west.modal.run",
    base_model="Qwen/Qwen3.8-27B",
    engine_model="qwen3_8_27b_miles_lora_128k",
    context_length=131072,
    max_generation_tokens=4096,
    pad_to_tokens=120000,
    save_every=2,
    deadline_hours=22.0,
)
print(f"CALL_ID={call.object_id}", flush=True)
