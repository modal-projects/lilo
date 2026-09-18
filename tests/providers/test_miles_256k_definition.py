def test_256k_definition_captures_on_checkpoint_volume() -> None:
    from lilo.providers.modal.definitions import (
        qwen3_8_27b_miles_lora_256k as definition,
    )

    config = definition.backend_config("instance-a")

    assert config["capture_dir"] == f"{definition.CHECKPOINT_ROOT}/.captures/instance-a"
