def test_256k_definition_captures_on_checkpoint_volume() -> None:
    from lilo.providers.modal.definitions import (
        qwen3_8_27b_miles_lora_256k as definition,
    )

    config = definition.backend_config("instance-a")

    assert config["capture_dir"] == f"{definition.CHECKPOINT_ROOT}/.captures/instance-a"


def test_256k_definition_widens_collective_timeout() -> None:
    from lilo.providers.modal.definitions import (
        qwen3_8_27b_miles_lora_256k as definition,
    )

    extra_args = definition.backend_config("instance-a")["miles"]["extra_args"]

    index = extra_args.index("--distributed-timeout-minutes")
    assert int(extra_args[index + 1]) > 10


def test_256k_definition_serializes_forward_backward_calls() -> None:
    from lilo.providers.modal.definitions import (
        qwen3_8_27b_miles_lora_256k as definition,
    )

    assert definition.MAX_FORWARD_BACKWARD_BATCH == 1
