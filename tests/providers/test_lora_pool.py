from lilo.providers.modal.lora_pool import LoraPoolSpec


def test_lora_pool_is_shared_by_every_adapter_for_definition() -> None:
    first = LoraPoolSpec("qwen3_5_9b_base_miles_lora_2k")
    second = LoraPoolSpec.from_dict(first.as_dict())

    assert first == second
    assert first.app_name == second.app_name
    assert first.app_name != LoraPoolSpec(first.definition_id, "old").app_name
    assert first.app_name.startswith("lilo-lora-")
    assert first.env() == {
        "LILO_LORA_POOL_APP_NAME": first.app_name,
        "LILO_LORA_POOL_DEFINITION_ID": first.definition_id,
    }
