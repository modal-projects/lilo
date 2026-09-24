from lilo.providers.modal.lora_pool import LoraPoolSpec


def test_lora_pool_is_shared_by_every_adapter_for_definition() -> None:
    first = LoraPoolSpec("deployment_example_0123456789abcdef")
    second = LoraPoolSpec.from_dict(first.as_dict())

    assert first == second
    assert first.app_name == second.app_name
    assert first.app_name != LoraPoolSpec(first.definition_id, "old").app_name
    assert first.app_name.startswith("lilo-lora-")
    assert first.env() == {
        "LILO_LORA_POOL_APP_NAME": first.app_name,
        "LILO_LORA_POOL_DEFINITION_ID": first.definition_id,
    }


def test_stop_already_stopped_lora_pool_succeeds_but_real_failure_propagates(
    monkeypatch,
):
    import subprocess
    import pytest
    from lilo.providers.modal import lora_pool

    monkeypatch.setattr(lora_pool.shutil, "which", lambda _: "/bin/modal")
    result = subprocess.CompletedProcess(
        [], 1, "", "App is already stopped. (Stopped yesterday).\n"
    )
    monkeypatch.setattr(lora_pool.subprocess, "run", lambda *args, **kwargs: result)
    spec = LoraPoolSpec("definition", "revision")
    lora_pool.stop_pool(spec)
    result.stderr = "authentication failed"
    with pytest.raises(subprocess.CalledProcessError):
        lora_pool.stop_pool(spec)


def test_pool_revision_comes_from_resolved_generation():
    first = LoraPoolSpec("deployment_example_0123456789abcdef")
    changed = LoraPoolSpec("deployment_example_fedcba9876543210")
    assert first.revision == "0123456789abcdef"
    assert first.app_name != changed.app_name


def test_python_definition_cannot_choose_a_pool_revision():
    import pytest

    with pytest.raises(ValueError, match="configured deployment id"):
        LoraPoolSpec("qwen3_5_9b_base_miles_lora_16k")
