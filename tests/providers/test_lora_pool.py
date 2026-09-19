import subprocess
from pathlib import Path

import pytest

from lilo.providers.modal import lora_pool
from lilo.providers.modal.lora_pool import LoraPoolSpec, _definition_sources


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


def test_stop_already_stopped_lora_pool_succeeds_but_real_failure_propagates(
    monkeypatch,
):
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


def test_pool_revision_tracks_inherited_settings_and_bulletin(monkeypatch):
    original = Path.read_bytes
    changed = None

    def read(path):
        value = original(path)
        return value + b"\n# changed\n" if path.name == changed else value

    monkeypatch.setattr(Path, "read_bytes", read)
    name = "qwen3_5_9b_base_miles_lora_16k_single"
    original_pool = LoraPoolSpec(name).app_name
    for changed in ("qwen3_5_9b_base_miles_lora_16k.py", "bulletin.py"):
        assert LoraPoolSpec(name).app_name != original_pool
    changed = "qwen3_5_4b_full_64k.py"
    assert LoraPoolSpec(name).app_name == original_pool


def test_definition_dependency_scan_handles_cycles_and_relative_imports(tmp_path):
    (tmp_path / "child.py").write_text("from .parent import CONFIG\n")
    (tmp_path / "parent.py").write_text("from . import shared\n")
    (tmp_path / "shared.py").write_text("from .child import CONFIG\n")
    sources = list(_definition_sources(tmp_path / "child.py"))
    assert [path.name for path in sources] == ["child.py", "parent.py", "shared.py"]
