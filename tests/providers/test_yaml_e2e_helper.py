from pathlib import Path
import runpy
from types import SimpleNamespace

import modal
import pytest

from lilo.deployments import load, preset_path, DeploymentRecord


@pytest.mark.parametrize("preset", ["qwen35-9b-lora-16k", "qwen35-4b-fft-64k"])
def test_e2e_helper_reads_active_deployed_configuration(monkeypatch, preset):
    helper = runpy.run_path(
        str(Path(__file__).parents[2] / "scripts/e2e_engine_definition.py")
    )
    row = DeploymentRecord.create(load(preset_path(preset)), revision="a" * 40, implementation="test")
    retired = row.model_copy(update={"active": False, "generation": "b" * 64})
    registry = SimpleNamespace(
        get=lambda *args: [retired.model_dump(), row.model_dump()]
    )
    monkeypatch.setattr(modal.Dict, "from_name", lambda name: registry)
    definition, mode = helper["_definition"]("test-frontend", row.spec.name)
    assert definition.DEFINITION_ID == row.definition_id
    assert definition.MAX_CONTEXT_LENGTH == row.spec.model.max_context_length
    assert definition.GPUS == row.spec.trainer.resources.gpu_count
    assert mode == row.spec.model.parameterization
    assert definition.MAX_TOKENS_PER_MICROBATCH > 0
    with pytest.raises(ValueError, match="one active YAML"):
        helper["_definition"]("test-frontend", "missing")
