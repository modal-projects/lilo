from pathlib import Path
import runpy

import pytest

from lilo.deployments import load, config_path, DeploymentRecord
from lilo.providers.modal import deployment_records


@pytest.mark.parametrize("preset", ["qwen35-9b-lora-16k", "qwen35-4b-fft-64k"])
def test_e2e_helper_reads_active_deployed_configuration(monkeypatch, preset):
    row = DeploymentRecord.create(load(config_path(preset)), revision="a" * 40)
    retired = row.model_copy(update={"active": False, "generation": "b" * 64})
    monkeypatch.setattr(
        deployment_records,
        "deployed_manifest",
        lambda frontend: [retired.model_dump(), row.model_dump()],
    )
    helper = runpy.run_path(
        str(Path(__file__).parents[2] / "scripts/e2e_engine_definition.py")
    )
    definition, mode = helper["_definition"]("test-frontend", row.spec.name)
    assert definition.DEFINITION_ID == row.definition_id
    assert definition.MAX_CONTEXT_LENGTH == row.spec.max_context_length
    assert definition.GPUS == row.spec.trainer_gpus_per_node
    assert mode == row.spec.parameterization
    assert definition.MAX_TOKENS_PER_MICROBATCH > 0
    with pytest.raises(ValueError, match="one active configuration"):
        helper["_definition"]("test-frontend", "missing")
