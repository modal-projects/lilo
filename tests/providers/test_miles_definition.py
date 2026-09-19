import ast
from pathlib import Path

from lilo.backends.miles_config import MILES_REF
from lilo.providers.modal import miles_image
from lilo.providers.modal.definitions import qwen3_5_9b_base_miles_lora_2k as definition
from lilo.providers.modal.definitions import (
    qwen3_5_9b_base_miles_lora_16k as shared,
)
from lilo.providers.modal.definitions import (
    qwen3_5_9b_base_miles_lora_16k_single as single,
)


def test_miles_definition_uses_one_lilo_driver_for_all_ray_workers() -> None:
    assert definition.PARAMETERIZATION == "lora"
    assert definition.TRAINER_MODELS_PER_INSTANCE == definition.MAX_LORA_SLOTS
    assert definition.GPUS == 4
    assert MILES_REF == "main"
    assert len(miles_image.MILES_COMMIT) == 40

    tree = ast.parse(Path(definition.__file__).read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_engine_with_backend"
    ]
    assert len(calls) == 1
    call = calls[0]
    assert isinstance(call.args[1], ast.Constant)
    assert call.args[1].value == "lilo.backends.miles_lora:build_executor"
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    assert isinstance(keywords["nproc"], ast.Constant)
    assert keywords["nproc"].value == 1
    assert isinstance(keywords["max_models"], ast.Name)
    assert keywords["max_models"].id == "MAX_LORA_SLOTS"


def test_long_context_miles_definition_bounds_retained_adapter_versions():
    assert shared.MAX_CONTEXT_LENGTH >= 8192 + 2048
    assert shared.MAX_LORA_SLOTS >= 6
    assert shared.ROLLOUT_MAX_LOADED_LORAS == 64
    assert shared.ROLLOUT_MAX_LOADED_LORAS >= shared.ROLLOUT_MAX_LORAS_PER_BATCH
    assert shared.ROLLOUT_MAX_CONTAINERS == 8
    assert shared.CATALOG_VISIBLE
    assert not definition.CATALOG_VISIBLE


def test_single_tenant_definition_preserves_shared_hardware_and_backend():
    assert single.TRAINER_MODELS_PER_INSTANCE == 1
    assert not single.CATALOG_VISIBLE
    assert single.run_trainer is shared.run_trainer
    for name in (
        "MODEL_NAME",
        "GPUS",
        "GPU_TYPE",
        "MAX_CONTEXT_LENGTH",
        "TENSOR_MODEL_PARALLEL_SIZE",
        "MAX_LORA_SLOTS",
        "MAX_LORA_RANK",
        "ROLLOUT_MIN_CONTAINERS",
        "ROLLOUT_MAX_CONTAINERS",
        "ROLLOUT_GPU_TYPE",
    ):
        assert getattr(single, name) == getattr(shared, name)
