import ast
from pathlib import Path

from lilo.backends.miles_config import MILES_REVISION
from lilo.providers.modal import miles_image
from lilo.providers.modal.definitions import qwen3_5_9b_base_miles_lora_2k as definition


def test_miles_definition_uses_one_lilo_driver_for_all_ray_workers() -> None:
    assert definition.PARAMETERIZATION == "lora"
    assert definition.TRAINER_MODELS_PER_INSTANCE == definition.MAX_LORA_SLOTS
    assert definition.GPUS == 4
    assert miles_image.MILES_REVISION == MILES_REVISION

    tree = ast.parse(Path(definition.__file__).read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run_engine_with_backend"
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


def test_long_context_miles_definition_supports_twenty_step_sampling():
    from lilo.providers.modal.definitions import qwen3_5_9b_base_miles_lora_16k as long

    assert long.MAX_CONTEXT_LENGTH >= 8192 + 2048
    assert long.MAX_LORA_SLOTS >= 2
    assert long.ROLLOUT_MAX_LOADED_LORAS >= 2 * (20 + 2)
    assert long.ROLLOUT_MAX_CONTAINERS == 2
    assert long.CATALOG_VISIBLE
    assert not definition.CATALOG_VISIBLE
