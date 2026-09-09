import ast
import runpy
from pathlib import Path
from unittest.mock import patch, sentinel

import modal

from lilo.providers.modal.app import DEFINITIONS
from lilo.providers.modal.checkpoint_storage import (
    CHECKPOINT_ROOT,
    CHECKPOINT_VOLUME_NAME,
    checkpoint_volume,
)

FULL_DEFINITIONS = tuple(
    definition for definition in DEFINITIONS if definition.PARAMETERIZATION == "full"
)


def source_tree(definition) -> ast.Module:
    return ast.parse(Path(definition.__file__).read_text())


def string_dict_entries(node: ast.Dict) -> dict[str, ast.expr]:
    return {
        key.value: value
        for key, value in zip(node.keys, node.values, strict=True)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }


def test_checkpoint_storage_creates_one_v2_volume_without_live_lookup() -> None:
    storage_path = Path(__file__).parents[2] / (
        "src/lilo/providers/modal/checkpoint_storage.py"
    )
    with patch.object(
        modal.Volume,
        "from_name",
        return_value=sentinel.checkpoint_volume,
    ) as from_name:
        storage = runpy.run_path(str(storage_path))

    from_name.assert_called_once_with(
        "lilo-checkpoints",
        create_if_missing=True,
        version=2,
    )
    assert storage["CHECKPOINT_ROOT"] == "/checkpoints"


def test_all_definitions_share_checkpoint_storage() -> None:
    assert len(FULL_DEFINITIONS) == 5
    assert CHECKPOINT_VOLUME_NAME == "lilo-checkpoints"
    assert CHECKPOINT_ROOT == "/checkpoints"

    for definition in DEFINITIONS:
        assert definition.TRAINER_VOLUMES[CHECKPOINT_ROOT] is checkpoint_volume
        assert (
            definition.TRAINER_VOLUMES[definition.BULLETIN_ROOT] is definition.bulletin
        )
        assert definition.bulletin is not checkpoint_volume


def test_full_definitions_configure_checkpoint_dir_and_environment() -> None:
    for definition in FULL_DEFINITIONS:
        tree = source_tree(definition)
        config_assignments = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "backend_config"
                for target in node.targets
            )
            and isinstance(node.value, ast.Dict)
        ]
        assert len(config_assignments) == 1
        config = string_dict_entries(config_assignments[0].value)
        checkpoint_dir = config["checkpoint_dir"]
        assert isinstance(checkpoint_dir, ast.Name)
        assert checkpoint_dir.id == "CHECKPOINT_ROOT"

        engine_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_engine_with_backend"
        ]
        assert len(engine_calls) == 1
        backend_env = next(
            keyword.value
            for keyword in engine_calls[0].keywords
            if keyword.arg == "backend_env"
        )
        assert isinstance(backend_env, ast.Dict)
        env = string_dict_entries(backend_env)
        volume_name = env["LILO_CHECKPOINT_VOLUME"]
        assert isinstance(volume_name, ast.Name)
        assert volume_name.id == "CHECKPOINT_VOLUME_NAME"
