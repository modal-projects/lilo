import ast
import runpy
from pathlib import Path
from unittest.mock import patch, sentinel

import modal

from lilo.providers.modal import scoped
from lilo.providers.modal.app import DEFINITIONS
from lilo.providers.modal.kernel_cache import (
    KERNEL_CACHE_ENV,
    KERNEL_CACHE_ROOT,
    KERNEL_CACHE_VOLUME_NAME,
    kernel_cache_volume,
)


def source_tree(module) -> ast.Module:
    return ast.parse(Path(module.__file__).read_text())


def string_dict_entries(node: ast.Dict) -> dict[str, ast.expr]:
    return {
        key.value: value
        for key, value in zip(node.keys, node.values, strict=True)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }


def spread_names(node: ast.Dict) -> set[str]:
    return {
        value.id
        for key, value in zip(node.keys, node.values, strict=True)
        if key is None and isinstance(value, ast.Name)
    }


def calls_named(tree: ast.Module, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == name)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == name)
        )
    ]


def keyword(call: ast.Call, name: str) -> ast.expr:
    return next(item.value for item in call.keywords if item.arg == name)


def test_kernel_cache_creates_one_v2_volume_without_live_lookup() -> None:
    module_path = Path(__file__).parents[2] / "src/lilo/providers/modal/kernel_cache.py"
    with patch.object(
        modal.Volume, "from_name", return_value=sentinel.kernel_cache_volume
    ) as from_name:
        module = runpy.run_path(str(module_path))

    from_name.assert_called_once_with(
        "lilo-kernel-cache", create_if_missing=True, version=2
    )
    assert module["KERNEL_CACHE_ROOT"] == "/root/.cache/kernel-cache"


def test_kernel_cache_env_points_triton_and_inductor_inside_the_mount() -> None:
    assert KERNEL_CACHE_VOLUME_NAME == "lilo-kernel-cache"
    assert KERNEL_CACHE_ENV == {
        "TRITON_CACHE_DIR": "/root/.cache/kernel-cache/triton",
        "TORCHINDUCTOR_CACHE_DIR": "/root/.cache/kernel-cache/inductor",
    }


def test_all_definitions_mount_kernel_cache_and_point_compilers_at_it() -> None:
    for definition in DEFINITIONS:
        assert definition.TRAINER_VOLUMES[KERNEL_CACHE_ROOT] is kernel_cache_volume

        engine_calls = calls_named(source_tree(definition), "run_engine_with_backend")
        if not engine_calls:
            continue  # Single-tenant variants reuse the shared definition's run_trainer.
        assert len(engine_calls) == 1, definition.DEFINITION_ID
        backend_env = keyword(engine_calls[0], "backend_env")
        assert isinstance(backend_env, ast.Dict)
        assert "KERNEL_CACHE_ENV" in spread_names(backend_env), definition.DEFINITION_ID


def test_scoped_trainer_mounts_kernel_cache_and_points_compilers_at_it() -> None:
    tree = source_tree(scoped)
    trainer = next(
        call
        for call in calls_named(tree, "function")
        if any(
            item.arg == "name"
            and isinstance(item.value, ast.Constant)
            and item.value.value == "trainer"
            for item in call.keywords
        )
    )
    volumes = string_dict_entries(keyword(trainer, "volumes"))
    assert isinstance(volumes[KERNEL_CACHE_ROOT], ast.Name)
    assert volumes[KERNEL_CACHE_ROOT].id == "kernel_cache_volume"

    (engine_call,) = calls_named(tree, "run_engine_with_backend")
    backend_env = string_dict_entries(keyword(engine_call, "backend_env"))
    for name, value in KERNEL_CACHE_ENV.items():
        assert isinstance(backend_env[name], ast.Constant)
        assert backend_env[name].value == value
