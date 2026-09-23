import ast
import runpy
from pathlib import Path
from unittest.mock import patch, sentinel

import modal

from lilo.engines import qwen3_5_4b_full_64k
from lilo.providers.modal.app import DEFINITIONS
from lilo.providers.modal.kernel_cache import (
    KERNEL_CACHE_ENV,
    KERNEL_CACHE_ROOT,
    KERNEL_CACHE_VOLUME_NAME,
    kernel_cache_volume,
)
from lilo.providers.modal.scoped import build_app
from lilo.run import Pool


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
    assert set(KERNEL_CACHE_ENV) == {"TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"}
    for value in KERNEL_CACHE_ENV.values():
        assert value.startswith(KERNEL_CACHE_ROOT + "/")


def test_all_definitions_mount_kernel_cache_and_set_compiler_env() -> None:
    for definition in DEFINITIONS:
        assert definition.TRAINER_VOLUMES[KERNEL_CACHE_ROOT] is kernel_cache_volume

        tree = ast.parse(Path(definition.__file__).read_text())
        decorators = [
            decorator
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            for decorator in node.decorator_list
            if isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "function"
        ]
        assert len(decorators) == 1, definition.DEFINITION_ID
        keywords = {keyword.arg: keyword.value for keyword in decorators[0].keywords}
        assert isinstance(keywords["volumes"], ast.Name)
        assert keywords["volumes"].id == "TRAINER_VOLUMES"
        env = keywords["env"]
        assert isinstance(env, ast.Call) and isinstance(env.func, ast.Name)
        assert env.func.id == "trainer_deployment_env", definition.DEFINITION_ID


def test_scoped_trainer_mounts_kernel_cache_and_sets_compiler_env() -> None:
    captured = {}
    original = modal.App.function

    def recording_function(self, *args, **kwargs):
        if kwargs.get("name") == "trainer":
            captured.update(kwargs)
        return original(self, *args, **kwargs)

    with patch.object(modal.App, "function", recording_function):
        build_app(
            qwen3_5_4b_full_64k(), "kernel-cache-test", "kernel-cache-test",
            "test-key", 1, Pool(), Pool(), "test-checkpoints",
            modal.Secret.from_dict({}),
        )

    assert captured["volumes"][KERNEL_CACHE_ROOT] is kernel_cache_volume
    for name, value in KERNEL_CACHE_ENV.items():
        assert captured["env"][name] == value
