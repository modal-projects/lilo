import ast
import runpy
from pathlib import Path
from unittest.mock import patch, sentinel

import modal

from lilo.providers.modal.app import DEFINITIONS
from lilo.providers.modal.kernel_cache import (
    KERNEL_CACHE_ENV,
    KERNEL_CACHE_ROOT,
    KERNEL_CACHE_VOLUME_NAME,
    kernel_cache_volume,
)


def test_kernel_cache_creates_one_v2_volume_without_live_lookup() -> None:
    module_path = Path(__file__).parents[2] / "src/lilo/providers/modal/kernel_cache.py"
    with patch.object(
        modal.Volume,
        "from_name",
        return_value=sentinel.kernel_cache_volume,
    ) as from_name:
        module = runpy.run_path(str(module_path))

    from_name.assert_called_once_with(
        "lilo-kernel-cache",
        create_if_missing=True,
        version=2,
    )
    assert module["KERNEL_CACHE_ROOT"] == "/root/.cache/kernel-cache"
    assert module["KERNEL_CACHE_ENV"] == {
        "TRITON_CACHE_DIR": "/root/.cache/kernel-cache/triton",
        "TORCHINDUCTOR_CACHE_DIR": "/root/.cache/kernel-cache/inductor",
    }


def test_all_definitions_mount_the_shared_kernel_cache() -> None:
    assert KERNEL_CACHE_VOLUME_NAME == "lilo-kernel-cache"
    for definition in DEFINITIONS:
        assert definition.TRAINER_VOLUMES[KERNEL_CACHE_ROOT] is kernel_cache_volume
    assert KERNEL_CACHE_ENV["TRITON_CACHE_DIR"].startswith(KERNEL_CACHE_ROOT + "/")
    assert KERNEL_CACHE_ENV["TORCHINDUCTOR_CACHE_DIR"].startswith(
        KERNEL_CACHE_ROOT + "/"
    )


def test_all_definitions_point_compilers_at_the_kernel_cache() -> None:
    """Every trainer's backend_env starts from KERNEL_CACHE_ENV so definition-specific
    entries can still override the cache locations."""
    for definition in DEFINITIONS:
        tree = ast.parse(Path(definition.__file__).read_text())
        engine_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_engine_with_backend"
        ]
        if not engine_calls:
            continue
        assert len(engine_calls) == 1, definition.DEFINITION_ID
        backend_env = next(
            keyword.value
            for keyword in engine_calls[0].keywords
            if keyword.arg == "backend_env"
        )
        assert isinstance(backend_env, ast.Dict), definition.DEFINITION_ID
        first_key, first_value = backend_env.keys[0], backend_env.values[0]
        assert first_key is None, definition.DEFINITION_ID
        assert isinstance(first_value, ast.Name), definition.DEFINITION_ID
        assert first_value.id == "KERNEL_CACHE_ENV", definition.DEFINITION_ID
