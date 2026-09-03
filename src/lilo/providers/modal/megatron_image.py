import modal

from .image_dependencies import (
    CORE_PACKAGES,
    MEGATRON_RUNTIME_CHECK,
    MEGATRON_RUNTIME_PACKAGES,
    STITCH_PACKAGE,
)

BASE_IMAGE = "radixark/miles:v0.1.0"
MEGATRON_CORE_REPOSITORY = "https://github.com/NVIDIA/Megatron-LM.git"
# This revision includes the Qwen3.5 output-gate slice required when
# tensor parallelism exceeds the number of query groups.
MEGATRON_CORE_REVISION = "2d1fa8d372a3990b0bb1334cd686f15005ee138f"
MEGATRON_CORE_PATH = "/root/Megatron-LM"
MEGATRON_BRIDGE_REPOSITORY = "https://github.com/cnnradams/Megatron-Bridge.git"
MEGATRON_BRIDGE_REVISION = "6aa2ba1e36013179492abda123f395873b30a3c7"
MEGATRON_BRIDGE_PATH = "/root/Megatron-Bridge"

image = (
    modal.Image.from_registry(BASE_IMAGE)
    .entrypoint([])
    .env({"LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"})
    .apt_install("git")
    .run_commands(
        f"git -C {MEGATRON_CORE_PATH} fetch --depth 1"
        f" {MEGATRON_CORE_REPOSITORY} {MEGATRON_CORE_REVISION}"
        f" && git -C {MEGATRON_CORE_PATH} checkout --detach FETCH_HEAD",
        f"pip install --no-build-isolation --no-deps -e {MEGATRON_CORE_PATH}",
        f"mkdir -p {MEGATRON_BRIDGE_PATH}"
        f" && git -C {MEGATRON_BRIDGE_PATH} init"
        f" && git -C {MEGATRON_BRIDGE_PATH} remote add origin"
        f" {MEGATRON_BRIDGE_REPOSITORY}"
        f" && git -C {MEGATRON_BRIDGE_PATH} fetch --depth 1 origin"
        f" {MEGATRON_BRIDGE_REVISION}"
        f" && git -C {MEGATRON_BRIDGE_PATH} checkout --detach FETCH_HEAD",
        "pip uninstall -y megatron-bridge megatron_bridge || true",
        'python3 -c "import importlib.util as u, os, shutil; '
        "s = u.find_spec('megatron.bridge'); "
        "p = os.path.dirname(s.origin) if s and getattr(s, 'origin', None) "
        'else None; shutil.rmtree(p) if p else None" || true',
        f"pip install --no-build-isolation --no-deps -e {MEGATRON_BRIDGE_PATH}",
    )
    .pip_install(*CORE_PACKAGES, STITCH_PACKAGE)
    .pip_install(*MEGATRON_RUNTIME_PACKAGES)
    .run_commands(MEGATRON_RUNTIME_CHECK)
    .add_local_python_source("lilo")
)
