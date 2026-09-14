import modal

from lilo.backends.miles_config import MILES_REVISION

from .image_dependencies import (
    CORE_PACKAGES,
    MEGATRON_RUNTIME_CHECK,
    MEGATRON_RUNTIME_PACKAGES,
    STITCH_PACKAGE,
)

BASE_IMAGE = "radixark/miles:v0.1.0"
MILES_REPOSITORY = "https://github.com/radixark/miles.git"
MILES_PATH = "/root/miles"
MEGATRON_REPOSITORY = "https://github.com/radixark/Megatron-LM.git"
MEGATRON_REVISION = "8c1e05747eb612b382df2632783df5c83a853646"
MEGATRON_PATH = "/root/Megatron-LM"
BRIDGE_REPOSITORY = "https://github.com/radixark/Megatron-Bridge.git"
BRIDGE_REVISION = "582783a05442245647239e4c5e7d733d7f0e00ea"
BRIDGE_PATH = "/root/Megatron-Bridge"

image = (
    modal.Image.from_registry(BASE_IMAGE)
    .entrypoint([])
    .env(
        {
            "LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        }
    )
    .apt_install("git")
    .run_commands(
        f"rm -rf {MILES_PATH}"
        f" && git clone --filter=blob:none {MILES_REPOSITORY} {MILES_PATH}"
        f" && git -C {MILES_PATH} fetch --depth 1 origin {MILES_REVISION}"
        f" && git -C {MILES_PATH} checkout --detach FETCH_HEAD",
        f"pip install --no-build-isolation --no-deps -e {MILES_PATH}",
        f"git -C {MEGATRON_PATH} fetch --depth 1"
        f" {MEGATRON_REPOSITORY} {MEGATRON_REVISION}"
        f" && git -C {MEGATRON_PATH} checkout --detach FETCH_HEAD",
        f"pip install --no-build-isolation --no-deps -e {MEGATRON_PATH}",
        f"rm -rf {BRIDGE_PATH}"
        f" && git clone --filter=blob:none {BRIDGE_REPOSITORY} {BRIDGE_PATH}"
        f" && git -C {BRIDGE_PATH} fetch --depth 1 origin {BRIDGE_REVISION}"
        f" && git -C {BRIDGE_PATH} checkout --detach FETCH_HEAD",
        "pip uninstall -y megatron-bridge megatron_bridge || true",
        f"pip install --no-build-isolation --no-deps -e {BRIDGE_PATH}",
    )
    .pip_install(*CORE_PACKAGES, STITCH_PACKAGE)
    .pip_install(
        *MEGATRON_RUNTIME_PACKAGES,
        "xxhash==3.7.1",
        "transformers==5.12.1",
        "pyyaml>=6.0.2",
        "opentelemetry-exporter-otlp==1.43.0",
        "opentelemetry-exporter-otlp-proto-grpc==1.43.0",
        "opentelemetry-exporter-otlp-proto-http==1.43.0",
    )
    .run_commands(
        "pip install --no-deps 'peft>=0.18.1'",
        MEGATRON_RUNTIME_CHECK,
        "python -c \"from miles.ray.train.group import TrainerController as T; "
        "assert all(hasattr(T, name) for name in "
        "('load_slot', 'unload_slot', 'forward_backward', "
        "'forward_only_logprobs', 'optim_step', 'save_slot'))\"",
    )
    .add_local_python_source("lilo")
)
