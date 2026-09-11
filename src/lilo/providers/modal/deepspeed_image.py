import modal

from .image_dependencies import CORE_PACKAGES, STITCH_PACKAGE, TINKER_PACKAGE

BASE_IMAGE = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime"

image = (
    modal.Image.from_registry(BASE_IMAGE)
    .entrypoint([])
    .apt_install("git", "ninja-build")
    .env(
        {
            "DS_BUILD_OPS": "0",
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
    .pip_install(
        *CORE_PACKAGES,
        STITCH_PACKAGE,
        TINKER_PACKAGE,
        "accelerate",
        "deepspeed",
        "huggingface-hub",
        "safetensors",
        "transformers>=5.15,<5.16",
    )
    .add_local_python_source("lilo")
)
