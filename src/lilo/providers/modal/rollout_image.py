import modal

from .image_dependencies import CORE_PACKAGES, STITCH_PACKAGE, TINKER_PACKAGE

SGLANG_IMAGE = "lmsysorg/sglang:v0.5.17"
SGLANG_REPOSITORY = "https://github.com/modal-projects/sglang.git"
SGLANG_BRANCH = "stitch-sglang-v0.5.17"
SGLANG_REVISION = "d050d06437d96196fc68d5b4e5c246408790d537"

image = (
    modal.Image.from_registry(SGLANG_IMAGE)
    .entrypoint([])
    .apt_install("git")
    .run_commands(
        "rm -rf /tmp/stitch-sglang-overlay"
        f" && git clone --filter=blob:none --single-branch --branch {SGLANG_BRANCH}"
        f" {SGLANG_REPOSITORY} /tmp/stitch-sglang-overlay"
        f" && git -C /tmp/stitch-sglang-overlay checkout --detach {SGLANG_REVISION}"
        " && rm -rf /sgl-workspace/sglang/python/sglang"
        " && cp -a /tmp/stitch-sglang-overlay/python/. /sgl-workspace/sglang/python/"
        " && rm -rf /tmp/stitch-sglang-overlay",
    )
    .pip_install(*CORE_PACKAGES, STITCH_PACKAGE, TINKER_PACKAGE)
    .pip_install("huggingface-hub")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_MODULES_CACHE": "/tmp/huggingface/modules",
            "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
            "SGLANG_DISABLE_CUDNN_CHECK": "1",
        }
    )
    .add_local_python_source("lilo")
)
