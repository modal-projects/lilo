from __future__ import annotations

import modal

KERNEL_CACHE_VOLUME_NAME = "lilo-kernel-cache"
KERNEL_CACHE_ROOT = "/root/.cache/kernel-cache"
KERNEL_CACHE_ENV = {
    "TRITON_CACHE_DIR": f"{KERNEL_CACHE_ROOT}/triton",
    "TORCHINDUCTOR_CACHE_DIR": f"{KERNEL_CACHE_ROOT}/inductor",
}

kernel_cache_volume = modal.Volume.from_name(
    KERNEL_CACHE_VOLUME_NAME,
    create_if_missing=True,
    version=2,
)
