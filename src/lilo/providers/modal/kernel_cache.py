"""Persistent Triton/TorchInductor kernel cache shared across trainer containers.

Trainers run in single-use containers, so without this every instance compiles
its kernels from scratch. Mounting one v2 Volume at ``KERNEL_CACHE_ROOT`` and
pointing the compilers at it lets later containers reuse earlier compiles.
"""

from __future__ import annotations

import modal

KERNEL_CACHE_VOLUME_NAME = "lilo-kernel-cache"
KERNEL_CACHE_ROOT = "/root/.cache/kernel-cache"

kernel_cache_volume = modal.Volume.from_name(
    KERNEL_CACHE_VOLUME_NAME,
    create_if_missing=True,
    version=2,
)

KERNEL_CACHE_ENV = {
    "TRITON_CACHE_DIR": f"{KERNEL_CACHE_ROOT}/triton",
    "TORCHINDUCTOR_CACHE_DIR": f"{KERNEL_CACHE_ROOT}/inductor",
}
