from __future__ import annotations

import modal

CHECKPOINT_VOLUME_NAME = "lilo-checkpoints"
CHECKPOINT_ROOT = "/checkpoints"

checkpoint_volume = modal.Volume.from_name(
    CHECKPOINT_VOLUME_NAME,
    create_if_missing=True,
    version=2,
)
