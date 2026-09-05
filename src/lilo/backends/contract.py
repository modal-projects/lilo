from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

from tinker import (
    AdamParams,
    Datum,
    ForwardBackwardOutput,
    LoraConfig,
    OptimStepResponse,
)

LossFn = Literal[
    "cross_entropy",
    "importance_sampling",
    "ppo",
    "cispo",
    "dro",
    "dppo",
]


@dataclass(frozen=True)
class ModelSpec:
    """base model x parameterization x (optional lora config)

    consumed by accept_model to creat new model instance
    """

    base_model: str
    parameterization: Literal["lora", "full"]
    lora_config: LoraConfig | None = None

    def __post_init__(self) -> None:
        if self.parameterization == "lora" and self.lora_config is None:
            raise ValueError("lora_config is required for lora parameterization")
        if self.parameterization == "full" and self.lora_config is not None:
            raise ValueError("lora_config must be absent for full parameterization")


@dataclass(frozen=True)
class ForwardItem:
    model_id: str
    data: tuple[Datum, ...]


@dataclass(frozen=True)
class ForwardBatch:
    """group compatible tinker forward-backward operations together"""

    items: tuple[ForwardItem, ...]
    loss_fn: LossFn
    loss_fn_config: Mapping[str, float] = field(default_factory=dict)
    forward_only: bool = False


@dataclass(frozen=True)
class SamplerPublication:
    """record for sampler publication to bulletin"""

    publish_version: int
    base_model: str
    optimizer_step: int


class CommandBackend(Protocol):
    def accept_model(self, model_id: str, spec: ModelSpec) -> None: ...

    def forward_backward(
        self,
        batch: ForwardBatch,
    ) -> tuple[ForwardBackwardOutput, ...]:
        """Execute forward or forward-backward operations.

        When forward_only is false, gradients accumulate without applying
        an optimizer update. Outputs correspond positionally to batch.items.
        """
        ...

    def optim_step(
        self,
        model_ids: tuple[str, ...],
        adam: AdamParams,
    ) -> tuple[OptimStepResponse, ...]:
        """Apply one optimizer update per model.

        Each model must have accumulated gradients. Outputs correspond
        positionally to model_ids.
        """
        ...

    def capture_checkpoint(
        self,
        model_id: str,
        snapshot_id: str,
        *,
        destination: str,
        include_optimizer: bool,
    ) -> None:
        """Capture immutable backend-local training state under snapshot_id.

        Once this returns, subsequent GPU operations may proceed without
        changing the captured state. destination is reserved for the later
        persistence call.
        """
        ...

    def persist_checkpoint(
        self,
        snapshot_id: str,
        destination: str,
        *,
        overwrite: bool = False,
    ) -> str:
        """Persist a captured checkpoint and return its durable URI.

        Success means every required rank shard is durable and loadable.
        """
        ...

    def load_checkpoint(
        self,
        model_id: str,
        uri: str,
        *,
        restore_optimizer: bool = False,
    ) -> None:
        """Replace model state from a durable checkpoint.

        Returning means the model is ready for subsequent training.
        """
        ...

    def capture_sampler_snapshot(
        self,
        model_id: str,
        capture_id: str,
        requested_version: int,
    ) -> SamplerPublication:
        """Capture immutable backend-local weights for rollout.

        Once this returns, subsequent GPU operations may proceed. No
        sampler-visible pointer may be advanced by this method. The returned
        receipt describes the eventual publication and is returned to the
        client only after finalization.
        """
        ...

    def persist_sampler_snapshot(self, capture_id: str) -> None:
        """Persist and publish a detached sampler snapshot.

        This runs asynchronously and may overlap later GPU operations. It
        releases capture resources on success or failure.
        """
        ...

    def unload_model(self, model_id: str) -> None:
        """Release per-model resources.

        Unloading an absent model is idempotent.
        """
        ...

    def close(self) -> None:
        """Release backend and distributed resources."""
        ...
