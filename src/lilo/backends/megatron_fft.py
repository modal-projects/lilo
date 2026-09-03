from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from megatron.bridge import AutoBridge
from megatron.core import parallel_state
from tinker import AdamParams, ForwardBackwardOutput, OptimStepResponse

from lilo.engine.spmd import DistributedExecutor, initialize_distributed_runtime
from lilo.inference.fft_bulletin import FFTSnapshotBulletin

from .contract import (
    CommandBackend,
    ForwardBatch,
    ModelSpec,
    SamplerPublication,
)
from .megatron_config import parse_backend_config
from .megatron_runtime.common.checkpoint_io import reload_checkpoint_volume
from .megatron_runtime.common.config import EngineModelConfig
from .megatron_runtime.common.distributed import initialize_megatron
from .megatron_runtime.common.forward_backward import (
    add_packing_metrics,
    build_outputs,
    prepare_microbatches,
    run_megatron_pipeline,
    synchronize_collectors,
)
from .megatron_runtime.fft.checkpoint import (
    capture_fft_checkpoint,
    capture_hf_weights,
    create_fft_checkpoint_metadata,
    fft_checkpoint_path,
    load_fft_training_checkpoint,
    restore_fft_optimizer_state,
    synchronize_checkpoint_preflight,
    write_fft_checkpoint,
)
from .megatron_runtime.fft.delta import FFTDeltaWriter
from .megatron_runtime.fft.model import (
    create_fft_model_and_optimizer,
    create_fft_optimizer,
)
from .megatron_runtime.fft.optimizer import run_fft_optimizer_step


class FFTMegatronBackend(CommandBackend):
    """Single-model Megatron backend for full-parameter training."""

    def __init__(
        self,
        config: EngineModelConfig,
        *,
        checkpoint_dir: Path,
        base_model: str,
        persistence_group,
    ) -> None:
        self.config = config
        self.base_model = base_model
        self.checkpoint_dir = checkpoint_dir
        self.persistence_group = persistence_group
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        initialize_megatron(config)

        self.model_id: str | None = None
        self.accumulating = False
        self.optimizer_step = 0
        self._reset_before_accept = False
        self._delta_writer = None
        self._sampler_captures: dict[str, Any] = {}
        self._checkpoint_captures: dict[str, dict[str, Any]] = {}
        self._create_model()

    def _create_model(self) -> None:
        self.model, self.optimizer, self.bridge = create_fft_model_and_optimizer(
            self.config
        )

    def _reset_model(self) -> None:
        del self.optimizer
        gc.collect()
        torch.cuda.empty_cache()
        self.bridge.load_hf_weights(self.model)
        self.optimizer = create_fft_optimizer(self.config, self.model)
        self._zero_grad()
        self._reset_before_accept = False

    def accept_model(self, model_id: str, spec: ModelSpec) -> None:
        if spec.parameterization != "full" or spec.lora_config is not None:
            raise ValueError("FFT backend requires full parameterization")
        if spec.base_model != self.base_model:
            raise ValueError(
                f"base model {spec.base_model!r} does not match deployment "
                f"{self.base_model!r}"
            )
        if self.model_id == model_id:
            return
        if self.model_id is not None:
            raise ValueError(f"FFT worker already hosts model {self.model_id}")
        if self._reset_before_accept:
            self._reset_model()
        self.model_id = model_id
        self.accumulating = False
        self.optimizer_step = 0
        self._delta_writer = None
        self._sampler_captures.clear()
        self._checkpoint_captures.clear()

    def _delete_model(self) -> None:
        self._zero_grad()
        self.model_id = None
        self.accumulating = False
        self.optimizer_step = 0
        self._reset_before_accept = True
        self._delta_writer = None
        self._sampler_captures.clear()
        self._checkpoint_captures.clear()

    def forward_backward(
        self,
        batch: ForwardBatch,
    ) -> tuple[ForwardBackwardOutput, ...]:
        for item in batch.items:
            self._require_model(item.model_id)

        microbatches, packing_metrics = prepare_microbatches(
            batch,
            None,
            config=self.config,
            rank=self.rank,
        )
        if not batch.forward_only and not self.accumulating:
            self._zero_grad()
        output_collector, metric_collector = run_megatron_pipeline(
            self.model,
            self.optimizer,
            microbatches,
            forward_only=batch.forward_only,
            route_adapters=False,
            defer_fp32_logits=self.config.defer_fp32_logits,
        )
        output_collector, metric_collector = synchronize_collectors(
            output_collector,
            metric_collector,
        )
        outputs = build_outputs(batch, output_collector, metric_collector)
        add_packing_metrics(outputs, packing_metrics)
        if not batch.forward_only:
            self.accumulating = True
        return outputs

    def optim_step(
        self,
        model_ids: tuple[str, ...],
        adam: AdamParams,
    ) -> tuple[OptimStepResponse, ...]:
        return (self._optim_step(model_ids[0], adam),)

    def _optim_step(
        self,
        model_id: str,
        adam: AdamParams,
    ) -> OptimStepResponse:
        self._require_model(model_id)
        if not self.accumulating:
            raise ValueError(f"model {model_id} has no accumulated gradients")
        successful, grad_norm = run_fft_optimizer_step(
            self.optimizer,
            self.model,
            adam=adam,
        )
        self.accumulating = False
        if successful:
            self.optimizer_step += 1
        return OptimStepResponse(
            metrics={
                "grad_norm:mean": grad_norm,
                "update_successful:mean": float(successful),
            },
        )

    def capture_sampler_snapshot(
        self,
        model_id: str,
        capture_id: str,
        requested_version: int,
    ) -> SamplerPublication:
        # FFT publications must be contiguous, while the requested
        # version is the model operation sequence and can contain gaps.
        del requested_version

        self._require_model(model_id)

        # require optim step was run right before publishing
        if self.accumulating:
            raise ValueError("cannot publish with accumulated gradients")
        bulletin_root = os.environ["LILO_BULLETIN_ROOT"]
        base_model = os.environ.get(
            "LILO_BASE_MODEL",
            self.config.hf_checkpoint,
        )
        bulletin = FFTSnapshotBulletin(bulletin_root)
        current = bulletin.read_latest(model_id)
        durable_version = current.version if current is not None else 0
        published_step = int(
            bulletin.metadata(current).get("optimizer_step", 0)
            if current is not None
            else 0
        )

        if (
            current is not None
            and current.version > 0
            and self.optimizer_step == published_step
            and self._delta_writer is not None
            and self._delta_writer.is_aligned_with(current)
        ):
            self._sampler_captures[capture_id] = None
            return SamplerPublication(
                publish_version=current.version,
                base_model=base_model,
                optimizer_step=self.optimizer_step,
            )
        publish_version = durable_version + 1

        if self._delta_writer is None:
            self._delta_writer = FFTDeltaWriter()
        snapshot = self._delta_writer.capture(
            bridge=self.bridge,
            model=self.model,
            model_id=model_id,
            publish_version=publish_version,
            optimizer_step=self.optimizer_step,
            base_model=base_model,
            hf_checkpoint=self.config.hf_checkpoint,
            bulletin_root=bulletin_root,
        )
        self._sampler_captures[capture_id] = snapshot
        return SamplerPublication(
            publish_version=publish_version,
            base_model=base_model,
            optimizer_step=self.optimizer_step,
        )

    def persist_sampler_snapshot(self, capture_id: str) -> None:
        snapshot = self._sampler_captures[capture_id]
        try:
            if snapshot is not None:
                self._delta_writer.persist(
                    snapshot,
                    bulletin_root=os.environ["LILO_BULLETIN_ROOT"],
                    bulletin_volume=os.environ["LILO_BULLETIN_VOLUME"],
                )
        finally:
            self._sampler_captures.pop(capture_id, None)

    def capture_checkpoint(
        self,
        model_id: str,
        snapshot_id: str,
        *,
        destination: str,
        include_optimizer: bool,
    ) -> None:
        self._require_model(model_id)
        hf_weights = capture_hf_weights(self.bridge, self.model)
        metadata = create_fft_checkpoint_metadata(
            self.config,
            checkpoint_id=snapshot_id,
            base_model=self.base_model,
            include_optimizer=include_optimizer,
            world_size=self.world_size,
        )
        checkpoint = capture_fft_checkpoint(
            self.model,
            self.optimizer,
            metadata=metadata,
            include_optimizer=include_optimizer,
        )
        checkpoint.update(
            optimizer_step=self.optimizer_step,
        )
        capture = {
            "model_id": model_id,
            "destination": destination,
            "path": str(self.checkpoint_dir / model_id / "weights" / destination),
            "checkpoint": checkpoint,
            "hf_weights": hf_weights,
            "metadata": {
                "schema_version": 1,
                "base_model": self.base_model,
                "engine_definition_id": os.environ["LILO_DEFINITION_ID"],
                "parameterization": {"type": "full"},
                "lora_config": None,
            },
        }
        self._checkpoint_captures[snapshot_id] = capture

    def persist_checkpoint(
        self,
        snapshot_id: str,
        destination: str,
        *,
        overwrite: bool = False,
    ) -> str:
        try:
            capture = self._checkpoint_captures[snapshot_id]
            if not overwrite and fft_checkpoint_path(capture["path"]).exists():
                raise FileExistsError(f"checkpoint already exists: {capture['path']}")
            return write_fft_checkpoint(
                capture["path"],
                capture["checkpoint"],
                hf_weights=capture["hf_weights"],
                hf_checkpoint=self.config.hf_checkpoint,
                metadata=capture["metadata"],
                persistence_group=self.persistence_group,
            )
        finally:
            self._checkpoint_captures.pop(snapshot_id, None)

    def load_checkpoint(
        self,
        model_id: str,
        uri: str,
        *,
        restore_optimizer: bool = False,
    ) -> None:
        self._require_model(model_id)

        reload_checkpoint_volume()
        if restore_optimizer:
            try:
                checkpoint, metadata = load_fft_training_checkpoint(
                    uri,
                    self.config,
                    base_model=self.base_model,
                    world_size=self.world_size,
                )
            except Exception as exc:
                synchronize_checkpoint_preflight(exc)
                raise
            synchronize_checkpoint_preflight(
                None,
                metadata_identity=metadata.identity(),
            )
            for chunk, state in zip(self.model, checkpoint["model"], strict=True):
                chunk.load_state_dict(state, strict=True)
            self.optimizer.reload_model_params()
            restore_fft_optimizer_state(
                self.optimizer,
                checkpoint["optimizer"],
                use_distributed_optimizer=self.config.use_distributed_optimizer,
            )
            self.optimizer_step = int(checkpoint.get("optimizer_step", 0))
            self._zero_grad()
            self.accumulating = False
        else:
            try:
                bridge = AutoBridge.from_hf_pretrained(
                    uri,
                    trust_remote_code=False,
                    local_files_only=True,
                )
            except Exception as exc:
                synchronize_checkpoint_preflight(exc)
                raise
            synchronize_checkpoint_preflight(None)

            bridge.load_hf_weights(self.model)
            del self.optimizer
            gc.collect()
            torch.cuda.empty_cache()
            self.optimizer = create_fft_optimizer(self.config, self.model)
            self.optimizer_step = 0
            self._zero_grad()
            self.accumulating = False

        self._delta_writer = None
        self._sampler_captures.clear()

    def unload_model(self, model_id: str) -> None:
        if self.model_id != model_id:
            return
        self._delete_model()

    def close(self) -> None:
        self._shutdown()

    def _require_model(self, model_id: str) -> None:
        if self.model_id != model_id:
            raise KeyError(f"unknown model: {model_id}")

    def _zero_grad(self) -> None:
        for chunk in self.model:
            chunk.zero_grad_buffer()
        self.optimizer.zero_grad()

    def _shutdown(self) -> None:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def build_executor() -> DistributedExecutor:
    (
        command_group,
        checkpoint_persistence_group,
        sampler_persistence_group,
    ) = initialize_distributed_runtime()
    config, checkpoint_dir = parse_backend_config(
        json.loads(os.environ["LILO_BACKEND_CONFIG"])
    )
    backend = FFTMegatronBackend(
        config,
        checkpoint_dir=checkpoint_dir,
        base_model=os.environ["LILO_BASE_MODEL"],
        persistence_group=checkpoint_persistence_group,
    )
    return DistributedExecutor(
        backend,
        command_group=command_group,
        checkpoint_persistence_group=checkpoint_persistence_group,
        sampler_persistence_group=sampler_persistence_group,
    )
