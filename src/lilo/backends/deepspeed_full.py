from __future__ import annotations

import gc
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import deepspeed
import modal
import torch
import torch.distributed as dist
from tinker import AdamParams, ForwardBackwardOutput, OptimStepResponse
from transformers import AutoModelForCausalLM

from lilo.engine.spmd import DistributedExecutor, initialize_distributed_runtime
from lilo.inference.fft_bulletin import FFTSnapshotBulletin
from lilo.inference.full_delta import FullDeltaWriter

from .contract import (
    CommandBackend,
    ForwardBatch,
    ModelSpec,
    SamplerPublication,
)
from .deepspeed_config import (
    DeepSpeedBackendConfig,
    parse_deepspeed_backend_config,
)
from .deepspeed_runtime.training import (
    apply_adam_params,
    optimizer_grad_norm,
    run_forward_backward,
)
_CAPTURE_ROOT = Path("/tmp/lilo-deepspeed-captures")
_CHECKPOINT_METADATA = "deepspeed_checkpoint.json"
_NATIVE_TAG = "state"


class _StateDictExporter:
    def export_hf_weights(
        self,
        model,
        *,
        cpu: bool,
        show_progress: bool,
        merge_adapter_weights: bool,
    ):
        del show_progress, merge_adapter_weights
        for name, value in model.state_dict().items():
            value = value.detach().contiguous()
            if cpu:
                value = value.cpu()
            yield SimpleNamespace(param_name=name, weight=value)


class DeepSpeedFullBackend(CommandBackend):
    """Single-model Hugging Face backend trained with DeepSpeed ZeRO."""

    def __init__(
        self,
        config: DeepSpeedBackendConfig,
        *,
        checkpoint_dir: Path,
        base_model: str,
        persistence_group,
    ) -> None:
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.base_model = base_model
        self.persistence_group = persistence_group
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.model_id: str | None = None
        self.accumulating = False
        self.optimizer_step = 0
        self._reset_before_accept = False
        self._delta_writer = None
        self._checkpoint_captures: dict[str, dict[str, Any]] = {}
        self._sampler_captures: dict[str, Any] = {}
        self._exporter = _StateDictExporter()
        self._create_engine(self.config.hf_checkpoint)

    def _create_engine(self, checkpoint: str) -> None:
        dtype = torch.bfloat16 if self.config.bf16 else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            dtype=dtype,
            local_files_only=True,
            attn_implementation="sdpa",
        )
        model.config.use_cache = False
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.optimizer.learning_rate,
            betas=tuple(self.config.optimizer.betas),
            eps=self.config.optimizer.eps,
            weight_decay=self.config.optimizer.weight_decay,
        )
        self.engine, self.optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            optimizer=optimizer,
            config=self.config.engine_config(self.world_size),
            dist_init_required=False,
        )

    def _destroy_engine(self) -> None:
        del self.optimizer
        del self.engine
        gc.collect()
        torch.cuda.empty_cache()

    def accept_model(self, model_id: str, spec: ModelSpec) -> None:
        if spec.parameterization != "full" or spec.lora_config is not None:
            raise ValueError("DeepSpeed backend requires full parameterization")
        if spec.base_model != self.base_model:
            raise ValueError(
                f"base model {spec.base_model!r} does not match deployment "
                f"{self.base_model!r}"
            )
        if self.model_id == model_id:
            return
        if self.model_id is not None:
            raise ValueError(f"DeepSpeed worker already hosts model {self.model_id}")
        if self._reset_before_accept:
            self._destroy_engine()
            self._create_engine(self.config.hf_checkpoint)
        self.model_id = model_id
        self.accumulating = False
        self.optimizer_step = 0
        self._reset_before_accept = False
        self._delta_writer = None
        self._checkpoint_captures.clear()
        self._sampler_captures.clear()

    def forward_backward(
        self,
        batch: ForwardBatch,
    ) -> tuple[ForwardBackwardOutput, ...]:
        for item in batch.items:
            self._require_model(item.model_id)
        if not batch.forward_only and not self.accumulating:
            self.engine.zero_grad()
        outputs = run_forward_backward(
            self.engine,
            batch,
            rank=self.rank,
            world_size=self.world_size,
            max_sequence_length=self.config.max_sequence_length,
        )
        if not batch.forward_only:
            self.accumulating = True
        return outputs

    def optim_step(
        self,
        model_ids: tuple[str, ...],
        adam: AdamParams,
    ) -> tuple[OptimStepResponse, ...]:
        if model_ids != (self.model_id,):
            raise ValueError("DeepSpeed full backend requires its single model")
        self._require_model(model_ids[0])
        if not self.accumulating:
            raise ValueError(f"model {model_ids[0]} has no accumulated gradients")
        apply_adam_params(self.engine, adam)
        self.engine.step()
        grad_norm = optimizer_grad_norm(self.engine)
        self.engine.zero_grad()
        self.accumulating = False
        self.optimizer_step += 1
        return (
            OptimStepResponse(
                metrics={
                    "grad_norm:mean": grad_norm,
                    "update_successful:mean": 1.0,
                }
            ),
        )

    def capture_checkpoint(
        self,
        model_id: str,
        snapshot_id: str,
        *,
        destination: str,
        include_optimizer: bool,
    ) -> None:
        self._require_model(model_id)
        capture_path = _CAPTURE_ROOT / snapshot_id
        if self.rank == 0:
            shutil.rmtree(capture_path, ignore_errors=True)
            capture_path.mkdir(parents=True)
            state = {
                name: value.detach().cpu().clone()
                for name, value in self.engine.module.state_dict().items()
            }
            self.engine.module.save_pretrained(
                capture_path,
                state_dict=state,
                safe_serialization=True,
                max_shard_size="2GB",
            )
        dist.barrier()
        if include_optimizer:
            self.engine.save_checkpoint(
                str(capture_path / "native"),
                tag=_NATIVE_TAG,
                client_state={"optimizer_step": self.optimizer_step},
                save_latest=False,
            )
        if self.rank == 0:
            (capture_path / _CHECKPOINT_METADATA).write_text(
                json.dumps(
                    {
                        "format": "lilo.deepspeed.checkpoint",
                        "version": 1,
                        "base_model": self.base_model,
                        "world_size": self.world_size,
                        "zero_stage": self.config.zero_stage,
                        "has_optimizer": include_optimizer,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            (capture_path / "metadata.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "base_model": self.base_model,
                        "engine_definition_id": os.environ["LILO_DEFINITION_ID"],
                        "parameterization": {"type": "full"},
                        "lora_config": None,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        dist.barrier()
        self._checkpoint_captures[snapshot_id] = {
            "model_id": model_id,
            "destination": destination,
            "capture_path": str(capture_path),
            "target_path": str(
                self.checkpoint_dir / model_id / "weights" / destination
            ),
        }

    def persist_checkpoint(
        self,
        snapshot_id: str,
        destination: str,
        *,
        overwrite: bool = False,
    ) -> str:
        capture = self._checkpoint_captures[snapshot_id]
        capture_path = Path(capture["capture_path"])
        target_path = Path(capture["target_path"])
        try:
            local_error = None
            if self.rank == 0:
                try:
                    if target_path.exists() and not overwrite:
                        raise FileExistsError(
                            f"checkpoint already exists: {target_path}"
                        )
                    if target_path.exists():
                        shutil.rmtree(target_path)
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(capture_path, target_path)
                except Exception as exc:  # noqa: BLE001 - broadcast rank-zero I/O
                    local_error = f"{type(exc).__name__}: {exc}"
            holder = [local_error]
            if self.persistence_group is not None:
                dist.broadcast_object_list(
                    holder,
                    src=0,
                    group=self.persistence_group,
                )
            if holder[0] is not None:
                raise RuntimeError(holder[0])
            self._commit_checkpoint_volume()
            return str(target_path)
        finally:
            self._checkpoint_captures.pop(snapshot_id, None)
            if self.rank == 0:
                shutil.rmtree(capture_path, ignore_errors=True)

    def load_checkpoint(
        self,
        model_id: str,
        uri: str,
        *,
        restore_optimizer: bool = False,
    ) -> None:
        self._require_model(model_id)
        self._reload_checkpoint_volume()
        source = Path(uri)
        metadata = json.loads(
            (source / _CHECKPOINT_METADATA).read_text(encoding="utf-8")
        )
        expected = {
            "base_model": self.base_model,
            "world_size": self.world_size,
            "zero_stage": self.config.zero_stage,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(
                    f"DeepSpeed checkpoint {key} mismatch: "
                    f"expected {value!r}, got {metadata.get(key)!r}"
                )
        if restore_optimizer and not metadata.get("has_optimizer"):
            raise ValueError("DeepSpeed checkpoint contains no optimizer state")

        load_path, client_state = self.engine.load_checkpoint(
            str(source / "native"),
            tag=_NATIVE_TAG,
            load_module_strict=True,
            load_optimizer_states=restore_optimizer,
            load_lr_scheduler_states=False,
        )
        if load_path is None:
            raise RuntimeError(f"DeepSpeed did not load checkpoint {source}")
        if restore_optimizer:
            self.optimizer_step = int(client_state.get("optimizer_step", 0))
        else:
            base_optimizer = getattr(
                self.engine.optimizer,
                "optimizer",
                self.engine.optimizer,
            )
            base_optimizer.state.clear()
            self.optimizer_step = 0
        self.engine.zero_grad()
        self.accumulating = False
        self._delta_writer = None
        self._sampler_captures.clear()

    def capture_sampler_snapshot(
        self,
        model_id: str,
        capture_id: str,
        requested_version: int,
    ) -> SamplerPublication:
        del requested_version
        self._require_model(model_id)
        if self.accumulating:
            raise ValueError("cannot publish with accumulated gradients")
        bulletin_root = os.environ["LILO_BULLETIN_ROOT"]
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
                current.version,
                self.base_model,
                self.optimizer_step,
            )
        publish_version = durable_version + 1
        if self._delta_writer is None:
            self._delta_writer = FullDeltaWriter()
        snapshot = self._delta_writer.capture(
            exporter=self._exporter,
            model=self.engine.module,
            model_id=model_id,
            publish_version=publish_version,
            optimizer_step=self.optimizer_step,
            base_model=self.base_model,
            hf_checkpoint=self.config.hf_checkpoint,
            bulletin_root=bulletin_root,
        )
        self._sampler_captures[capture_id] = snapshot
        return SamplerPublication(
            publish_version,
            self.base_model,
            self.optimizer_step,
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

    def unload_model(self, model_id: str) -> None:
        if self.model_id != model_id:
            return
        self.engine.zero_grad()
        self.model_id = None
        self.accumulating = False
        self.optimizer_step = 0
        self._reset_before_accept = True
        self._delta_writer = None
        self._checkpoint_captures.clear()
        self._sampler_captures.clear()

    def close(self) -> None:
        self._destroy_engine()
        dist.destroy_process_group()

    def _require_model(self, model_id: str) -> None:
        if self.model_id != model_id:
            raise KeyError(f"unknown model: {model_id}")

    def _reload_checkpoint_volume(self) -> None:
        dist.barrier()
        if self.rank == 0:
            modal.Volume.from_name(os.environ["LILO_CHECKPOINT_VOLUME"]).reload()
        dist.barrier()

    def _commit_checkpoint_volume(self) -> None:
        if self.persistence_group is not None:
            dist.barrier(group=self.persistence_group)
        if self.persistence_group is None or dist.get_rank(
            group=self.persistence_group
        ) == 0:
            modal.Volume.from_name(os.environ["LILO_CHECKPOINT_VOLUME"]).commit()


def build_executor() -> DistributedExecutor:
    (
        command_group,
        checkpoint_persistence_group,
        sampler_persistence_group,
    ) = initialize_distributed_runtime()
    config, checkpoint_dir = parse_deepspeed_backend_config(
        json.loads(os.environ["LILO_BACKEND_CONFIG"])
    )
    backend = DeepSpeedFullBackend(
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
