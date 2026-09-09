from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stitch.types import VersionRef
from tinker import AdamParams, ForwardBackwardOutput, OptimStepResponse

from lilo.engine.spmd import DistributedExecutor
from lilo.inference.bulletin import SnapshotBulletin

from .contract import (
    CommandBackend,
    ForwardBatch,
    ModelSpec,
    SamplerPublication,
)
from .miles_config import MILES_REVISION, MilesBackendConfig, parse_backend_config
from .miles_runtime.data import build_outputs, prepare_batch
from .miles_runtime.runtime import MilesRuntime


@dataclass(slots=True)
class MilesJobState:
    rank: int
    alpha: float
    seed: int | None
    train_attn: bool
    train_mlp: bool
    train_unembed: bool
    accumulating: bool = False
    optimizer_step: int = 0


class MilesCommandBackend(CommandBackend):
    """Lilo command protocol backed by one Miles Ray trainer group."""

    def __init__(
        self,
        config: MilesBackendConfig,
        *,
        checkpoint_dir: Path,
        capture_dir: Path,
        base_model: str,
        runtime: Any | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.base_model = base_model
        self.checkpoint_dir = checkpoint_dir
        self.capture_dir = capture_dir
        self.runtime = runtime if runtime is not None else MilesRuntime(config)
        self.max_slots = config.max_lora_slots
        self.free_slots = set(range(self.max_slots))
        self.jobs: dict[str, MilesJobState] = {}
        self.job_to_slot: dict[str, int] = {}
        self.slot_to_job: dict[int, str] = {}
        self._checkpoint_captures: dict[str, dict[str, Any]] = {}
        self._sampler_captures: dict[str, dict[str, Any]] = {}
        self._closed = False

    def accept_model(self, model_id: str, spec: ModelSpec) -> None:
        if spec.parameterization != "lora" or spec.lora_config is None:
            raise ValueError("Miles backend requires lora parameterization")
        if spec.base_model != self.base_model:
            raise ValueError(
                f"base model {spec.base_model!r} does not match deployment "
                f"{self.base_model!r}"
            )
        lora = spec.lora_config
        state = MilesJobState(
            rank=int(lora.rank),
            alpha=float(self.config.default_lora_alpha),
            seed=lora.seed,
            train_attn=bool(lora.train_attn),
            train_mlp=bool(lora.train_mlp),
            train_unembed=bool(lora.train_unembed),
        )
        self._validate_job(state)
        if model_id in self.jobs:
            current = self.jobs[model_id]
            if (
                current.rank,
                current.alpha,
                current.seed,
                current.train_attn,
                current.train_mlp,
                current.train_unembed,
            ) != (
                state.rank,
                state.alpha,
                state.seed,
                state.train_attn,
                state.train_mlp,
                state.train_unembed,
            ):
                raise ValueError(
                    f"model {model_id} already has a different specification"
                )
            return
        if not self.free_slots:
            raise ValueError("no free Miles LoRA slots")

        slot = min(self.free_slots)
        self.free_slots.remove(slot)
        self.jobs[model_id] = state
        self.job_to_slot[model_id] = slot
        self.slot_to_job[slot] = model_id
        try:
            self.runtime.load_slot(slot, state.rank, state.alpha)
        except BaseException:
            self._release(model_id, slot)
            self.jobs.pop(model_id, None)
            raise

    def forward_backward(
        self,
        batch: ForwardBatch,
    ) -> tuple[ForwardBackwardOutput, ...]:
        if not batch.items:
            return ()
        self._require_jobs(tuple(item.model_id for item in batch.items))
        prepared = prepare_batch(batch, self.job_to_slot)
        raw_outputs = self.runtime.forward_backward(
            prepared.slot_rows,
            loss_fn=str(batch.loss_fn),
            loss_fn_config=dict(batch.loss_fn_config),
            forward_only=batch.forward_only,
        )
        if not batch.forward_only:
            for item in batch.items:
                self.jobs[item.model_id].accumulating = True
        return build_outputs(batch, prepared, raw_outputs)

    def optim_step(
        self,
        model_ids: tuple[str, ...],
        adam: AdamParams,
    ) -> tuple[OptimStepResponse, ...]:
        if not model_ids:
            return ()
        if len(set(model_ids)) != len(model_ids):
            raise ValueError("optim_step contains duplicate model ids")
        self._require_jobs(model_ids)
        for model_id in model_ids:
            if not self.jobs[model_id].accumulating:
                raise ValueError(f"model {model_id} has no accumulated gradients")
        parameters = _adam_parameters(adam)
        by_slot = {
            self.job_to_slot[model_id]: parameters.copy() for model_id in model_ids
        }
        grad_norms = self.runtime.optim_step(by_slot)
        outputs = []
        for model_id in model_ids:
            state = self.jobs[model_id]
            state.accumulating = False
            state.optimizer_step += 1
            slot = self.job_to_slot[model_id]
            outputs.append(
                OptimStepResponse(
                    metrics={
                        "grad_norm:mean": float(grad_norms[slot]),
                        "update_successful:mean": 1.0,
                    }
                )
            )
        return tuple(outputs)

    def capture_checkpoint(
        self,
        model_id: str,
        snapshot_id: str,
        *,
        destination: str,
        include_optimizer: bool,
    ) -> None:
        self._require_jobs((model_id,))
        if snapshot_id in self._checkpoint_captures:
            raise ValueError(f"checkpoint snapshot already exists: {snapshot_id}")
        capture_path = self.capture_dir / "checkpoints" / snapshot_id
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        state = self.jobs[model_id]
        self.runtime.save_slot(self.job_to_slot[model_id], str(capture_path))
        if not include_optimizer:
            for path in capture_path.glob("optim_rank*.pt"):
                path.unlink()
        metadata = {
            "schema_version": 1,
            "backend": "miles",
            "miles_revision": MILES_REVISION,
            "base_model": self.base_model,
            "engine_definition_id": os.environ.get("LILO_DEFINITION_ID"),
            "parameterization": {"type": "lora"},
            "lora_config": {
                "rank": state.rank,
                "alpha": state.alpha,
                "seed": state.seed,
                "train_attn": state.train_attn,
                "train_mlp": state.train_mlp,
                "train_unembed": state.train_unembed,
            },
            "optimizer_step": state.optimizer_step,
            "has_optimizer": include_optimizer,
            "topology": {
                "world_size": self.config.world_size,
                "tensor_model_parallel_size": (self.config.tensor_model_parallel_size),
                "expert_model_parallel_size": (self.config.expert_model_parallel_size),
                "expert_tensor_parallel_size": (
                    self.config.expert_tensor_parallel_size
                ),
                "pipeline_model_parallel_size": 1,
            },
        }
        (capture_path / "metadata.json").write_text(
            json.dumps(metadata, sort_keys=True),
            encoding="utf-8",
        )
        self._checkpoint_captures[snapshot_id] = {
            "model_id": model_id,
            "destination": destination,
            "path": capture_path,
        }

    def persist_checkpoint(
        self,
        snapshot_id: str,
        destination: str,
        *,
        overwrite: bool = False,
    ) -> str:
        try:
            capture = self._checkpoint_captures[snapshot_id]
            if capture["destination"] != destination:
                raise ValueError("snapshot destination does not match its capture")
            model_id = capture["model_id"]
            target = self.checkpoint_dir / model_id / "weights" / destination
            _install_directory(capture["path"], target, overwrite=overwrite)
            _commit_volume(os.environ.get("LILO_CHECKPOINT_VOLUME"))
            return str(target)
        finally:
            capture = self._checkpoint_captures.pop(snapshot_id, None)
            if capture is not None:
                shutil.rmtree(capture["path"], ignore_errors=True)

    def load_checkpoint(
        self,
        model_id: str,
        uri: str,
        *,
        restore_optimizer: bool = False,
    ) -> None:
        self._require_jobs((model_id,))
        checkpoint = Path(uri)
        try:
            metadata = json.loads(
                (checkpoint / "metadata.json").read_text(encoding="utf-8")
            )
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid Miles checkpoint: {uri}") from exc
        state = self.jobs[model_id]
        self._validate_checkpoint(metadata, state, restore_optimizer)
        self.runtime.load_slot(
            self.job_to_slot[model_id],
            state.rank,
            state.alpha,
            checkpoint=uri,
            restore_optimizer=restore_optimizer,
        )
        state.accumulating = False
        state.optimizer_step = (
            int(metadata.get("optimizer_step", 0)) if restore_optimizer else 0
        )

    def capture_sampler_snapshot(
        self,
        model_id: str,
        capture_id: str,
        requested_version: int,
    ) -> SamplerPublication:
        self._require_jobs((model_id,))
        if capture_id in self._sampler_captures:
            raise ValueError(f"sampler capture already exists: {capture_id}")
        state = self.jobs[model_id]
        path = self.capture_dir / "sampler" / capture_id
        path.parent.mkdir(parents=True, exist_ok=True)
        self.runtime.export_slot_peft(
            slot=self.job_to_slot[model_id],
            path=str(path),
            rank=state.rank,
            alpha=state.alpha,
            base_model=self.base_model,
            target_modules=self.config.peft_target_modules,
            lora_dropout=self.config.lora_dropout,
        )
        self._sampler_captures[capture_id] = {
            "model_id": model_id,
            "publish_version": requested_version,
            "path": path,
        }
        return SamplerPublication(
            publish_version=requested_version,
            base_model=self.base_model,
            optimizer_step=state.optimizer_step,
        )

    def persist_sampler_snapshot(self, capture_id: str) -> None:
        try:
            capture = self._sampler_captures[capture_id]
            volume_name = os.environ.get("LILO_BULLETIN_VOLUME")
            bulletin = SnapshotBulletin(
                os.environ["LILO_BULLETIN_ROOT"],
                commit=(
                    (lambda: _commit_volume(volume_name))
                    if volume_name is not None
                    else None
                ),
            )
            bulletin.publish(
                VersionRef(
                    capture["model_id"],
                    int(capture["publish_version"]),
                ),
                capture["path"],
            )
        finally:
            capture = self._sampler_captures.pop(capture_id, None)
            if capture is not None:
                shutil.rmtree(capture["path"], ignore_errors=True)

    def unload_model(self, model_id: str) -> None:
        if model_id not in self.jobs:
            return
        slot = self.job_to_slot[model_id]
        self.runtime.unload_slot(slot)
        self._release(model_id, slot)
        self.jobs.pop(model_id, None)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.runtime.close()

    def _validate_job(self, state: MilesJobState) -> None:
        if not 0 < state.rank <= self.config.max_lora_rank:
            raise ValueError(
                f"LoRA rank must be between 1 and {self.config.max_lora_rank}"
            )
        if state.seed is not None:
            raise ValueError("Miles multi-LoRA does not support per-model seeds")
        leaves = {module.rsplit(".", 1)[-1] for module in self.config.target_modules}
        configured = (
            bool(
                leaves
                & {"linear_qkv", "linear_q", "linear_k", "linear_v", "linear_proj"}
            ),
            bool(
                leaves
                & {"linear_fc1", "linear_fc1_gate", "linear_fc1_up", "linear_fc2"}
            ),
            bool(leaves & {"output_layer"}),
        )
        requested = (state.train_attn, state.train_mlp, state.train_unembed)
        if requested != configured:
            raise ValueError(
                "Miles uses deployment-wide LoRA targets; model train_attn, "
                "train_mlp, and train_unembed must match the deployment"
            )

    def _require_jobs(self, model_ids: tuple[str, ...]) -> None:
        for model_id in model_ids:
            if model_id not in self.jobs:
                raise KeyError(f"unknown model: {model_id}")
            if model_id not in self.job_to_slot:
                raise ValueError(f"model {model_id} is not loaded")

    def _release(self, model_id: str, slot: int) -> None:
        self.job_to_slot.pop(model_id, None)
        self.slot_to_job.pop(slot, None)
        self.free_slots.add(slot)

    def _validate_checkpoint(
        self,
        metadata: dict[str, Any],
        state: MilesJobState,
        restore_optimizer: bool,
    ) -> None:
        if metadata.get("schema_version") != 1:
            raise ValueError("unsupported Miles checkpoint schema")
        if metadata.get("backend") != "miles":
            raise ValueError("checkpoint was not created by the Miles backend")
        if metadata.get("miles_revision") != MILES_REVISION:
            raise ValueError("checkpoint Miles revision does not match the deployment")
        if metadata.get("base_model") != self.base_model:
            raise ValueError("checkpoint base model does not match the deployment")
        lora = metadata.get("lora_config") or {}
        if (
            int(lora.get("rank", 0)) != state.rank
            or float(lora.get("alpha", 0)) != state.alpha
        ):
            raise ValueError("checkpoint LoRA configuration does not match the model")
        expected_targets = {
            "train_attn": state.train_attn,
            "train_mlp": state.train_mlp,
            "train_unembed": state.train_unembed,
        }
        if any(lora.get(name) != value for name, value in expected_targets.items()):
            raise ValueError("checkpoint LoRA targets do not match the model")
        expected_topology = {
            "world_size": self.config.world_size,
            "tensor_model_parallel_size": self.config.tensor_model_parallel_size,
            "expert_model_parallel_size": self.config.expert_model_parallel_size,
            "expert_tensor_parallel_size": self.config.expert_tensor_parallel_size,
            "pipeline_model_parallel_size": 1,
        }
        if metadata.get("topology") != expected_topology:
            raise ValueError("checkpoint topology does not match deployment")
        if restore_optimizer:
            if not metadata.get("has_optimizer"):
                raise ValueError("checkpoint does not include optimizer state")


def _adam_parameters(adam: AdamParams) -> dict[str, float]:
    values = {
        "learning_rate": float(adam.learning_rate),
        "beta1": float(adam.beta1),
        "beta2": float(adam.beta2),
        "eps": float(adam.eps),
        "weight_decay": float(adam.weight_decay),
        "grad_clip_norm": float(adam.grad_clip_norm),
    }
    if values["learning_rate"] < 0:
        raise ValueError("learning_rate must be non-negative")
    if not 0 <= values["beta1"] < 1 or not 0 <= values["beta2"] < 1:
        raise ValueError("adam betas must be in [0, 1)")
    if values["eps"] <= 0:
        raise ValueError("adam eps must be positive")
    if values["weight_decay"] < 0 or values["grad_clip_norm"] < 0:
        raise ValueError("weight_decay and grad_clip_norm must be non-negative")
    return values


def _install_directory(source: Path, target: Path, *, overwrite: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    try:
        shutil.copytree(source, temporary)
        if target.exists():
            if not overwrite:
                raise FileExistsError(f"checkpoint already exists: {target}")
            shutil.rmtree(target)
        os.replace(temporary, target)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _commit_volume(name: str | None) -> None:
    if name is None:
        return
    import modal

    modal.Volume.from_name(name).commit()


def build_executor() -> DistributedExecutor:
    config, checkpoint_dir, capture_dir = parse_backend_config(
        json.loads(os.environ["LILO_BACKEND_CONFIG"])
    )
    backend = MilesCommandBackend(
        config,
        checkpoint_dir=checkpoint_dir,
        capture_dir=capture_dir,
        base_model=os.environ["LILO_BASE_MODEL"],
    )
    return DistributedExecutor(backend)
