from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch.distributed as dist
from megatron.bridge.peft.multi_lora_layers import load_adapter
from megatron.core import parallel_state
from tinker import AdamParams, ForwardBackwardOutput, OptimStepResponse

from lilo.engine.spmd import DistributedExecutor, initialize_distributed_runtime

from .contract import (
    CommandBackend,
    ForwardBatch,
    ModelSpec,
    SamplerPublication,
)
from .megatron_config import parse_backend_config
from .megatron_runtime.common.checkpoint_io import rank_tag
from .megatron_runtime.common.config import EngineModelConfig
from .megatron_runtime.common.distributed import initialize_megatron
from .megatron_runtime.common.forward_backward import (
    add_packing_metrics,
    build_outputs,
    prepare_microbatches,
    run_megatron_pipeline,
    synchronize_collectors,
)
from .megatron_runtime.lora.adapters import (
    clear_adapter_slot_preserving_rng,
    iter_named_multi_lora_modules,
    reset_adapter_slot_with_seed,
    zero_adapter_grads_for_slots,
)
from .megatron_runtime.lora.checkpoint import (
    extract_adapter_state,
    load_training_checkpoint,
    write_training_checkpoint,
)
from .megatron_runtime.lora.model import create_model_and_optimizer
from .megatron_runtime.lora.optimizer import (
    capture_optimizer_state_for_adapter,
    clear_optimizer_state_for_adapter,
    restore_optimizer_state_for_adapter,
    run_optimizer_step,
)
from .megatron_runtime.lora.peft import (
    capture_adapter_snapshot,
    persist_adapter_snapshot,
)


@dataclass
class LoraJobState:
    rank: int
    alpha: float
    seed: int | None = None
    train_attn: bool = True
    train_mlp: bool = True
    train_unembed: bool = True
    accumulating: bool = False
    optimizer_step: int = 0
    load_optimizer: bool = False


class LoraMegatronBackend(CommandBackend):
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
        self.rank = dist.get_rank()
        self.persistence_group = persistence_group
        initialize_megatron(config)

        self.model, self.optimizers, self.bridge = create_model_and_optimizer(config)

        # multi lora state vars
        self.max_slots = config.max_lora_slots
        self.free_slots = set(range(self.max_slots))

        self.jobs: dict[str, LoraJobState] = {}
        self.job_to_slot: dict[str, int] = {}
        self.slot_to_job: dict[int, str] = {}
        self.checkpoint_dir = checkpoint_dir
        self._checkpoint_captures: dict[str, dict[str, Any]] = {}
        self._sampler_captures: dict[str, Any] = {}

    def accept_model(self, model_id: str, spec: ModelSpec) -> None:
        if spec.parameterization != "lora" or spec.lora_config is None:
            raise ValueError("LoRA backend requires lora parameterization")
        if spec.base_model != self.base_model:
            raise ValueError(
                f"base model {spec.base_model!r} does not match deployment "
                f"{self.base_model!r}"
            )
        lora = spec.lora_config
        alpha = float(self.config.default_lora_alpha)
        if model_id in self.jobs:
            state = self.jobs[model_id]
            actual = (
                state.rank,
                state.alpha,
                state.seed,
                state.train_attn,
                state.train_mlp,
                state.train_unembed,
            )
            expected = (
                int(lora.rank),
                alpha,
                lora.seed,
                bool(lora.train_attn),
                bool(lora.train_mlp),
                bool(lora.train_unembed),
            )
            if actual != expected:
                raise ValueError(
                    f"model {model_id} already has a different specification"
                )
        else:
            self._register_job(
                model_id,
                rank=int(lora.rank),
                alpha=alpha,
                seed=lora.seed,
                train_attn=bool(lora.train_attn),
                train_mlp=bool(lora.train_mlp),
                train_unembed=bool(lora.train_unembed),
            )
        if model_id not in self.job_to_slot:
            self._load_job_to_slot(model_id)

    def forward_backward(
        self,
        batch: ForwardBatch,
    ) -> tuple[ForwardBackwardOutput, ...]:
        if not batch.items:
            return ()
        return self._forward_backward_batch(batch)

    def optim_step(
        self,
        model_ids: tuple[str, ...],
        adam: AdamParams,
    ) -> tuple[OptimStepResponse, ...]:
        return self._optim_step_batch(model_ids, adam)

    def capture_checkpoint(
        self,
        model_id: str,
        snapshot_id: str,
        *,
        destination: str,
        include_optimizer: bool,
    ) -> None:
        payload = self._capture_checkpoint_snapshot(
            model_id,
            destination=destination,
            include_optimizer=include_optimizer,
        )
        if snapshot_id in self._checkpoint_captures:
            raise ValueError(f"checkpoint snapshot already exists: {snapshot_id}")
        state = self.jobs[model_id]
        self._checkpoint_captures[snapshot_id] = {
            "model_id": model_id,
            "destination": destination,
            "payload": payload,
            "metadata": {
                "schema_version": 1,
                "base_model": self.base_model,
                "engine_definition_id": os.environ["LILO_DEFINITION_ID"],
                "parameterization": {"type": "lora"},
                "lora_config": {
                    "rank": state.rank,
                    "seed": state.seed,
                    "train_attn": state.train_attn,
                    "train_mlp": state.train_mlp,
                    "train_unembed": state.train_unembed,
                },
            },
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
            payload = capture["payload"]
            durable_path = str(payload["_path"])
            shard = Path(durable_path) / f"checkpoint_{rank_tag()}.pt"
            if not overwrite and shard.exists():
                raise FileExistsError(f"checkpoint already exists: {durable_path}")
            path = write_training_checkpoint(
                durable_path,
                adapter_state=payload["adapter"],
                optimizer_state=payload["optimizer"],
                optimizer_step=payload["optimizer_step"],
                metadata=capture["metadata"],
                persistence_group=self.persistence_group,
            )
            return path
        finally:
            self._checkpoint_captures.pop(snapshot_id, None)

    def load_checkpoint(
        self,
        model_id: str,
        uri: str,
        *,
        restore_optimizer: bool = False,
    ) -> None:
        if model_id not in self.jobs:
            raise KeyError(f"unknown model: {model_id}")

        checkpoint = load_training_checkpoint(uri)
        adapter = checkpoint.get("adapter_megatron")
        if not isinstance(adapter, dict) or not adapter:
            raise ValueError(f"checkpoint has invalid adapter weights: {uri}")
        if model_id in self.job_to_slot:
            self._offload_job_from_slot(model_id)
        state = self.jobs[model_id]
        state.load_optimizer = restore_optimizer
        self._load_job_to_slot(model_id, checkpoint)

    def capture_sampler_snapshot(
        self,
        model_id: str,
        capture_id: str,
        publish_version: int,
    ) -> SamplerPublication:
        if model_id not in self.jobs:
            raise KeyError(f"unknown job: {model_id}")
        if model_id not in self.job_to_slot:
            raise ValueError(f"job {model_id} is not loaded")
        bulletin_root = os.environ["LILO_BULLETIN_ROOT"]
        state = self.jobs[model_id]
        base_model = os.environ.get(
            "LILO_BASE_MODEL",
            self.config.hf_checkpoint,
        )
        snapshot = capture_adapter_snapshot(
            bridge=self.bridge,
            model=self.model,
            slot=self.job_to_slot[model_id],
            model_id=model_id,
            publish_version=publish_version,
            base_model=base_model,
            adapter_rank=state.rank,
            adapter_alpha=state.alpha,
            lora_dropout=self.config.lora_dropout,
            split_qkv=self.config.split_qkv,
            split_gdn=self.config.split_gdn,
            split_mamba=self.config.split_mamba,
            target_modules=self.config.target_modules,
            train_attn=state.train_attn,
            train_mlp=state.train_mlp,
            train_unembed=state.train_unembed,
            bulletin_root=bulletin_root,
            bulletin_volume=os.environ["LILO_BULLETIN_VOLUME"],
        )
        if capture_id in self._sampler_captures:
            raise ValueError(f"sampler snapshot already exists: {capture_id}")
        self._sampler_captures[capture_id] = snapshot
        return SamplerPublication(
            publish_version=publish_version,
            base_model=base_model,
            optimizer_step=state.optimizer_step,
        )

    def unload_model(self, model_id: str) -> None:
        if model_id not in self.jobs:
            return
        if model_id in self.job_to_slot:
            self._offload_job_from_slot(model_id)
        self.jobs.pop(model_id)

    def close(self) -> None:
        self._shutdown()

    def _register_job(
        self,
        model_id: str,
        *,
        rank: int,
        alpha: float,
        seed: int | None,
        train_attn: bool,
        train_mlp: bool,
        train_unembed: bool,
    ) -> None:
        if model_id in self.jobs:
            raise ValueError(f"Job {model_id} already registered")
        if not 0 < rank <= self.config.max_lora_rank:
            raise ValueError(
                f"LoRA rank must be between 1 and {self.config.max_lora_rank}"
            )
        targets = {target.rsplit(".", 1)[-1] for target in self.config.target_modules}
        requirements = (
            (
                train_attn,
                {"linear_qkv", "linear_proj", "in_proj", "out_proj"},
                "attention",
            ),
            (train_mlp, {"linear_fc1", "linear_fc2"}, "MLP"),
            (train_unembed, {"output_layer"}, "unembedding"),
        )
        for enabled, candidates, label in requirements:
            if enabled and targets.isdisjoint(candidates):
                raise ValueError(f"{label} LoRA is not enabled by target_modules")

        self.jobs[model_id] = LoraJobState(
            rank=rank,
            alpha=alpha,
            seed=seed,
            train_attn=train_attn,
            train_mlp=train_mlp,
            train_unembed=train_unembed,
        )

    # ------------------------------------------------------------
    # slot management (onload/offload loras to/from gpu slots)
    # ------------------------------------------------------------
    def _allocate_slot(self, job_id: str) -> int:
        if not self.free_slots:
            # todo: implement evictions
            raise ValueError("No free slots available")

        slot = min(self.free_slots)
        self.free_slots.remove(slot)
        self.slot_to_job[slot] = job_id
        self.job_to_slot[job_id] = slot
        return slot

    def _release_slot_assignment(self, job_id: str, slot: int) -> None:
        self.job_to_slot.pop(job_id, None)
        self.slot_to_job.pop(slot, None)
        self.free_slots.add(slot)

    def _load_job_to_slot(
        self,
        model_id: str,
        checkpoint: dict[str, Any] | None = None,
    ) -> int:
        if model_id not in self.jobs:
            raise KeyError(f"unknown job: {model_id}")
        if model_id in self.job_to_slot:
            raise ValueError(
                f"job {model_id} is already loaded in slot {self.job_to_slot[model_id]}"
            )

        slot = self._allocate_slot(model_id)
        try:
            self._load_job_into_slot(model_id, slot, checkpoint)
        except Exception:
            self._release_slot_assignment(model_id, slot)
            raise
        return slot

    def _offload_job_from_slot(self, job_id: str) -> None:
        state = self.jobs[job_id]
        slot = self.job_to_slot[job_id]
        clear_adapter_slot_preserving_rng(self.model, slot)
        optimizer = self.optimizers[slot]
        clear_optimizer_state_for_adapter(optimizer, self.model, slot)
        zero_adapter_grads_for_slots(self.model, [slot])
        optimizer.reload_model_params()
        self._release_slot_assignment(job_id, slot)
        state.accumulating = False

    def _forward_backward_batch(
        self,
        batch: ForwardBatch,
    ) -> tuple[ForwardBackwardOutput, ...]:
        for job in batch.items:
            if job.model_id not in self.jobs:
                raise KeyError(f"unknown job: {job.model_id}")
            if job.model_id not in self.job_to_slot:
                raise ValueError(f"job {job.model_id} is not loaded")

        microbatches, packing_metrics = prepare_microbatches(
            batch,
            self.job_to_slot,
            max_slots=self.max_slots,
            config=self.config,
            rank=self.rank,
        )
        fresh_slots = (
            set()
            if batch.forward_only
            else {
                self.job_to_slot[job.model_id]
                for job in batch.items
                if not self.jobs[job.model_id].accumulating
            }
        )
        if fresh_slots:
            zero_adapter_grads_for_slots(self.model, tuple(fresh_slots))
        output_collector, metric_collector = run_megatron_pipeline(
            self.model,
            self.optimizers[0],
            microbatches,
            forward_only=batch.forward_only,
            route_adapters=True,
        )
        output_collector, metric_collector = synchronize_collectors(
            output_collector,
            metric_collector,
        )
        outputs = build_outputs(batch, output_collector, metric_collector)
        add_packing_metrics(outputs, packing_metrics)
        if not batch.forward_only:
            for item in batch.items:
                self.jobs[item.model_id].accumulating = True
        return outputs

    def _optim_step_batch(
        self,
        model_ids: tuple[str, ...],
        adam: AdamParams,
    ) -> tuple[OptimStepResponse, ...]:
        if not model_ids:
            return ()

        if len(set(model_ids)) != len(model_ids):
            raise ValueError("optim_step contains duplicate model ids")

        for model_id in model_ids:
            if model_id not in self.jobs:
                raise KeyError(f"unknown job: {model_id}")
            if model_id not in self.job_to_slot:
                raise ValueError(f"job {model_id} is not loaded")
            if not self.jobs[model_id].accumulating:
                raise ValueError(f"job {model_id} has no accumulated gradients")

        selected = set(model_ids)
        active_slots = [self.job_to_slot[model_id] for model_id in model_ids]
        preserve_grad_slots = [
            self.job_to_slot[model_id]
            for model_id, state in self.jobs.items()
            if model_id not in selected
            and model_id in self.job_to_slot
            and state.accumulating
        ]

        successful, grad_norm = run_optimizer_step(
            self.optimizers,
            self.model,
            active_slots=active_slots,
            preserve_grad_slots=preserve_grad_slots,
            adam=adam,
        )

        for model_id in model_ids:
            state = self.jobs[model_id]
            state.accumulating = False
            if successful:
                state.optimizer_step += 1

        metrics = {
            "grad_norm:mean": grad_norm,
            "update_successful:mean": float(successful),
        }
        return tuple(
            OptimStepResponse(metrics=metrics.copy()) for _model_id in model_ids
        )

    def persist_sampler_snapshot(self, capture_id: str) -> None:
        snapshot = self._sampler_captures[capture_id]
        try:
            persist_adapter_snapshot(snapshot)
        finally:
            self._sampler_captures.pop(capture_id, None)

    def _capture_checkpoint_snapshot(
        self,
        model_id: str,
        *,
        destination: str,
        include_optimizer: bool,
    ) -> dict[str, Any]:
        slot = self.job_to_slot[model_id]
        state = self.jobs[model_id]
        return {
            "adapter": extract_adapter_state(model=self.model, slot=slot),
            "optimizer": (
                capture_optimizer_state_for_adapter(
                    self.optimizers[slot],
                    self.model,
                    slot,
                )
                if include_optimizer
                else None
            ),
            "optimizer_step": state.optimizer_step,
            "_path": str(self.checkpoint_dir / model_id / "weights" / destination),
        }

    def _shutdown(self) -> None:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()

    def _load_job_into_slot(
        self,
        job_id: str,
        slot: int,
        checkpoint: dict[str, Any] | None = None,
    ) -> None:
        job_state = self.jobs[job_id]
        optimizer = self.optimizers[slot]

        def reset_slot() -> None:
            if job_state.seed is None:
                clear_adapter_slot_preserving_rng(self.model, slot)
            else:
                reset_adapter_slot_with_seed(
                    self.model,
                    slot,
                    job_state.seed,
                )
            clear_optimizer_state_for_adapter(optimizer, self.model, slot)
            zero_adapter_grads_for_slots(self.model, [slot])

        reset_slot()
        try:
            optimizer_step = 0
            if checkpoint is not None:
                adapter_state = checkpoint["adapter_megatron"]
                loaded_tensors = load_adapter(
                    self.model,
                    slot,
                    adapter_state,
                )
                if loaded_tensors != len(adapter_state):
                    raise ValueError(
                        f"checkpoint for job {job_id} loaded {loaded_tensors} of "
                        f"{len(adapter_state)} adapter tensors"
                    )

                optimizer_state = (
                    checkpoint.get("optimizer") if job_state.load_optimizer else None
                )
                if optimizer_state is not None:
                    if not isinstance(optimizer_state, dict):
                        raise ValueError(
                            f"checkpoint for job {job_id} has invalid optimizer state"
                        )
                    optimizer_step = int(checkpoint.get("optimizer_step", 0))
                    restore_optimizer_state_for_adapter(
                        optimizer,
                        self.model,
                        slot,
                        optimizer_state,
                        optimizer_step=optimizer_step,
                    )

            for module_name, module in iter_named_multi_lora_modules(self.model):
                enabled = (
                    job_state.train_unembed
                    if "output_layer" in module_name
                    else job_state.train_mlp
                    if "linear_fc" in module_name or ".mlp." in module_name
                    else job_state.train_attn
                )
                module.init_adapter_slot(
                    slot,
                    rank=job_state.rank,
                    alpha=job_state.alpha if enabled else 0.0,
                )
            optimizer.reload_model_params()
            job_state.optimizer_step = optimizer_step
            job_state.accumulating = False
        except Exception:
            reset_slot()
            optimizer.reload_model_params()
            raise


def build_executor() -> DistributedExecutor:
    (
        command_group,
        checkpoint_persistence_group,
        sampler_persistence_group,
    ) = initialize_distributed_runtime()
    config, checkpoint_dir = parse_backend_config(
        json.loads(os.environ["LILO_BACKEND_CONFIG"])
    )
    backend = LoraMegatronBackend(
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
