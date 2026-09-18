from __future__ import annotations

import asyncio
import os
import shlex
import sys
import threading
from collections.abc import Coroutine
from contextlib import contextmanager, suppress
from functools import wraps
from itertools import count
from pathlib import Path
from typing import Any

from lilo.backends.miles_config import MilesBackendConfig
from lilo.errors import BackendFailed


class MilesRuntime:
    """Synchronous owner of Miles's asynchronous Ray trainer controller."""

    def __init__(self, config: MilesBackendConfig) -> None:
        self.config = config
        # Baked into the image after resolving main, never the moving ref name.
        self.revision = os.environ["LILO_MILES_COMMIT"]
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="lilo-miles-runtime",
            daemon=True,
        )
        self._thread.start()
        self._closed = False
        self._failure: BaseException | None = None
        self._trainer = None
        self._worker_manager = None
        self._bridge = None
        self._owns_ray = False
        self._owns_object_store = False
        self._unit_ids = count(1)
        try:
            self._call(self._start())
        except BaseException:
            with suppress(BaseException):
                self._call(self._close())
            self._stop_loop()
            raise

    def load_slot(
        self,
        slot: int,
        rank: int,
        alpha: float,
        *,
        checkpoint: str | None = None,
        restore_optimizer: bool = True,
    ) -> None:
        self._run(
            self._bridge.load_slot(
                slot,
                rank,
                alpha,
                ckpt_path=checkpoint,
                load_optimizer=restore_optimizer,
            )
        )

    def unload_slot(self, slot: int) -> None:
        self._run(self._bridge.unload_slot(slot))

    def forward_backward(
        self,
        slot_rows: tuple[tuple[int, dict[str, Any]], ...],
        *,
        loss_fn: str,
        loss_fn_config: dict[str, float],
        forward_only: bool,
    ) -> list[dict[str, Any]]:
        unit_id = next(self._unit_ids)
        method = (
            self._bridge.forward_only if forward_only else self._bridge.forward_backward
        )
        return self._run(
            method(
                unit_id,
                list(slot_rows),
                loss_fn,
                loss_fn_config,
            )
        )

    def optim_step(
        self, adam_params_by_slot: dict[int, dict[str, float]]
    ) -> dict[int, dict[str, float]]:
        return self._run(self._bridge.optim_step(adam_params_by_slot))

    def save_slot(
        self, slot: int, path: str, *, include_optimizer: bool = True
    ) -> None:
        if include_optimizer:
            self._run(self._bridge.save_slot(slot, path))
        else:
            self._run(
                self._trainer._execute_slots("save_slot_weights", slot=slot, path=path)
            )
        _materialize_capture(path)

    def export_slot_peft(
        self,
        *,
        slot: int,
        path: str,
        rank: int,
        alpha: float,
        base_model: str,
        target_modules: tuple[str, ...],
        lora_dropout: float,
    ) -> None:
        self._run(self._bridge.export_slot(slot, rank, alpha, path))
        _materialize_capture(path)

    def torch_profile_start(self) -> None:
        self._run_best_effort(self._trainer._execute_slots("torch_profile_start"))

    def torch_profile_stop(self, output_dir: str) -> list[dict | None]:
        return self._run_best_effort(
            self._trainer._execute_slots("torch_profile_stop", output_dir=output_dir)
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._call(self._close())
        finally:
            self._stop_loop()

    def _run(self, coroutine: Coroutine[Any, Any, Any]) -> Any:
        if self._closed:
            coroutine.close()
            raise BackendFailed("Miles runtime is closed")
        if self._failure is not None:
            coroutine.close()
            raise BackendFailed("Miles trainer is unavailable") from self._failure
        try:
            result = self._call(coroutine)
            _check_result(result)
            return result
        except Exception as exc:
            self._failure = exc
            raise BackendFailed(f"Miles trainer failed: {exc}") from exc

    def _run_best_effort(self, coroutine: Coroutine[Any, Any, Any]) -> Any:
        """Like ``_run`` but an exception does not mark the trainer as failed."""
        if self._closed or self._failure is not None:
            coroutine.close()
            raise BackendFailed("Miles trainer is unavailable")
        result = self._call(coroutine)
        _check_result(result)
        return result

    def _call(self, coroutine: Coroutine[Any, Any, Any]) -> Any:
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        return future.result()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _stop_loop(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=30)
        if self._thread.is_alive():
            raise RuntimeError("Miles runtime event loop did not stop")
        self._loop.close()

    async def _start(self) -> None:
        os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
        os.environ.setdefault("no_proxy", "127.0.0.1")

        import ray
        from miles.ray.specs import train as train_specs
        from miles.ray.train.group import TrainerController
        from miles.ray.wiring import launch_worker_manager
        from miles.tinker.runtime import MilesBackend
        from miles.utils import object_store
        from miles.utils.arguments import parse_args
        from miles.utils.audit_utils.process_identity import MainProcessIdentity
        from miles.utils.external_utils.model_args_utils import load_model_args
        from miles.utils.logging_utils import configure_logger

        _configure_actor_spec(train_specs)
        _allow_context_parallel_multi_lora()
        architecture = shlex.split(load_model_args(self.config.model_type))
        with _temporary_argv([*architecture, *self.config.miles_arguments()]):
            args = parse_args(entry="serve")
        args.use_dynamic_global_batch_size = True
        args.delay_split_train_data_by_dp = True
        configure_logger(args, source=MainProcessIdentity())

        if ray.is_initialized():
            raise RuntimeError("MilesRuntime requires exclusive ownership of Ray")
        ray.init(
            ignore_reinit_error=False,
            include_dashboard=False,
            log_to_driver=True,
            num_cpus=max(4, self.config.world_size * 2),
            num_gpus=self.config.world_size,
        )
        self._owns_ray = True

        self._worker_manager = launch_worker_manager(args)
        object_store.init_instance(args, contribute_segment=False)
        self._owns_object_store = True
        self._trainer = TrainerController(
            args=args,
            role="actor",
            with_ref=False,
            with_opd_teacher=False,
            inference_controller=None,
            rollout_executor=None,
        )
        await self._trainer.init()
        self._bridge = MilesBackend(self._trainer, router_url="", dp_size=1)

    async def _close(self) -> None:
        import ray
        from miles.utils import object_store

        try:
            if self._trainer is not None:
                cell_ids = list(self._trainer.cell_ids)
                try:
                    await self._trainer.dispose()
                finally:
                    if self._worker_manager is not None and cell_ids:
                        with suppress(BaseException):
                            await self._worker_manager.stop_cells.remote(cell_ids)
        finally:
            if self._worker_manager is not None:
                with suppress(BaseException):
                    ray.kill(self._worker_manager, no_restart=True)
            if self._owns_object_store:
                object_store._INSTANCE = None
                self._owns_object_store = False
            if self._owns_ray and ray.is_initialized():
                ray.shutdown()
            self._owns_ray = False
            self._bridge = None
            self._trainer = None
            self._worker_manager = None


@contextmanager
def _temporary_argv(arguments: list[str]):
    original = sys.argv
    sys.argv = ["lilo-miles-backend", *arguments]
    try:
        yield
    finally:
        sys.argv = original


def _check_result(result) -> None:
    if isinstance(result, (list, tuple)):
        for item in result:
            _check_result(item)
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(f"Miles operation failed: {result['error']}")


def _configure_actor_spec(train_specs) -> None:
    original = train_specs._compute_spec_trainer
    if getattr(original, "_lilo_actor_spec", False):
        return

    @wraps(original)
    def compute(*args, **kwargs):
        spec = original(*args, **kwargs)
        if (
            spec.worker_class
            == "miles.backends.megatron_utils.lora.actor.MultiLoRATrainRayActor"
        ):
            return spec.model_copy(
                update={
                    "worker_class": "lilo.backends.miles_runtime.actor.LiloMilesTrainRayActor"
                }
            )
        return spec

    compute._lilo_actor_spec = True
    train_specs._compute_spec_trainer = compute


def _allow_context_parallel_multi_lora() -> None:
    """Miles rejects multi-LoRA with CP>1 because its Tinker losses zip
    full-length per-datum vectors against CP-sharded log probs. The Lilo actor
    gathers those log probs back to full response length before the loss runs,
    so the guard does not apply to this path."""
    from miles.utils import multi_lora

    original = multi_lora.validate_multi_lora_args
    if getattr(original, "__lilo_allows_cp__", False):
        return

    @wraps(original)
    def validate(args) -> None:
        context_parallel_size = getattr(args, "context_parallel_size", 1)
        args.context_parallel_size = 1
        try:
            original(args)
        finally:
            args.context_parallel_size = context_parallel_size

    validate.__lilo_allows_cp__ = True
    multi_lora.validate_multi_lora_args = validate


def _materialize_capture(path: str) -> None:
    """Detach Miles's completed version directory before Lilo copies/cleans it."""
    capture = Path(path)
    if not capture.is_symlink():
        return
    version = capture.resolve(strict=True)
    if version.parent != capture.parent.resolve() or not version.name.startswith(
        f"_version_{capture.name}_"
    ):
        raise RuntimeError(f"unexpected Miles capture target: {version}")
    capture.unlink()
    version.rename(capture)
