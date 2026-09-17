from __future__ import annotations

import contextlib
import gzip
import json
import os
import tempfile
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass

PROFILE_STEP_ENV = "LILO_TORCH_PROFILE_STEP"
PROFILE_DIR_ENV = "LILO_TORCH_PROFILE_DIR"

_MAX_TRACKED_STEPS = 8


@dataclass(frozen=True, slots=True)
class TorchProfileConfig:
    """Opt-in torch.profiler capture for one optimizer step."""

    step: int | None
    output_dir: str

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> TorchProfileConfig:
        raw_step = env.get(PROFILE_STEP_ENV)
        step: int | None = None
        if raw_step:
            step = int(raw_step)
            if step < 0:
                raise ValueError(f"{PROFILE_STEP_ENV} must be non-negative")
        checkpoint_root = env.get("LILO_CHECKPOINT_ROOT") or "/checkpoints"
        output_dir = env.get(PROFILE_DIR_ENV) or os.path.join(
            checkpoint_root,
            "torch-profile",
            env.get("LILO_DEFINITION_ID", "trainer"),
        )
        return cls(step=step, output_dir=output_dir)

    @property
    def enabled(self) -> bool:
        return self.step is not None


class StepPhaseTimer:
    """Always-on, cheap per-phase wall-clock timer for backend operations.

    Thread-safe: backend methods may run on different threads via
    ``asyncio.to_thread``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._steps: dict[int, dict[str, float]] = {}
        self._last_activity_end: float | None = None

    def record(
        self,
        phase: str,
        seconds: float,
        *,
        step: int,
        model_id: str | None = None,
        **extra: object,
    ) -> None:
        with self._lock:
            accumulators = self._steps.setdefault(step, {})
            self._prune_locked()
            accumulators[phase] = accumulators.get(phase, 0.0) + seconds
            calls_key = f"{phase}_calls"
            accumulators[calls_key] = accumulators.get(calls_key, 0.0) + 1
            self._last_activity_end = time.perf_counter()
        print(
            json.dumps(
                {
                    "event": "lilo_step_timing",
                    "phase": phase,
                    "step": step,
                    "seconds": round(seconds, 6),
                    "model_id": model_id,
                    "ts": time.time(),
                    **extra,
                }
            ),
            flush=True,
        )

    def note_request(self, step: int) -> None:
        """Account for idle time since the previous backend op ended."""
        with self._lock:
            now = time.perf_counter()
            if self._last_activity_end is not None:
                gap = now - self._last_activity_end
                if gap > 0:
                    accumulators = self._steps.setdefault(step, {})
                    self._prune_locked()
                    accumulators["idle_wait"] = accumulators.get("idle_wait", 0.0) + gap
                    accumulators["idle_wait_calls"] = (
                        accumulators.get("idle_wait_calls", 0.0) + 1
                    )

    @contextlib.contextmanager
    def phase(
        self,
        name: str,
        step: int,
        model_id: str | None = None,
        **extra: object,
    ):
        self.note_request(step)
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(
                name, time.perf_counter() - start, step=step, model_id=model_id, **extra
            )

    def metrics_for_step(self, step: int) -> dict[str, float]:
        """Return ``timing/*`` metrics for one optimizer step.

        Save/publish work for step k runs *after* ``optim_step`` k returns, so
        it is attributed to the model's current optimizer step at the time it
        runs (k + 1). Values logged under step k + 1 therefore refer to the
        checkpoint taken after step k.
        """
        with self._lock:
            accumulators = dict(self._steps.get(step, {}))
        metrics: dict[str, float] = {}
        for key, value in accumulators.items():
            if key.endswith("_calls"):
                metrics[f"timing/{key}"] = float(value)
            else:
                metrics[f"timing/{key}_s"] = float(value)
        forward_backward = accumulators.get("forward_backward")
        optim_step = accumulators.get("optim_step")
        if forward_backward is not None and optim_step is not None:
            metrics["timing/trainer_step_s"] = forward_backward + optim_step
        return metrics

    def pop_step(self, step: int) -> None:
        with self._lock:
            self._steps.pop(step, None)

    def _prune_locked(self) -> None:
        while len(self._steps) > _MAX_TRACKED_STEPS:
            self._steps.pop(min(self._steps))


class RankProfiler:
    """Rank-local torch.profiler wrapper usable in any process.

    ``torch`` is imported lazily so importing this module never requires CUDA.
    """

    def __init__(self, *, activities_cpu_only: bool = False) -> None:
        self._cpu_only = activities_cpu_only
        self._profile = None
        self._torch = None

    def start(self) -> None:
        import torch

        activities = [torch.profiler.ProfilerActivity.CPU]
        if not self._cpu_only:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self._torch = torch
        self._profile = torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=not self._cpu_only,
            with_stack=False,
        )
        self._profile.__enter__()

    def stop(self, output_dir: str, name: str) -> dict[str, str]:
        if self._profile is None:
            raise RuntimeError("RankProfiler.stop() called without start()")
        torch = self._torch
        if not self._cpu_only and torch.cuda.is_available():
            torch.cuda.synchronize()
        profile = self._profile
        self._profile = None
        profile.__exit__(None, None, None)
        os.makedirs(output_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", dir=output_dir, delete=False
        ) as temporary:
            temporary_path = temporary.name
        try:
            profile.export_chrome_trace(temporary_path)
            trace_path = os.path.join(output_dir, f"{name}.trace.json.gz")
            with (
                open(temporary_path, "rb") as source,
                gzip.open(trace_path, "wb") as target,
            ):
                target.write(source.read())
        finally:
            os.unlink(temporary_path)
        table_path = os.path.join(output_dir, f"{name}.key_averages.txt")
        sort_key = "self_cpu_time_total" if self._cpu_only else "cuda_time_total"
        with open(table_path, "w", encoding="utf-8") as table:
            table.write(profile.key_averages().table(sort_by=sort_key, row_limit=60))
        return {"trace": trace_path, "table": table_path}
