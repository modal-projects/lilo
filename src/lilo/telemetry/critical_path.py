"""Always-on critical-path accounting for the trainer process.

``CriticalPath`` is an engine ``Observer``: it turns the ``begin``/``span``
callbacks the engine already emits into bounded in-memory phase totals, read
over HTTP by a training client and printed as one JSON line per report when no
client is polling. Nothing here depends on an exporter, so runs without OTLP
still explain where a step went: queue wait, execution, persistence,
admission, and startup.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lilo.engine.api import FutureState
    from lilo.engine.server import Operation

EVENT_NAME = "lilo_critical_path"
MAX_SERIES = 512
MAX_PENDING = 4096
ALL_MODELS = "*"
ACCEPT = "accept"
_PROCESS_START = time.monotonic()


def uptime_s() -> float:
    return time.monotonic() - _PROCESS_START


@dataclass
class Series:
    count: int = 0
    total_s: float = 0.0
    max_s: float = 0.0
    batched: int = 0

    def add(self, seconds: float, batch: int) -> None:
        self.count += 1
        self.total_s += seconds
        self.max_s = max(self.max_s, seconds)
        self.batched += batch

    def as_dict(self) -> dict[str, float]:
        return {
            "count": self.count,
            "total_s": round(self.total_s, 6),
            "mean_s": round(self.total_s / self.count, 6) if self.count else 0.0,
            "max_s": round(self.max_s, 6),
            "mean_batch": round(self.batched / self.count, 4) if self.count else 0.0,
        }


@dataclass
class CriticalPath:
    """Bounded per-model phase totals plus single-valued lifecycle gauges.

    Implements the engine ``Observer`` protocol so the engine needs no timing
    code of its own: ``begin``/``register_model`` stamp submission, ``span``
    credits ``<kind>.queue_wait`` on a command's first span and
    ``<kind>.<phase>`` for the span itself, and ``forget_model`` evicts a
    model's series so a long-lived trainer never fills ``MAX_SERIES``.
    """

    series: dict[tuple[str, str], Series] = field(default_factory=dict)
    gauges: dict[str, float] = field(default_factory=dict)
    pending: dict[tuple[str, str], float] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record(
        self,
        phase: str,
        seconds: float,
        *,
        model_id: str = ALL_MODELS,
        batch: int = 1,
    ) -> None:
        if seconds < 0:
            return
        keys = [(ALL_MODELS, phase)]
        if model_id != ALL_MODELS:
            keys.append((model_id, phase))
        with self.lock:
            for key in keys:
                entry = self.series.get(key)
                if entry is None:
                    if key[0] != ALL_MODELS and len(self.series) >= MAX_SERIES:
                        continue
                    entry = self.series[key] = Series()
                entry.add(seconds, batch)

    # Observer protocol -----------------------------------------------------

    def register_model(self, model_id: str, spec: object) -> None:
        self._submitted((model_id, ACCEPT))

    def forget_model(self, model_id: str) -> None:
        with self.lock:
            for key in [key for key in self.series if key[0] == model_id]:
                del self.series[key]
            for key in [key for key in self.pending if key[0] == model_id]:
                del self.pending[key]

    def begin(self, operation: Operation) -> None:
        self._submitted((operation.model_id, operation.request_id))

    def reuse(self, request_id: str) -> None:
        return

    def finish(self, operation: Operation, state: FutureState) -> None:
        with self.lock:
            self.pending.pop((operation.model_id, operation.request_id), None)

    def set_activity(self, lane: str, operation: str) -> None:
        return

    def state(
        self,
        model: str | tuple[str, ...] | list[str],
        state: str,
        **detail: object,
    ) -> None:
        return

    def span(
        self,
        model: str | tuple[str, ...] | list[str],
        name: str,
        lane: str,
        t0: float,
        t1: float | None = None,
        **attrs: object,
    ) -> None:
        models = (model,) if isinstance(model, str) else tuple(model)
        ended = t1 if t1 is not None else time.time()
        phase, _, kind = name.rpartition(":")
        phase = phase or "execute"
        waits = self._waits(kind, models, attrs)
        with self.lock:
            submitted = [(key[0], self.pending.pop(key, None)) for key in waits]
        for model_id, at in submitted:
            if at is not None:
                self.record(f"{kind}.queue_wait", t0 - at, model_id=model_id)
        batch = attrs.get("n")
        for model_id in models:
            self.record(
                f"{kind}.{phase}",
                ended - t0,
                model_id=model_id,
                batch=batch if isinstance(batch, int) else 1,
            )
        if kind == ACCEPT and attrs.get("ok", True):
            self.gauge("trainer.first_model_ready_s", uptime_s(), once=True)

    def close(self) -> None:
        return

    @staticmethod
    def _waits(
        kind: str,
        models: tuple[str, ...],
        attrs: dict[str, object],
    ) -> list[tuple[str, str]]:
        if kind == ACCEPT:
            return [(m, ACCEPT) for m in models]
        request_ids = attrs.get("request_ids")
        if isinstance(request_ids, list | tuple):
            return [(str(r).rpartition(":")[0], str(r)) for r in request_ids]
        seq_ids = attrs.get("seq_ids")
        if isinstance(seq_ids, list | tuple) and len(models) == 1:
            return [(models[0], f"{models[0]}:{s}") for s in seq_ids]
        return []

    def _submitted(self, key: tuple[str, str]) -> None:
        with self.lock:
            if len(self.pending) >= MAX_PENDING:
                return
            self.pending[key] = time.time()

    def gauge(self, name: str, value: float, *, once: bool = False) -> None:
        with self.lock:
            if once and name in self.gauges:
                return
            self.gauges[name] = round(value, 6)

    def snapshot(
        self,
        *,
        model_id: str | None = None,
        reset: bool = False,
    ) -> dict[str, object]:
        wanted = model_id or ALL_MODELS
        with self.lock:
            phases = {
                phase: entry.as_dict()
                for (model, phase), entry in self.series.items()
                if model == wanted
            }
            gauges = dict(self.gauges)
            if reset:
                for key in [key for key in self.series if key[0] == wanted]:
                    del self.series[key]
        return {
            "model_id": wanted,
            "uptime_s": round(uptime_s(), 6),
            "phases": phases,
            "gauges": gauges,
        }

    def emit(self, **fields: object) -> None:
        print(
            json.dumps({"event": EVENT_NAME, "ts": time.time(), **fields}),
            flush=True,
        )

    def report(self, *, model_id: str | None = None) -> dict[str, object]:
        snapshot = self.snapshot(model_id=model_id)
        self.emit(**snapshot)
        return snapshot


def flatten(snapshot: dict[str, object], prefix: str = "lilo") -> dict[str, float]:
    """Flatten a snapshot into scalar metrics, ready for ``wandb.log``."""
    metrics: dict[str, float] = {}
    uptime = snapshot.get("uptime_s")
    if isinstance(uptime, int | float):
        metrics[f"{prefix}/uptime_s"] = float(uptime)
    phases = snapshot.get("phases")
    if isinstance(phases, dict):
        for phase, values in phases.items():
            if not isinstance(values, dict):
                continue
            for name, value in values.items():
                if isinstance(value, int | float):
                    metrics[f"{prefix}/{phase}.{name}"] = float(value)
    gauges = snapshot.get("gauges")
    if isinstance(gauges, dict):
        for name, value in gauges.items():
            if isinstance(value, int | float):
                metrics[f"{prefix}/{name}"] = float(value)
    return metrics


current = CriticalPath()
