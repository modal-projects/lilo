"""Always-on critical-path accounting for the trainer process.

Phases are recorded in memory by the engine, read over HTTP by a training
client, and printed as one JSON line per report when no client is polling.
Nothing here depends on an exporter, so runs without OTLP still explain where
a step went: queue wait, execution, persistence, admission, and startup.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field

EVENT_NAME = "lilo_critical_path"
MAX_SERIES = 512
ALL_MODELS = "*"
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
    """Bounded per-model phase totals plus single-valued lifecycle gauges."""

    series: dict[tuple[str, str], Series] = field(default_factory=dict)
    gauges: dict[str, float] = field(default_factory=dict)
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
        with self.lock:
            for key in ((ALL_MODELS, phase), (model_id, phase)):
                entry = self.series.get(key)
                if entry is None:
                    if len(self.series) >= MAX_SERIES:
                        continue
                    entry = self.series[key] = Series()
                entry.add(seconds, batch)

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
