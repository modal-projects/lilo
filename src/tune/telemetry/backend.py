"""Bounded backend measurements carried on the private executor HTTP response.

Only the serving rank records measurements. Context propagation through
asyncio.to_thread keeps concurrent execution and persistence requests isolated.
No exporter or distributed collective is required in the model workers.
"""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path

PHASES = frozenset(
    {
        "prepare",
        "forward_backward",
        "forward",
        "collect",
        "outputs",
        "optimizer",
        "checkpoint_write",
        "checkpoint_commit",
    }
)
COUNTS = frozenset(
    {
        "tune.loss_tokens",
        "tune.padded_tokens",
        "tune.packed_microbatch_count",
        "tune.checkpoint_bytes",
    }
)
MAX_PHASES = 32


@dataclass
class Measurements:
    attributes: dict[str, int] = field(default_factory=dict)
    models: dict[str, dict[str, int]] = field(default_factory=dict)
    phases: list[dict] = field(default_factory=list)

    def as_dict(self):
        return {
            "attributes": self.attributes,
            "models": self.models,
            "phases": self.phases,
        }


active: ContextVar[Measurements | None] = ContextVar(
    "tune_backend_measurements", default=None
)
received: ContextVar[dict | None] = ContextVar(
    "tune_received_measurements", default=None
)


@contextmanager
def recording(enabled=True):
    measurements = Measurements() if enabled else None
    token = active.set(measurements)
    try:
        yield measurements
    finally:
        active.reset(token)


def count(name: str, value: int, *, model_id: str | None = None):
    measurements = active.get()
    if (
        measurements is None
        or name not in COUNTS
        or type(value) is not int
        or value < 0
    ):
        return
    target = (
        measurements.attributes
        if model_id is None
        else measurements.models.setdefault(model_id, {})
    )
    target[name] = target.get(name, 0) + value


@contextmanager
def phase(name: str):
    measurements = active.get()
    if (
        measurements is None
        or name not in PHASES
        or len(measurements.phases) >= MAX_PHASES
    ):
        yield
        return
    entry = {"name": name, "start_ns": time.time_ns(), "ok": False}
    measurements.phases.append(entry)
    try:
        yield
        entry["ok"] = True
    finally:
        entry["end_ns"] = time.time_ns()


def measured(name: str):
    def decorate(function):
        @functools.wraps(function)
        def call(*args, **kwargs):
            with phase(name):
                return function(*args, **kwargs)

        return call

    return decorate


def checkpoint_size(directory: str):
    """Logical file bytes after all shards have been written; no file contents read."""
    if active.get() is None:
        return
    try:
        size = sum(
            p.stat().st_size
            for p in Path(directory).rglob("*")
            if p.is_file() and not p.is_symlink()
        )
    except OSError:
        return
    count("tune.checkpoint_bytes", size)


def numeric_attributes(value):
    if not isinstance(value, dict):
        return {}
    return {k: v for k, v in value.items() if k in COUNTS and type(v) is int and v >= 0}
