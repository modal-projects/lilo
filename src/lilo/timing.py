"""Read the trainer's critical-path breakdown from a training loop.

``critical_path_metrics`` returns scalars ready for ``wandb.log``; ``log_critical_path``
logs them to an active W&B run when one exists and otherwise prints one JSON
line, so a run without W&B still records where each step went. By default the
fetch and log happen on a background thread so the round-trip to the control
plane never stalls a training step.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import weakref

import httpx

from lilo.telemetry.critical_path import EVENT_NAME, flatten

TIMING_PATH = "/api/v1/timing"
STEP_METRIC = "step"

logger = logging.getLogger(__name__)


def critical_path_metrics(
    model_id: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    reset: bool = True,
    prefix: str = "lilo",
    timeout: float = 10.0,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, float]:
    url = (base_url or os.environ["TINKER_BASE_URL"]).rstrip("/") + TIMING_PATH
    key = api_key if api_key is not None else os.environ.get("TINKER_API_KEY")
    with httpx.Client(transport=transport, timeout=timeout) as client:
        response = client.get(
            url,
            params={"model_id": model_id, "reset": "true" if reset else "false"},
            headers={"X-API-Key": key} if key else {},
        )
    response.raise_for_status()
    return flatten(response.json(), prefix)


def log_critical_path(
    model_id: str,
    *,
    step: int | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    reset: bool = True,
    prefix: str = "lilo",
    transport: httpx.BaseTransport | None = None,
    background: bool = True,
) -> dict[str, float] | None:
    """Collect the breakdown since the last call; never raise into a training step.

    With ``background=True`` (the default) the HTTP fetch and the log run on a
    worker thread and this returns ``None`` immediately. A call made while the
    previous one is still in flight is dropped rather than queued, so a slow
    control plane costs at most one outstanding request. With
    ``background=False`` the call blocks and returns the logged metrics (``{}``
    when the trainer was unreachable).
    """

    run = _active_wandb_run()

    def work() -> dict[str, float]:
        metrics: dict[str, float] = {}
        try:
            metrics = critical_path_metrics(
                model_id,
                base_url=base_url,
                api_key=api_key,
                reset=reset,
                prefix=prefix,
                transport=transport,
            )
            _emit(run, metrics, model_id=model_id, step=step, prefix=prefix)
        except Exception:
            logger.warning("critical path report failed", exc_info=True)
        return metrics

    if not background:
        return work()
    if not _reporter.submit(work):
        logger.debug("critical path report for step %s skipped: fetch in flight", step)
    return None


def flush_critical_path(timeout: float = 10.0) -> bool:
    """Wait for an in-flight background report; call before ``wandb.finish()``."""
    return _reporter.join(timeout)


def _emit(
    run,
    metrics: dict[str, float],
    *,
    model_id: str,
    step: int | None,
    prefix: str,
) -> None:
    if run is not None:
        _log_wandb(run, metrics, step=step, prefix=prefix)
        return
    print(
        json.dumps(
            {
                "event": EVENT_NAME,
                "ts": time.time(),
                "model_id": model_id,
                "step": step,
                "metrics": metrics,
            }
        ),
        flush=True,
    )


def _log_wandb(
    run, metrics: dict[str, float], *, step: int | None, prefix: str
) -> None:
    """Log against a ``<prefix>/step`` x-axis instead of W&B's global step.

    A background report can land after the training loop has already committed
    a later step, and W&B drops rows whose explicit ``step`` is behind the run.
    ``commit=False`` attaches the values to the loop's next commit without
    advancing the step counter; ``define_metric`` makes charts plot them
    against ``<prefix>/step`` so they line up with the step they describe.
    """
    if step is None:
        run.log(metrics, commit=False)
        return
    step_metric = f"{prefix}/{STEP_METRIC}"
    _reporter.define_step_metric(run, f"{prefix}/*", step_metric)
    run.log({**metrics, step_metric: step}, commit=False)


class _Reporter:
    """One in-flight background report at a time."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.defined: weakref.WeakKeyDictionary[object, set[str]] = (
            weakref.WeakKeyDictionary()
        )

    def submit(self, work) -> bool:
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                return False
            self.thread = threading.Thread(
                target=work, name="lilo-critical-path", daemon=True
            )
            self.thread.start()
        return True

    def join(self, timeout: float) -> bool:
        with self.lock:
            thread = self.thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def define_step_metric(self, run, pattern: str, step_metric: str) -> None:
        with self.lock:
            patterns = self.defined.setdefault(run, set())
            if pattern in patterns:
                return
            patterns.add(pattern)
        run.define_metric(step_metric, hidden=True)
        run.define_metric(pattern, step_metric=step_metric)


_reporter = _Reporter()


def _active_wandb_run():
    wandb = sys.modules.get("wandb")
    return None if wandb is None else wandb.run
