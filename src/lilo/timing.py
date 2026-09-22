"""Read the trainer's critical-path breakdown from a training loop.

``critical_path_metrics`` returns scalars ready for ``wandb.log``; ``log_critical_path``
logs them to an active W&B run when one exists and otherwise prints one JSON
line, so a run without W&B still records where each step went.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

import httpx

from lilo.telemetry.critical_path import EVENT_NAME, flatten

TIMING_PATH = "/api/v1/timing"


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
) -> dict[str, float]:
    """Collect the breakdown since the last call; never raise into a training step."""
    try:
        metrics = critical_path_metrics(
            model_id,
            base_url=base_url,
            api_key=api_key,
            reset=reset,
            prefix=prefix,
            transport=transport,
        )
    except Exception:
        logging.getLogger(__name__).warning("critical path unavailable", exc_info=True)
        return {}
    run = _active_wandb_run()
    if run is not None:
        run.log(metrics, step=step)
        return metrics
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
    return metrics


def _active_wandb_run():
    wandb = sys.modules.get("wandb")
    return None if wandb is None else wandb.run
