"""Probe how quickly a strict GPU type schedules for a 2-node x 8-GPU clustered shape.

Modal's strict forms: ``H100!`` (no auto-upgrade to H200); ``H200`` is already exact
(there is no ``H200!``; the server rejects it).

    MODAL_ENVIRONMENT=micah-dev uv run --with modal modal run \
        scripts/probe_gpu_scheduling.py --gpu H200
"""

import subprocess
import time

import modal
from modal.experimental import clustered

app = modal.App("probe-gpu-scheduling")
image = modal.Image.debian_slim()
NODES = 2


def _nvidia_smi() -> str:
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        text=True,
    )


@app.function(gpu="H200:8", timeout=900, image=image)
@clustered(NODES, rdma=True)
def probe_h200() -> str:
    return _nvidia_smi()


@app.function(gpu="H100!:8", timeout=900, image=image)
@clustered(NODES, rdma=True)
def probe_h100() -> str:
    return _nvidia_smi()


@app.local_entrypoint()
def main(gpu: str = "H200"):
    fn = {"H200": probe_h200, "H100!": probe_h100}[gpu]
    t0 = time.time()
    out = fn.remote()
    print(f"gpu={gpu} nodes={NODES} landed_in={time.time() - t0:.1f}s")
    print(out)
