"""Bring up a multi-node trainer cluster in Lilo.

Rank 0 runs the Lilo engine (HTTP backend, Miles driver, sampler publication)
and owns the Ray head; every other rank only joins the head with its GPUs and
then idles until Modal tears the cluster down with the head container.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from collections.abc import Callable

import modal.experimental

try:
    import ray
except ModuleNotFoundError as exc:
    if exc.name != "ray":
        raise
    ray = None

RAY_PORT = 6379

_GCS_HEALTH_CHECK_ENV = {
    "RAY_health_check_period_ms": "10000",
    "RAY_health_check_timeout_ms": "20000",
    "RAY_health_check_failure_threshold": "60",
}


class ClusterTopology:
    def __init__(self, *, nodes: int, rank: int, head_addr: str, node_ip: str) -> None:
        self.nodes = nodes
        self.rank = rank
        self.head_addr = head_addr
        self.node_ip = node_ip

    @property
    def is_head(self) -> bool:
        return self.rank == 0

    @property
    def ray_address(self) -> str:
        return f"{self.head_addr}:{RAY_PORT}"


def discover_topology(nodes: int) -> ClusterTopology:
    if nodes == 1:
        return ClusterTopology(
            nodes=1, rank=0, head_addr="127.0.0.1", node_ip="127.0.0.1"
        )
    info = modal.experimental.get_cluster_info()
    ips = list(info.container_ipv4_ips or [])
    if len(ips) != nodes:
        raise RuntimeError(
            f"Modal cluster size mismatch: expected {nodes} nodes, got {len(ips)}"
        )
    return ClusterTopology(
        nodes=nodes, rank=info.rank, head_addr=ips[0], node_ip=ips[info.rank]
    )


def start_trainer_cluster(
    nodes: int,
    *,
    join_timeout: float = 900.0,
    before_head: Callable[[], None] | None = None,
    before_worker_join: Callable[[], None] | None = None,
) -> str | None:
    """Bring up Ray across the cluster.

    Returns the Ray address for rank 0, or ``None`` on ranks that only
    contribute GPUs; those ranks block here until the container is stopped.
    """
    topology = discover_topology(nodes)
    _configure_cluster_env(topology)
    if nodes == 1:
        return None

    if topology.is_head:
        if before_head is not None:
            before_head()
        _start_head(topology, join_timeout=join_timeout)
        return topology.ray_address

    _wait_for_head(topology, timeout=join_timeout)
    if before_worker_join is not None:
        before_worker_join()
    _start_worker(topology)
    _idle_until_head_exits(topology)
    return None


def _configure_cluster_env(topology: ClusterTopology) -> None:
    no_proxy = [
        entry for entry in os.environ.get("no_proxy", "").split(",") if entry.strip()
    ]
    for entry in ("127.0.0.1", topology.node_ip, topology.head_addr):
        if entry not in no_proxy:
            no_proxy.append(entry)
    os.environ["no_proxy"] = ",".join(no_proxy)
    os.environ["NO_PROXY"] = os.environ["no_proxy"]
    os.environ.setdefault("MASTER_ADDR", topology.head_addr)
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "eth0")
    os.environ.update(_GCS_HEALTH_CHECK_ENV)


def _start_head(topology: ClusterTopology, *, join_timeout: float) -> None:
    if ray is None:
        raise ImportError("The trainer image must include Ray to start a cluster")
    subprocess.run(
        [
            "ray",
            "start",
            "--head",
            f"--node-ip-address={topology.node_ip}",
            f"--port={RAY_PORT}",
            "--dashboard-host=0.0.0.0",
        ],
        check=True,
        env=dict(os.environ),
    )
    # `ray start` returns before the GCS accepts drivers; poll instead of racing it.
    deadline = time.monotonic() + join_timeout
    while True:
        try:
            ray.init(address="auto", ignore_reinit_error=True)
            break
        except (ConnectionError, RuntimeError) as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Ray head never became reachable: {exc}") from exc
            time.sleep(1.0)
    try:
        while True:
            alive = [node for node in ray.nodes() if node["Alive"]]
            print(f"[lilo-cluster] Ray nodes alive: {len(alive)}/{topology.nodes}")
            if len(alive) >= topology.nodes:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Timed out waiting for {topology.nodes} Ray nodes to join"
                )
            time.sleep(2.0)
    finally:
        # The engine subprocess reconnects as its own driver.
        ray.shutdown()


def _wait_for_head(topology: ClusterTopology, *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            with socket.create_connection((topology.head_addr, RAY_PORT), timeout=5):
                return
        except OSError as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Ray head {topology.ray_address} unreachable: {exc}"
                ) from exc
            time.sleep(2.0)


def _start_worker(topology: ClusterTopology, *, attempts: int = 30) -> None:
    command = [
        "ray",
        "start",
        f"--node-ip-address={topology.node_ip}",
        "--address",
        topology.ray_address,
    ]
    for attempt in range(1, attempts + 1):
        result = subprocess.run(command, env=dict(os.environ), check=False)
        if result.returncode == 0:
            print(f"[lilo-cluster] rank {topology.rank} joined {topology.ray_address}")
            return
        print(
            f"[lilo-cluster] rank {topology.rank} join attempt {attempt} failed; "
            "retrying"
        )
        time.sleep(5.0)
    raise RuntimeError(f"rank {topology.rank} could not join {topology.ray_address}")


def _idle_until_head_exits(
    topology: ClusterTopology, *, poll_seconds: float = 30.0
) -> None:
    """Keep the worker container alive for as long as the head serves Ray."""
    while True:
        time.sleep(poll_seconds)
        try:
            with socket.create_connection((topology.head_addr, RAY_PORT), timeout=10):
                continue
        except OSError:
            print(f"[lilo-cluster] rank {topology.rank} lost the head; exiting")
            return
