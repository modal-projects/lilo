from lilo.providers.modal import ray_cluster
from lilo.providers.modal.ray_cluster import ClusterTopology


def test_start_trainer_cluster_runs_asset_hooks_in_order(monkeypatch) -> None:
    head = ClusterTopology(nodes=2, rank=0, head_addr="head", node_ip="head-ip")
    worker = ClusterTopology(nodes=2, rank=1, head_addr="head", node_ip="worker-ip")
    topologies = iter((head, worker))
    events: list[str] = []

    monkeypatch.setattr(ray_cluster, "discover_topology", lambda _: next(topologies))
    monkeypatch.setattr(
        ray_cluster,
        "_start_head",
        lambda topology, *, join_timeout: events.append("start_head"),
    )
    monkeypatch.setattr(
        ray_cluster,
        "_wait_for_head",
        lambda topology, *, timeout: events.append("wait_for_head"),
    )
    monkeypatch.setattr(
        ray_cluster,
        "_start_worker",
        lambda topology: events.append("start_worker"),
    )
    monkeypatch.setattr(
        ray_cluster,
        "_idle_until_head_exits",
        lambda topology: events.append("idle"),
    )

    assert (
        ray_cluster.start_trainer_cluster(
            2,
            before_head=lambda: events.append("before_head"),
            before_worker_join=lambda: events.append("before_worker_join"),
        )
        == "head:6379"
    )
    assert events == ["before_head", "start_head"]

    assert (
        ray_cluster.start_trainer_cluster(
            2,
            before_head=lambda: events.append("before_head"),
            before_worker_join=lambda: events.append("before_worker_join"),
        )
        is None
    )
    assert events == [
        "before_head",
        "start_head",
        "wait_for_head",
        "before_worker_join",
        "start_worker",
        "idle",
    ]
