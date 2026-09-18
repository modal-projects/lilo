"""Accounting checks: common wall time, token conservation and run isolation."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "sweep_analysis", Path(__file__).parents[1] / "scripts/analyze_dapo_client_sweep.py"
)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def report():
    clients = []
    for client, end in [(0, 160), (1, 220)]:
        rows = [
            dict(
                step=step,
                step_seconds=(end - 100) / 6,
                completed_at=100 + (end - 100) * (step - 2) / 6,
                rollout_start=100,
                policy_lag=1,
                optimizer={"update_successful:mean": 1},
                training={"tokens:sum": 6400},
                prompt_tokens=6464 - 3200,
                output_tokens=3200,
                train_seconds=5,
                publish_seconds=1,
                rollout_wait_seconds=2,
                truncated=0,
            )
            for step in range(1, 9)
        ]
        clients.append(
            dict(index=client, status="completed", steps=rows, completed_at=end)
        )
    return dict(
        run="test",
        status="completed",
        config={"clients": 2},
        measurement_start=100,
        measurement_end=220,
        clients=clients,
    )


def test_uses_common_wall_time_not_sum_of_overlapping_times():
    result = analysis.summarize(report())
    assert result["output_tps"] == 38400 / 120
    assert result["output_tps_per_client"] == 160
    assert result["training_tps"] == 76800 / 120
    assert result["cost"]["lilo_total_gpu"] == 16 * 120 * 0.001261
    assert result["mean_step_seconds"] == 15


def test_rejects_prefetch_outside_measurement_window():
    r = report()
    r["clients"][0]["steps"][2]["rollout_start"] = 99
    with pytest.raises(ValueError, match="crosses"):
        analysis.summarize(r)


def test_rejects_missing_or_inconsistent_training_tokens():
    r = report()
    r["clients"][0]["steps"][2]["training"]["tokens:sum"] += 1
    with pytest.raises(ValueError, match="reconcile"):
        analysis.summarize(r)


def test_requires_completed_cleanup_and_unique_deployments():
    state = dict(
        drained=True,
        remaining_tasks=[],
        cleanup_errors=[],
        app_id="a",
        pool_app="p",
        finished_at=200,
    )
    analysis.validate_isolation([(state, dict(started_at=100))])
    with pytest.raises(ValueError, match="reused"):
        analysis.validate_isolation(
            [(state, dict(started_at=100)), (state, dict(started_at=300))]
        )
    with pytest.raises(ValueError, match="drain"):
        analysis.validate_isolation(
            [({**state, "drained": False}, dict(started_at=100))]
        )


def test_rejects_partial_gpu_allocation(tmp_path):
    import json

    task = dict(gpu_type="H200", started_at=1, finished_at=0)
    trainer = dict(task, app_id="a", app_name="trainer", gpu_count=8)
    replica = dict(task, app_id="p", app_name="pool", gpu_count=1)
    path = tmp_path / "resources.jsonl"
    state = dict(app_id="a", pool_app="pool")
    r = dict(measurement_start=100, measurement_end=130)

    def write(n):
        path.write_text(
            "\n".join(
                json.dumps(dict(time=t, tasks=[trainer] + [replica] * n))
                for t in [95, 105, 115, 125, 135]
            )
        )

    lifetimes = [dict(t, finished_at=140) for t in [trainer] + [replica] * 8]
    write(8)
    assert analysis.validate_resources(path, r, state, lifetimes)["inference_gpus"] == 8
    write(7)
    with pytest.raises(ValueError, match="eight inference"):
        analysis.validate_resources(path, r, state, lifetimes)


def test_rejects_gpu_stopping_before_last_update(tmp_path):
    import json

    trainer = dict(
        app_id="a",
        app_name="trainer",
        gpu_count=8,
        gpu_type="H200",
        started_at=1,
        finished_at=0,
    )
    replica = dict(
        app_id="p",
        app_name="pool",
        gpu_count=1,
        gpu_type="H200",
        started_at=1,
        finished_at=0,
    )
    tasks = [trainer] + [replica] * 8
    path = tmp_path / "resources.jsonl"
    path.write_text(
        "\n".join(json.dumps(dict(time=t, tasks=tasks)) for t in [95, 105, 115, 125])
    )
    # Polling alone cannot prove GPUs survived the final five seconds.
    lifetimes = [dict(t, finished_at=140) for t in tasks]
    lifetimes[0]["finished_at"] = 129
    with pytest.raises(ValueError, match="lifetimes"):
        analysis.validate_resources(
            path,
            dict(measurement_start=100, measurement_end=130),
            dict(app_id="a", pool_app="pool"),
            lifetimes,
        )


def test_controller_comparison_allows_only_preemption_placement_change():
    import hashlib

    old = "@app.function(cpu=16,\n)\ndef controller():\n    train(steps=8)\n"
    new = old.replace("\n)", "\n    nonpreemptible=True,\n)")
    def digest(text):
        return hashlib.sha256(text.encode()).hexdigest()

    for source in [old, new]:
        assert analysis.controller_workload_hash(digest(source), new) == digest(old)
    for changed in [new.replace("steps=8", "steps=4"), new.replace("cpu=16", "cpu=32")]:
        with pytest.raises(ValueError, match="preemption setting"):
            analysis.controller_workload_hash(digest(changed), new)
