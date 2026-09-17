import gzip
import json
import time
from pathlib import Path

import pytest
from test_miles import FakeMilesRuntime, _backend, _datum, _spec
from tinker import AdamParams

from lilo.backends import ForwardBatch, ForwardItem
from lilo.backends.miles_runtime.profiling import (
    RankProfiler,
    StepPhaseTimer,
    TorchProfileConfig,
)


def test_torch_profile_config_disabled_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("LILO_TORCH_PROFILE_STEP", raising=False)
    monkeypatch.delenv("LILO_TORCH_PROFILE_DIR", raising=False)

    config = TorchProfileConfig.from_env({})

    assert not config.enabled
    assert config.step is None


def test_torch_profile_config_reads_step_and_dir() -> None:
    config = TorchProfileConfig.from_env(
        {"LILO_TORCH_PROFILE_STEP": "2", "LILO_TORCH_PROFILE_DIR": "/traces"}
    )

    assert config.enabled
    assert config.step == 2
    assert config.output_dir == "/traces"


def test_torch_profile_config_rejects_bad_step() -> None:
    with pytest.raises(ValueError):
        TorchProfileConfig.from_env({"LILO_TORCH_PROFILE_STEP": "abc"})
    with pytest.raises(ValueError):
        TorchProfileConfig.from_env({"LILO_TORCH_PROFILE_STEP": "-1"})


def test_torch_profile_config_defaults_dir_to_checkpoint_volume() -> None:
    config = TorchProfileConfig.from_env(
        {
            "LILO_TORCH_PROFILE_STEP": "1",
            "LILO_CHECKPOINT_ROOT": "/mnt/ckpt",
            "LILO_DEFINITION_ID": "def-a",
        }
    )

    assert config.output_dir == "/mnt/ckpt/torch-profile/def-a"


def test_timer_records_phases_and_metrics(capsys) -> None:
    timer = StepPhaseTimer()
    with timer.phase("forward_backward", 0, model_id="model-a"):
        pass
    with timer.phase("optim_step", 0, model_id="model-a"):
        pass

    metrics = timer.metrics_for_step(0)

    assert "timing/forward_backward_s" in metrics
    assert metrics["timing/forward_backward_calls"] == 1.0
    assert "timing/optim_step_s" in metrics
    assert metrics["timing/trainer_step_s"] == (
        metrics["timing/forward_backward_s"] + metrics["timing/optim_step_s"]
    )
    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert all(line["event"] == "lilo_step_timing" for line in lines)
    assert lines[0]["phase"] == "forward_backward"
    assert lines[0]["step"] == 0
    assert lines[0]["model_id"] == "model-a"


def test_timer_records_on_exception(capsys) -> None:
    timer = StepPhaseTimer()
    with pytest.raises(RuntimeError), timer.phase("forward_backward", 3):
        raise RuntimeError("boom")

    assert "timing/forward_backward_s" in timer.metrics_for_step(3)


def test_timer_accumulates_idle_between_ops() -> None:
    timer = StepPhaseTimer()
    with timer.phase("forward_backward", 0):
        pass
    time.sleep(0.01)
    timer.note_request(0)

    metrics = timer.metrics_for_step(0)
    assert metrics["timing/idle_wait_s"] >= 0.01


def test_rank_profiler_writes_cpu_trace(tmp_path) -> None:
    import torch

    profiler = RankProfiler(activities_cpu_only=True)
    profiler.start()
    torch.randn(64, 64) @ torch.randn(64, 64)
    result = profiler.stop(str(tmp_path), "controller")

    trace = Path(result["trace"])
    table = Path(result["table"])
    assert trace.name == "controller.trace.json.gz"
    assert json.loads(gzip.decompress(trace.read_bytes()))
    assert table.name == "controller.key_averages.txt"
    assert table.read_text()


def test_backend_timing_metrics_without_profiling(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("LILO_TORCH_PROFILE_STEP", raising=False)
    runtime = FakeMilesRuntime()
    backend = _backend(tmp_path, runtime)
    backend.accept_model("model-a", _spec())

    backend.forward_backward(
        ForwardBatch(
            items=(ForwardItem("model-a", (_datum([1, 2, 3], 4),)),),
            loss_fn="cross_entropy",
        )
    )
    (result,) = backend.optim_step(("model-a",), AdamParams(learning_rate=2e-4))

    assert ("torch_profile_start",) not in runtime.calls
    assert result.metrics["timing/optimizer_step"] == 0.0
    assert "timing/forward_backward_s" in result.metrics
    assert "timing/optim_step_s" in result.metrics
    assert "timing/trainer_step_s" in result.metrics
    assert result.metrics["grad_norm:mean"] == 0.5


def _step(backend, model_id="model-a"):
    backend.forward_backward(
        ForwardBatch(
            items=(ForwardItem(model_id, (_datum([1, 2, 3], 4),)),),
            loss_fn="cross_entropy",
        )
    )
    (result,) = backend.optim_step((model_id,), AdamParams(learning_rate=2e-4))
    return result


def test_backend_starts_and_stops_torch_profile(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LILO_TORCH_PROFILE_STEP", "1")
    monkeypatch.setenv("LILO_TORCH_PROFILE_DIR", str(tmp_path / "traces"))
    monkeypatch.delenv("LILO_CHECKPOINT_VOLUME", raising=False)

    class FakeControllerProfiler:
        def __init__(self, **_kwargs):
            self.started = False

        def start(self):
            self.started = True

        def stop(self, output_dir, name):
            return {"trace": f"{output_dir}/{name}.trace.json.gz"}

    monkeypatch.setattr("lilo.backends.miles_lora.RankProfiler", FakeControllerProfiler)
    runtime = FakeMilesRuntime()
    backend = _backend(tmp_path, runtime)
    backend.accept_model("model-a", _spec())

    _step(backend)  # step 0: no profiling yet
    assert ("torch_profile_start",) not in runtime.calls

    _step(backend)  # step 1: profiler starts at first forward_backward
    starts = [c for c in runtime.calls if c == ("torch_profile_start",)]
    assert len(starts) == 1
    assert backend._profiling_active

    backend.capture_sampler_snapshot("model-a", "capture-a", 1)

    result = _step(backend)  # step 2: profiler stops before this fb
    stops = [c for c in runtime.calls if c[0] == "torch_profile_stop"]
    assert stops == [("torch_profile_stop", str(tmp_path / "traces"))]
    fb_index = max(i for i, c in enumerate(runtime.calls) if c[0] == "forward_backward")
    assert runtime.calls.index(stops[0]) < fb_index
    assert not backend._profiling_active

    # The capture ran while optimizer_step was 2, so it is reported on the
    # step-2 optim response.
    assert "timing/save_sampler_weights_s" in result.metrics

    _step(backend)  # step 3: profiling stays off
    assert len([c for c in runtime.calls if c == ("torch_profile_start",)]) == 1
    assert len([c for c in runtime.calls if c[0] == "torch_profile_stop"]) == 1


def test_trainer_deployment_env_forwards_profile_vars(monkeypatch) -> None:
    monkeypatch.delenv("LILO_TRAINER_MAX_CONTAINERS", raising=False)
    monkeypatch.delenv("LILO_TORCH_PROFILE_STEP", raising=False)
    monkeypatch.delenv("LILO_TORCH_PROFILE_DIR", raising=False)
    monkeypatch.setenv("LILO_TORCH_PROFILE_STEP", "2")
    monkeypatch.setenv("LILO_TORCH_PROFILE_DIR", "/traces")

    from lilo.providers.modal.deployment import trainer_deployment_env

    env = trainer_deployment_env()
    assert env["LILO_TORCH_PROFILE_STEP"] == "2"
    assert env["LILO_TORCH_PROFILE_DIR"] == "/traces"
    assert "LILO_APP_NAME" not in env
