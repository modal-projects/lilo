import asyncio

import pytest

from lilo.telemetry import backend


def test_measurements_are_isolated_across_tasks_and_propagate_to_threads():
    async def task(model, tokens):
        with backend.recording() as measurements:

            def work():
                with backend.phase("prepare"):
                    backend.count("lilo.loss_tokens", tokens, model_id=model)

            await asyncio.to_thread(work)
            await asyncio.sleep(0)
            return measurements.as_dict()

    async def run():
        return await asyncio.gather(task("a", 2), task("b", 3))

    first, second = asyncio.run(run())
    assert first["models"] == {"a": {"lilo.loss_tokens": 2}}
    assert second["models"] == {"b": {"lilo.loss_tokens": 3}}
    for measurement in (first, second):
        (phase,) = measurement["phases"]
        assert phase["start_ns"] <= phase["end_ns"]
        assert phase["ok"] is True
    assert backend.active.get() is None


def test_disabled_and_bounded_recording_and_exception_preservation():
    with backend.recording(False) as disabled, backend.phase("prepare"):
        backend.count("lilo.loss_tokens", 2)
    assert disabled is None
    with backend.recording() as measurements:
        with pytest.raises(ValueError, match="original"), backend.phase("prepare"):
            raise ValueError("original")
        for _ in range(100):
            with backend.phase("outputs"):
                pass
        backend.count("secret", 12)
        backend.count("lilo.loss_tokens", -1)
    assert len(measurements.phases) == backend.MAX_PHASES
    assert measurements.phases[0]["ok"] is False
    assert measurements.attributes == {}
    assert "original" not in str(measurements.as_dict())


def test_checkpoint_size_counts_files_without_reading_contents(tmp_path, monkeypatch):
    (tmp_path / "rank0.pt").write_bytes(b"abc")
    (tmp_path / "rank1.pt").write_bytes(b"12345")
    (tmp_path / "metadata.json").write_bytes(b"{}")
    (tmp_path / "alias").symlink_to(tmp_path / "rank0.pt")
    with backend.recording() as measurements:
        backend.checkpoint_size(str(tmp_path))
    assert measurements.attributes == {"lilo.checkpoint_bytes": 10}

    def failed_stat(*args, **kwargs):
        raise OSError("unavailable")

    monkeypatch.setattr(backend.Path, "stat", failed_stat)
    with backend.recording() as measurements:
        backend.checkpoint_size(str(tmp_path))
    assert measurements.attributes == {}
