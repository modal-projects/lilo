import asyncio

from lilo.engine import Engine, FutureStatus
from lilo.telemetry.critical_path import CriticalPath
from tests.engine.test_server import forward_backward
from tests.support import EchoExecutor


def test_execution_records_queue_wait_execute_and_batch() -> None:
    async def run():
        release = asyncio.Event()

        class Executor(EchoExecutor):
            async def execute_forward_backward_batch(self, executions):
                await release.wait()
                return await EchoExecutor.execute_forward_backward_batch(
                    self, executions
                )

        timings = CriticalPath()
        server = Engine(Executor(), timings=timings)
        await server.accept_model("model-a", {})
        first = await forward_backward(server, 1)
        second = await forward_backward(server, 2)
        await asyncio.sleep(0.05)
        release.set()
        assert (
            await server.retrieve_future(first, timeout=1)
        ).status == FutureStatus.COMPLETE
        assert (
            await server.retrieve_future(second, timeout=1)
        ).status == FutureStatus.COMPLETE
        await server.close()

        phases = timings.snapshot(model_id="model-a")["phases"]
        assert phases["forward_backward.queue_wait"]["count"] == 2
        assert phases["forward_backward.execute"]["count"] == 1
        assert phases["forward_backward.execute"]["mean_batch"] == 2.0
        assert phases["forward_backward.execute"]["mean_s"] >= 0.05
        assert phases["accept.execute"]["count"] == 1

    asyncio.run(run())


def test_failed_execution_is_still_timed() -> None:
    async def run():
        class Executor(EchoExecutor):
            async def execute_forward_backward_batch(self, executions):
                raise RuntimeError("boom")

        timings = CriticalPath()
        server = Engine(Executor(), timings=timings)
        await server.accept_model("model-a", {})
        request = await forward_backward(server, 1)
        assert (
            await server.retrieve_future(request, timeout=1)
        ).status == FutureStatus.FAILED
        await server.close()

        phases = timings.snapshot(model_id="model-a")["phases"]
        assert phases["forward_backward.execute"]["count"] == 1

    asyncio.run(run())


def test_sampler_publication_separates_capture_from_persist() -> None:
    async def run():
        timings = CriticalPath()
        server = Engine(EchoExecutor(), timings=timings)
        await server.accept_model("model-a", {})
        request = await server.save_weights_for_sampler(
            {"model_id": "model-a", "seq_id": 1, "publish_version": 1}
        )
        assert (
            await server.retrieve_future(request, timeout=1)
        ).status == FutureStatus.COMPLETE
        await server.close()

        phases = timings.snapshot(model_id="model-a")["phases"]
        assert phases["save_weights_for_sampler.capture"]["count"] == 1
        assert phases["save_weights_for_sampler.persist"]["count"] == 1

    asyncio.run(run())


def test_timing_snapshot_is_readable_and_resettable() -> None:
    async def run():
        timings = CriticalPath()
        server = Engine(EchoExecutor(), timings=timings)
        await server.accept_model("model-a", {})
        request = await forward_backward(server, 1)
        assert (
            await server.retrieve_future(request, timeout=1)
        ).status == FutureStatus.COMPLETE

        snapshot = await server.timing(model_id="model-a", reset=True)
        assert snapshot["model_id"] == "model-a"
        assert "forward_backward.execute" in snapshot["phases"]
        assert snapshot["gauges"]["trainer.first_model_ready_s"] > 0
        assert (await server.timing(model_id="model-a"))["phases"] == {}
        await server.close()

    asyncio.run(run())


def test_idle_trainer_reports_to_stdout(capsys) -> None:
    async def run():
        timings = CriticalPath()
        server = Engine(EchoExecutor(), timings=timings, timing_report_interval_s=0.05)
        await server.accept_model("model-a", {})
        await asyncio.sleep(0.12)
        await server.close()

    asyncio.run(run())
    assert "lilo_critical_path" in capsys.readouterr().out


def test_polled_trainer_stays_quiet(capsys) -> None:
    async def run():
        timings = CriticalPath()
        server = Engine(EchoExecutor(), timings=timings, timing_report_interval_s=0.05)
        await server.accept_model("model-a", {})
        for _ in range(4):
            await server.timing()
            await asyncio.sleep(0.03)
        await server.close()

    asyncio.run(run())
    assert "lilo_critical_path" not in capsys.readouterr().out
