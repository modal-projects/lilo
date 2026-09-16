import asyncio

import pytest

from codegolf.pipeline import RolloutBuffer


def test_prefill_overlap_and_lag_bound():
    async def run():
        observed = []

        async def produce(ticket, step, sampler):
            observed.append((ticket, step, sampler))
            await asyncio.sleep(0)
            return [{"ticket": ticket}]

        buffer = RolloutBuffer(
            produce,
            policy=(350, "old"),
            start_ticket=350,
            workers=2,
            capacity=3,
            max_lag=4,
        )
        try:
            await buffer.prefill(3)
            first = await buffer.get(354)
            assert first.policy_step == 350  # exactly four updates old is allowed
            buffer.publish(355, "new")
            second = await buffer.get(355)
            assert second.policy_step == 355  # old queued/in-flight batches discarded
            assert buffer.discarded >= 2
            assert second.ticket != first.ticket
            assert second.records == [{"ticket": second.ticket}]
            assert any(s == 355 and client == "new" for _, s, client in observed)
            assert buffer.queue.qsize() <= 3
        finally:
            await buffer.close()
        assert all(t.done() for t in buffer.tasks)
        assert buffer.queue.empty()

    asyncio.run(run())


def test_worker_failure_does_not_deadlock_empty_consumer_or_prefill():
    async def run(prefill):
        async def fail(*args):
            await asyncio.sleep(0)
            raise RuntimeError("sampler failed")

        buffer = RolloutBuffer(fail, policy=(0, None), start_ticket=0)
        try:
            with pytest.raises(RuntimeError, match="sampler failed"):
                await asyncio.wait_for(
                    buffer.prefill(1) if prefill else buffer.get(0), 1
                )
        finally:
            await buffer.close()

    asyncio.run(run(True))
    asyncio.run(run(False))


def test_recovery_drains_producer_cleanup_and_never_reuses_old_queue():
    async def run():
        started = asyncio.Event()
        cleaned = asyncio.Event()

        async def produce(*args):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                cleaned.set()

        buffer = RolloutBuffer(produce, policy=(0, None), start_ticket=0, workers=1)
        await started.wait()
        await buffer.close()
        assert cleaned.is_set()
        assert buffer.queue.empty()

    asyncio.run(run())
