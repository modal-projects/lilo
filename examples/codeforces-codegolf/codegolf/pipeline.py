"""Bounded rollout producers; one consumer owns every trainer mutation."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass


@dataclass
class Batch:
    ticket: int
    policy_step: int
    records: list
    seconds: float


class RolloutBuffer:
    def __init__(
        self, produce, *, policy, start_ticket, workers=4, capacity=4, max_lag=4
    ):
        if min(workers, capacity) < 1 or max_lag < 0:
            raise ValueError("Invalid buffer limits")
        self.produce = produce
        self.policy = policy
        self.ticket = start_ticket
        self.max_lag = max_lag
        self.queue = asyncio.Queue(maxsize=capacity)
        self.failure = asyncio.get_running_loop().create_future()
        self.changed = asyncio.Event()
        self.discarded = 0
        self.produced = 0
        self.inflight = 0
        self.closed = False
        self.tasks = [asyncio.create_task(self.worker()) for _ in range(workers)]

    def publish(self, step, sampler):
        # This sampler provides a minimum weight version, so step is a conservative
        # lower bound on every token's behavior policy. Never relabel queued work.
        self.policy = (step, sampler)

    async def worker(self):
        try:
            while True:
                policy_step, sampler = self.policy
                self.ticket += 1
                ticket = self.ticket
                self.inflight += 1
                started = time.monotonic()
                try:
                    records = await self.produce(ticket, policy_step, sampler)
                finally:
                    self.inflight -= 1
                batch = Batch(ticket, policy_step, records, time.monotonic() - started)
                await self.queue.put(batch)
                self.produced += 1
                self.changed.set()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self.failure.done():
                self.failure.set_result(exc)
            self.changed.set()

    def check(self):
        if self.failure.done():
            raise self.failure.result()

    async def prefill(self, target):
        if not 1 <= target <= self.queue.maxsize:
            raise ValueError("Prefill exceeds ready buffer capacity")
        while self.queue.qsize() < target:
            self.changed.clear()
            self.check()
            await self.changed.wait()
        self.check()

    async def get(self, trainer_step):
        while True:
            self.check()
            item = asyncio.create_task(self.queue.get())
            try:
                done, _ = await asyncio.wait(
                    [item, self.failure], return_when=asyncio.FIRST_COMPLETED
                )
                self.check()
                batch = await item
            finally:
                if not item.done():
                    item.cancel()
                    await asyncio.gather(item, return_exceptions=True)
            lag = trainer_step - batch.policy_step
            if lag < 0:
                raise RuntimeError("Rollout policy is ahead of the trainer")
            if lag > self.max_lag:
                self.discarded += 1
                continue
            return batch

    async def close(self):
        self.closed = True
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        while not self.queue.empty():
            self.queue.get_nowait()
