from __future__ import annotations

import asyncio

import tinker
from tinker_cookbook.completers import StopCondition, TokensWithLogprobs


class GroupedTinkerTokenCompleter:
    """Send one Tinker sampling request for an entire rollout group."""

    def __init__(
        self,
        sampling_client: tinker.SamplingClient,
        max_tokens: int,
        temperature: float = 1.0,
        context_window: int | None = None,
        *,
        group_size: int,
    ) -> None:
        self.sampling_client = sampling_client
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.context_window = context_window
        self.group_size = group_size
        self._pending: dict[
            tuple[tuple[int, ...], str, int],
            list[asyncio.Future[TokensWithLogprobs]],
        ] = {}

    async def __call__(
        self,
        model_input: tinker.ModelInput,
        stop: StopCondition,
        *,
        max_tokens: int | None = None,
    ) -> TokensWithLogprobs:
        effective_max_tokens = (
            min(self.max_tokens, max_tokens)
            if max_tokens is not None
            else self.max_tokens
        )
        if self.context_window is not None:
            effective_max_tokens = min(
                effective_max_tokens,
                self.context_window - model_input.length,
            )
            if effective_max_tokens <= 0:
                raise ValueError(
                    f"prompt length {model_input.length} leaves no generation room "
                    f"in context window {self.context_window}"
                )

        key = (
            tuple(model_input.to_ints()),
            repr(stop),
            effective_max_tokens,
        )
        loop = asyncio.get_running_loop()
        future: asyncio.Future[TokensWithLogprobs] = loop.create_future()
        waiters = self._pending.setdefault(key, [])
        waiters.append(future)
        if len(waiters) == 1:
            asyncio.create_task(
                self._flush(
                    key,
                    model_input,
                    stop,
                    effective_max_tokens,
                )
            )
        return await future

    async def _flush(
        self,
        key: tuple[tuple[int, ...], str, int],
        model_input: tinker.ModelInput,
        stop: StopCondition,
        max_tokens: int,
    ) -> None:
        await asyncio.sleep(0)
        waiters = self._pending.pop(key)
        try:
            if len(waiters) != self.group_size:
                raise RuntimeError(
                    f"expected {self.group_size} concurrent samples for one rollout "
                    f"group, received {len(waiters)}"
                )
            result = await self.sampling_client.sample_async(
                prompt=model_input,
                num_samples=self.group_size,
                sampling_params=tinker.SamplingParams(
                    stop=stop,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                ),
            )
            if len(result.sequences) != self.group_size:
                raise RuntimeError(
                    f"expected {self.group_size} grouped samples, "
                    f"received {len(result.sequences)}"
                )
            for waiter, sequence in zip(waiters, result.sequences, strict=True):
                if sequence.logprobs is None:
                    raise RuntimeError("grouped sample is missing token logprobs")
                waiter.set_result(
                    TokensWithLogprobs(
                        tokens=sequence.tokens,
                        maybe_logprobs=sequence.logprobs,
                        stop_reason=sequence.stop_reason,
                    )
                )
        except Exception as exc:  # noqa: BLE001 - propagate sampling errors
            for waiter in waiters:
                if not waiter.done():
                    waiter.set_exception(exc)
