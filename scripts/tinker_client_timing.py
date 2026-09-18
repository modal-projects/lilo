"""Client-side per-call timers for the Tinker API request path.

Patches ``tinker.TrainingClient`` async methods and ``httpx.AsyncClient.send``
to accumulate wall-clock timings that can be flushed into per-step metrics
alongside the server-side ``lilo_request_mark`` lines. No dependency on lilo.

Usage in a training loop::

    timer = TinkerClientTimer().install()
    ...one step...
    metrics.update(timer.flush())
"""

from __future__ import annotations

import functools
import json
import sys
import time
from collections.abc import Awaitable, Callable
from typing import ParamSpec, TypeVar

import httpx
import tinker
from tinker.lib.public_interfaces.api_future import APIFuture

P = ParamSpec("P")
T = TypeVar("T")

# TrainingClient coroutine methods that submit work to the control plane.
# Each returns an APIFuture[T]; the future's result_async()/result() resolves
# once the control plane reports completion via retrieve_future long-polls.
TIMED_METHOD_NAMES = (
    "forward_backward_async",
    "optim_step_async",
    "save_weights_for_sampler_async",
    "save_weights_and_get_sampling_client_async",
)

CLIENT_EVENT = "lilo_client_mark"


def _emit(mark_name: str, **fields: object) -> None:
    print(
        json.dumps(
            {"event": CLIENT_EVENT, "mark": mark_name, "ts": time.time(), **fields}
        ),
        file=sys.stderr,
        flush=True,
    )


def _short_name(method: str) -> str:
    return method.removesuffix("_async")


class _TimedFuture(APIFuture[T]):
    """APIFuture proxy that times ``result()``/``result_async()``.

    Timing starts when the wrapped future is handed out (submit returned), so
    ``wait_s`` measures submit-return to result-return latency.
    """

    def __init__(
        self,
        inner: APIFuture[T],
        on_result: Callable[[float, float], None],
    ) -> None:
        self._inner = inner
        self._on_result = on_result
        self._released_ts = time.time()
        self._done = False

    def _record(self) -> None:
        if self._done:
            return
        self._done = True
        now = time.time()
        self._on_result(now - self._released_ts, now)

    async def result_async(self, timeout: float | None = None) -> T:
        try:
            return await self._inner.result_async(timeout)
        finally:
            self._record()

    def result(self, timeout: float | None = None) -> T:
        try:
            return self._inner.result(timeout)
        finally:
            self._record()


class TinkerClientTimer:
    """Monkeypatches TrainingClient async calls and httpx.AsyncClient.send and
    accumulates per-step client-side timings.

    Metrics emitted by ``flush()``:

    - ``client/<name>_calls`` / ``client/<name>_submit_s``: coroutine calls and
      total submit wall time per wrapped TrainingClient method.
    - ``client/<name>_wait_s``: submit-return to result-return, summed.
    - ``client/<name>_submit_to_result_s``: submit-start to result-return,
      summed.
    - ``client/<name>_submit_ts`` / ``client/<name>_result_ts``: epoch of the
      LAST call's submit/result, for correlating with server marks.
    - ``client/http_<segment>_calls`` / ``_s`` / ``_request_bytes`` /
      ``_response_bytes``: per last-URL-segment httpx traffic.
    - ``client/http_status_408_calls``: try_again/long-poll timeout responses.
    - ``client/http_calls``: total instrumented httpx requests.

    Each submit/result/http event also prints one ``lilo_client_mark`` JSON
    line to stderr so a per-call table can be built from the client log.
    """

    def __init__(self) -> None:
        self._metrics: dict[str, float] = {}
        self._installed = False
        self._original_methods: dict[str, Callable[..., object]] = {}
        self._original_send: Callable[..., Awaitable[httpx.Response]] | None = None

    def _add(self, key: str, value: float) -> None:
        self._metrics[key] = self._metrics.get(key, 0.0) + value

    def _set(self, key: str, value: float) -> None:
        self._metrics[key] = value

    def record_call(self, name: str, submit_s: float) -> None:
        """Record one wrapped TrainingClient call's submit latency."""
        self._add(f"client/{name}_calls", 1.0)
        self._add(f"client/{name}_submit_s", submit_s)
        self._set(f"client/{name}_submit_ts", time.time())
        _emit(f"client/{name}.submitted", submit_s=submit_s)

    def record_result(
        self, name: str, wait_s: float, submit_to_result_s: float
    ) -> None:
        """Record a wrapped future resolving."""
        self._add(f"client/{name}_wait_s", wait_s)
        self._add(f"client/{name}_submit_to_result_s", submit_to_result_s)
        self._set(f"client/{name}_result_ts", time.time())
        _emit(
            f"client/{name}.result",
            wait_s=wait_s,
            submit_to_result_s=submit_to_result_s,
        )

    def record_http(
        self,
        segment: str,
        http_s: float,
        request_bytes: int,
        response_bytes: int | None,
        status: int,
    ) -> None:
        """Record one instrumented httpx.AsyncClient.send."""
        self._add("client/http_calls", 1.0)
        self._add(f"client/http_{segment}_calls", 1.0)
        self._add(f"client/http_{segment}_s", http_s)
        self._add(f"client/http_{segment}_request_bytes", float(request_bytes))
        if response_bytes is not None:
            self._add(f"client/http_{segment}_response_bytes", float(response_bytes))
        if status == 408:
            self._add("client/http_status_408_calls", 1.0)
        _emit(
            f"client/http_{segment}.responded",
            http_s=http_s,
            status=status,
            request_bytes=request_bytes,
            response_bytes=response_bytes,
        )

    def flush(self) -> dict[str, float]:
        """Return accumulated metrics and reset the counters."""
        metrics, self._metrics = self._metrics, {}
        return metrics

    def install(self) -> TinkerClientTimer:
        """Monkeypatch TrainingClient methods and httpx.AsyncClient.send.

        Idempotent: repeated calls leave the first install in place.
        """
        if self._installed:
            return self

        originals = {
            "forward_backward_async": tinker.TrainingClient.forward_backward_async,
            "optim_step_async": tinker.TrainingClient.optim_step_async,
            "save_weights_for_sampler_async": (
                tinker.TrainingClient.save_weights_for_sampler_async
            ),
            "save_weights_and_get_sampling_client_async": (
                tinker.TrainingClient.save_weights_and_get_sampling_client_async
            ),
        }
        for name, original in originals.items():
            self._original_methods[name] = original
            patched = self._wrap_training_method(original, _short_name(name))
            match name:
                case "forward_backward_async":
                    tinker.TrainingClient.forward_backward_async = patched  # type: ignore[method-assign]
                case "optim_step_async":
                    tinker.TrainingClient.optim_step_async = patched  # type: ignore[method-assign]
                case "save_weights_for_sampler_async":
                    tinker.TrainingClient.save_weights_for_sampler_async = patched  # type: ignore[method-assign]
                case "save_weights_and_get_sampling_client_async":
                    tinker.TrainingClient.save_weights_and_get_sampling_client_async = (
                        patched  # type: ignore[method-assign]
                    )

        self._original_send = httpx.AsyncClient.send
        httpx.AsyncClient.send = self._wrap_http_send(self._original_send)  # type: ignore[method-assign]
        self._installed = True
        return self

    def uninstall(self) -> None:
        """Restore the original methods."""
        if not self._installed:
            return

        for name, original in self._original_methods.items():
            match name:
                case "forward_backward_async":
                    tinker.TrainingClient.forward_backward_async = original  # type: ignore[method-assign]
                case "optim_step_async":
                    tinker.TrainingClient.optim_step_async = original  # type: ignore[method-assign]
                case "save_weights_for_sampler_async":
                    tinker.TrainingClient.save_weights_for_sampler_async = original  # type: ignore[method-assign]
                case "save_weights_and_get_sampling_client_async":
                    tinker.TrainingClient.save_weights_and_get_sampling_client_async = (
                        original  # type: ignore[method-assign]
                    )
        if self._original_send is not None:
            httpx.AsyncClient.send = self._original_send  # type: ignore[method-assign]
        self._original_methods = {}
        self._original_send = None
        self._installed = False

    def _wrap_training_method(
        self,
        original: Callable[P, Awaitable[APIFuture[T]]],
        name: str,
    ) -> Callable[P, Awaitable[APIFuture[T]]]:
        timer = self

        @functools.wraps(original)
        async def wrapped(*args: P.args, **kwargs: P.kwargs) -> APIFuture[T]:
            submit_started = time.time()
            future = await original(*args, **kwargs)
            submit_s = time.time() - submit_started
            timer.record_call(name, submit_s)
            if not isinstance(future, APIFuture):
                # save_weights_and_get_sampling_client_async resolves inline
                # and returns a SamplingClient; the call itself is the wait.
                timer.record_result(name, 0.0, submit_s)
                return future

            def on_result(wait_s: float, result_ts: float) -> None:
                timer.record_result(name, wait_s, result_ts - submit_started)

            return _TimedFuture(future, on_result)

        return wrapped

    def _wrap_http_send(
        self,
        original: Callable[P, Awaitable[httpx.Response]],
    ) -> Callable[P, Awaitable[httpx.Response]]:
        timer = self

        @functools.wraps(original)
        async def wrapped(*args: P.args, **kwargs: P.kwargs) -> httpx.Response:
            request = args[1]
            assert isinstance(request, httpx.Request)
            segment = request.url.path.rsplit("/", 1)[-1] or "root"
            started = time.time()
            response = await original(*args, **kwargs)
            http_s = time.time() - started
            try:
                request_bytes = len(request.content)
            except httpx.RequestNotRead:
                request_bytes = -1
            try:
                response_bytes: int | None = len(response.content)
            except httpx.ResponseNotRead:
                response_bytes = None
            timer.record_http(
                segment, http_s, request_bytes, response_bytes, response.status_code
            )
            return response

        return wrapped
