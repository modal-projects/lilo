import asyncio
import json
import signal

import httpx
import pytest

from lilo.engine import FutureStatus
from lilo.engine.backend_http import HttpBackendClient
from lilo.providers.modal import serve


@pytest.mark.parametrize("error_type", [httpx.ReadError, httpx.ReadTimeout])
def test_backend_transport_failure_terminates_ranks_and_exits_engine(
    monkeypatch, error_type
):
    class Backend:
        pid = 12345
        returncode = None

        def poll(self):
            return self.returncode

    backend = Backend()
    signals = []
    posted = []
    observed = []
    closed = []

    def popen(args, **kwargs):
        assert kwargs["start_new_session"] is True
        return backend

    def killpg(pid, sig):
        assert pid == backend.pid
        signals.append(sig)
        backend.returncode = -sig

    async def transport(request):
        posted.append(request.url.path)
        if request.url.path == "/execute_forward_backward_batch":
            raise error_type("lost worker response", request=request)
        return httpx.Response(200, json={"result": None})

    def executor(url, **kwargs):
        return HttpBackendClient(url, transport=httpx.MockTransport(transport), **kwargs)

    async def serving(kv, make_server, **kwargs):
        engine = await make_server()
        try:
            await engine.accept_model("model-a", {"base_model": "test/model"})
            request_id = await engine.forward_backward(
                json.dumps(
                    {
                        "model_id": "model-a",
                        "seq_id": 1,
                        "forward_backward_input": {
                            "data": [],
                            "loss_fn": "cross_entropy",
                        },
                    }
                ).encode(),
                "application/json",
            )
            observed.append(await engine.retrieve_future(request_id, timeout=1))
            # Model remains registered. It must not require the idle sweeper to
            # notice a dead controller before the process monitor exits.
            assert await engine.model_ids() == ("model-a",)
            await asyncio.Event().wait()
        finally:
            await engine.close()
            closed.append(True)

    monkeypatch.setattr(serve.subprocess, "Popen", popen)
    monkeypatch.setattr(serve.os, "killpg", killpg)
    monkeypatch.setattr(serve, "HttpBackendClient", executor)
    monkeypatch.setattr(serve, "serve_engine", serving)

    with pytest.raises(RuntimeError, match="backend exited with code"):
        serve.run_engine_with_backend(
            None,
            "unused:executor",
            definition_id="test",
            revision="test",
            instance_id="test",
        )
    assert observed[0].status == FutureStatus.FAILED
    assert signals and all(sig == signal.SIGKILL for sig in signals)
    assert posted.count("/execute_forward_backward_batch") == 1
    assert closed == [True]
