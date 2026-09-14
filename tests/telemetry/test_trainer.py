import asyncio

import httpx
import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lilo.engine import Engine
from lilo.engine.http import HttpEngineClient, create_engine_app
from lilo.telemetry import trainer
from tests.support import EchoExecutor


@pytest.fixture
def setup(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trainer, "provider", lambda: provider)
    reader = InMemoryMetricReader()
    telemetry = trainer.TrainerTelemetry(
        "instance", "definition", "boot", metric_reader=reader
    )
    yield telemetry, exporter, reader
    telemetry.close()
    provider.shutdown()


def test_transport_queue_execution_and_duplicate_submission(setup):
    telemetry, exporter, _ = setup

    async def run():
        server = Engine(EchoExecutor(), observer=telemetry)
        await server.accept_model("model", {})
        client = HttpEngineClient(
            "http://engine",
            transport=httpx.ASGITransport(app=create_engine_app(server)),
        )

        async def submit(scope, receive, send):
            request = {"model_id": "model", "seq_id": 1, "adam_params": {}}
            rid = await client.optim_step(request)
            assert await client.optim_step(request) == rid
            assert (
                await server.retrieve_future(rid, timeout=1)
            ).status.value == "complete"
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"{}"})

        app = trainer.CommandMiddleware(submit)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://control"
        ) as http:
            assert (await http.post("/api/v1/optim_step")).status_code == 200
        await client.close()
        await server.close()

    asyncio.run(run())
    spans = exporter.get_finished_spans()
    (command,) = [s for s in spans if s.name == "lilo.command.optim_step"]
    (control,) = [s for s in spans if s.name == "lilo.control.submit"]
    assert command.parent is None
    assert control.parent.span_id == command.context.span_id
    assert not any(s.name == "lilo.trainer.queue" for s in spans)
    for name in ("lilo.command.execute", "lilo.trainer.result_ready"):
        (child,) = [s for s in spans if s.name == name]
        assert child.parent.span_id == command.context.span_id
        assert child.start_time >= command.start_time
        assert child.end_time <= command.end_time
    assert not telemetry.commands


def test_state_one_hot_and_background_overlap(setup):
    telemetry, _, reader = setup
    telemetry.state(("model",), "executing:forward_backward")
    telemetry.set_activity("checkpoint", "save_weights")
    data = reader.get_metrics_data()
    points = data.resource_metrics[0].scope_metrics[0].metrics[0].data.data_points
    for lane in ("execution", "checkpoint", "sampler"):
        assert sum(p.value for p in points if p.attributes["lilo.lane"] == lane) == 1
    active = {
        (p.attributes["lilo.lane"], p.attributes["lilo.operation"])
        for p in points
        if p.value
    }
    assert active == {
        ("execution", "forward_backward"),
        ("checkpoint", "save_weights"),
        ("sampler", "idle"),
    }
    telemetry.state(("model",), "idle")
    assert telemetry.activity["checkpoint"] == "save_weights"


def test_batch_links_all_commands_and_error_omits_payload(setup):
    from types import SimpleNamespace

    from lilo.engine import FutureState, FutureStatus, OperationKind

    telemetry, exporter, _ = setup
    ops = [
        SimpleNamespace(
            request_id=f"{m}:1",
            model_id=m,
            seq_id=1,
            kind=OperationKind.FORWARD_BACKWARD,
        )
        for m in ("a", "b")
    ]
    from lilo.engine.operations import parse_operation_payload

    for index, op in enumerate(ops):
        telemetry.register_model(
            op.model_id, {"user_metadata": {"run_id": "run", "attempt_id": str(index)}}
        )
        op.payload = parse_operation_payload(
            OperationKind.FORWARD_BACKWARD,
            {
                "data": [
                    {
                        "model_input": {
                            "chunks": [
                                {
                                    "type": "encoded_text",
                                    "tokens": list(range(index + 3)),
                                }
                            ]
                        },
                        "loss_fn_inputs": {},
                    }
                ]
                * (index + 1),
                "loss_fn": "cross_entropy",
            },
        )
        telemetry.begin(op)
    import time

    from lilo.telemetry import backend

    backend.received.set(
        {
            "attributes": {
                "lilo.padded_tokens": 16,
                "lilo.packed_microbatch_count": 2,
                "secret": "PRIVATE",
            },
            "models": {"a": {"lilo.loss_tokens": 2}, "b": {"lilo.loss_tokens": 5}},
        }
    )
    telemetry.span(
        ("a", "b"),
        "forward_backward",
        "gpu",
        time.time(),
        seq_ids=[1, 1],
        n=2,
        ok=False,
        error="PRIVATE",
    )
    for op in ops:
        telemetry.finish(op, FutureState(FutureStatus.FAILED, error="PRIVATE"))
    spans = exporter.get_finished_spans()
    (batch,) = [s for s in spans if s.name == "lilo.trainer.forward_backward"]
    roots = [s for s in spans if s.name == "lilo.command.forward_backward"]
    assert batch.parent is None
    assert batch.attributes["lilo.run_id"] == "run"
    assert "lilo.run_attempt_id" not in batch.attributes
    assert {s.attributes["lilo.run_attempt_id"] for s in roots} == {"0", "1"}
    assert batch.attributes["lilo.command_count"] == 2
    assert batch.attributes["lilo.example_count"] == 3
    assert batch.attributes["lilo.input_tokens"] == 11
    assert batch.attributes["lilo.loss_tokens"] == 7
    assert batch.attributes["lilo.padded_tokens"] == 16
    assert sorted(s.attributes["lilo.loss_tokens"] for s in roots) == [2, 5]
    assert all("lilo.padded_tokens" not in s.attributes for s in roots)
    assert backend.received.get() is None
    assert sorted(s.attributes["lilo.input_tokens"] for s in roots) == [3, 8]
    assert {l.context.span_id for l in batch.links} == {
        s.context.span_id for s in roots
    }
    assert all("PRIVATE" not in str(s.attributes) and not s.events for s in spans)


def test_persistence_keeps_original_command_and_overlaps_next_operation(setup):
    telemetry, exporter, _ = setup

    async def run():
        persisting, release = asyncio.Event(), asyncio.Event()

        class Executor(EchoExecutor):
            async def persist_checkpoint(self, *args):
                persisting.set()
                await release.wait()
                return await super().persist_checkpoint(*args)

        server = Engine(Executor(), observer=telemetry)
        await server.accept_model("model", {})
        save_id = await server.save_weights(
            {"model_id": "model", "seq_id": 1, "path": "private-checkpoint"}
        )
        await asyncio.wait_for(persisting.wait(), 1)
        assert telemetry.activity["checkpoint"] == "save_weights"
        optim_id = await server.optim_step(
            {"model_id": "model", "seq_id": 2, "adam_params": {}}
        )
        assert (await server.retrieve_future(optim_id, 1)).status.value == "complete"
        assert (await server.retrieve_future(save_id)).status.value == "pending"
        assert save_id in telemetry.commands and optim_id not in telemetry.commands
        release.set()
        assert (await server.retrieve_future(save_id, 1)).status.value == "complete"
        await server.close()

    asyncio.run(run())
    spans = exporter.get_finished_spans()
    (command,) = [s for s in spans if s.name == "lilo.command.save_weights"]
    (persist,) = [s for s in spans if s.name == "lilo.trainer.persist.save_weights"]
    (optim,) = [s for s in spans if s.name == "lilo.trainer.optim_step"]
    assert persist.parent is None
    assert persist.links[0].context.span_id == command.context.span_id
    assert persist.start_time <= optim.start_time < optim.end_time <= persist.end_time
    for phase in ("capture", "persist"):
        (child,) = [s for s in spans if s.name == "lilo.command." + phase]
        (physical,) = [
            s for s in spans if s.name == "lilo.trainer." + phase + ".save_weights"
        ]
        assert child.parent.span_id == command.context.span_id
        assert (child.start_time, child.end_time) == (
            physical.start_time,
            physical.end_time,
        )
        assert child.links[0].context.span_id == physical.context.span_id
    assert not any(s.name == "lilo.command.wait_persistence" for s in spans)
    assert all("private-checkpoint" not in str(s.attributes) for s in spans)


def test_unload_ends_buffered_command_without_retaining_span(setup):
    telemetry, exporter, _ = setup

    async def run():
        server = Engine(EchoExecutor(), observer=telemetry)
        await server.accept_model("model", {})
        # Sequence 2 must wait for missing sequence 1, then be discarded on unload.
        await server.optim_step({"model_id": "model", "seq_id": 2, "adam_params": {}})
        assert telemetry.commands
        await server.unload_model("model")
        assert not telemetry.commands
        await server.close()

    asyncio.run(run())
    (command,) = [
        s for s in exporter.get_finished_spans() if s.name == "lilo.command.optim_step"
    ]
    assert command.status.status_code.name == "ERROR"


def test_workload_counts_and_independent_execution_for_one_command(setup):
    from tests.engine.test_server import forward_backward

    telemetry, exporter, _ = setup

    async def run():
        server = Engine(EchoExecutor(), observer=telemetry)
        await server.accept_model("model-a", {})
        rid = await forward_backward(server, 1, [1, 2, 3, 4, 5])
        assert (await server.retrieve_future(rid, 1)).status.value == "complete"
        await server.close()

    asyncio.run(run())
    spans = exporter.get_finished_spans()
    (command,) = [s for s in spans if s.name == "lilo.command.forward_backward"]
    (batch,) = [s for s in spans if s.name == "lilo.trainer.forward_backward"]
    assert command.parent is None and batch.parent is None
    assert command.context.trace_id != batch.context.trace_id
    assert batch.links[0].context == command.context
    for span in (command, batch):
        assert span.attributes["lilo.example_count"] == 1
        assert span.attributes["lilo.input_tokens"] == 5
    assert batch.attributes["lilo.command_count"] == 1
    assert "lilo.batch_size" not in batch.attributes


def test_retried_http_submission_joins_original_completed_root(setup):
    telemetry, exporter, _ = setup

    async def run():
        server = Engine(EchoExecutor(), observer=telemetry)
        await server.accept_model("model", {})
        client = HttpEngineClient(
            "http://engine",
            transport=httpx.ASGITransport(app=create_engine_app(server)),
        )

        async def submit(scope, receive, send):
            rid = await client.optim_step(
                {"model_id": "model", "seq_id": 1, "adam_params": {}}
            )
            await server.retrieve_future(rid, 1)
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"{}"})

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=trainer.CommandMiddleware(submit)),
            base_url="http://control",
        ) as http:
            for _ in range(2):
                assert (await http.post("/api/v1/optim_step")).status_code == 200
        await client.close()
        await server.close()

    asyncio.run(run())
    spans = exporter.get_finished_spans()
    (command,) = [s for s in spans if s.name == "lilo.command.optim_step"]
    submissions = [s for s in spans if s.name == "lilo.control.submit"]
    assert len(submissions) == 2
    assert all(
        s.parent.span_id == command.context.span_id
        and s.context.trace_id == command.context.trace_id
        for s in submissions
    )


def test_engine_combines_commands_once_with_aggregate_workload(setup):
    import json

    from lilo.engine import OperationKind

    telemetry, exporter, _ = setup

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()

        class Executor(EchoExecutor):
            async def execute(self, model_id, kind, payload):
                if kind == OperationKind.OPTIM_STEP:
                    entered.set()
                    await release.wait()
                return await super().execute(model_id, kind, payload)

        server = Engine(Executor(), observer=telemetry)
        for model in ("a", "b"):
            await server.accept_model(
                model, {"user_metadata": {"run_id": "run", "attempt_id": model}}
            )
        await server.optim_step({"model_id": "a", "seq_id": 1, "adam_params": {}})
        await asyncio.wait_for(entered.wait(), 1)
        requests = []
        for model, seq, count, tokens in [
            ("a", 2, 1, [1, 2, 3]),
            ("b", 1, 2, [4, 5, 6, 7]),
        ]:
            body = {
                "model_id": model,
                "seq_id": seq,
                "forward_backward_input": {
                    "data": [
                        {
                            "model_input": {"chunks": [{"tokens": tokens}]},
                            "loss_fn_inputs": {},
                        }
                    ]
                    * count,
                    "loss_fn": "cross_entropy",
                },
            }
            requests.append(
                await server.forward_backward(
                    json.dumps(body).encode(), "application/json"
                )
            )
        release.set()
        for rid in requests:
            assert (await server.retrieve_future(rid, 1)).status.value == "complete"
        await server.close()

    asyncio.run(run())
    spans = exporter.get_finished_spans()
    (batch,) = [s for s in spans if s.name == "lilo.trainer.forward_backward"]
    commands = [s for s in spans if s.name == "lilo.command.forward_backward"]
    assert len(commands) == 2
    assert batch.parent is None
    assert batch.attributes["lilo.command_count"] == 2
    assert batch.attributes["lilo.example_count"] == 3
    assert batch.attributes["lilo.input_tokens"] == 11
    assert batch.attributes["lilo.run_id"] == "run"
    assert "lilo.run_attempt_id" not in batch.attributes
    assert {(link.context.trace_id, link.context.span_id) for link in batch.links} == {
        (command.context.trace_id, command.context.span_id) for command in commands
    }

    children = [
        s
        for s in spans
        if s.name == "lilo.command.execute"
        and s.attributes["lilo.operation"] == "forward_backward"
    ]
    assert len(children) == 2
    for command in commands:
        (child,) = [s for s in children if s.parent.span_id == command.context.span_id]
        assert child.context.trace_id == command.context.trace_id
        assert child.start_time == batch.start_time > command.start_time
        assert child.end_time == batch.end_time <= command.end_time
        assert (
            child.attributes["lilo.input_tokens"]
            == command.attributes["lilo.input_tokens"]
        )
        assert (
            child.attributes["lilo.run_attempt_id"]
            == command.attributes["lilo.run_attempt_id"]
        )
        assert [
            (link.context.trace_id, link.context.span_id) for link in child.links
        ] == [(batch.context.trace_id, batch.context.span_id)]


def test_only_scoped_metrics_promote_the_deployment_run_resource(monkeypatch):
    monkeypatch.setenv(
        "OTEL_RESOURCE_ATTRIBUTES",
        "lilo.run_id=owned-run,lilo.run_attempt_id=not-a-metric-tag",
    )
    for scoped in (False, True):
        telemetry = trainer.TrainerTelemetry(
            "physical-instance",
            "definition",
            "boot",
            metric_reader=InMemoryMetricReader(),
            scoped=scoped,
        )
        try:
            # Changing model attempts must not change physical deployment labels.
            telemetry.register_model(
                "m",
                {"user_metadata": {"run_id": "different", "attempt_id": "replacement"}},
            )
            observations = telemetry.observe(None)
            assert observations
            for point in observations:
                assert point.attributes.get("lilo.run_id") == (
                    "owned-run" if scoped else None
                )
                assert "lilo.run_attempt_id" not in point.attributes
                assert (
                    point.attributes["lilo.trainer_instance_id"] == "physical-instance"
                )
        finally:
            telemetry.close()


def test_backend_measurements_cross_http_without_changing_results(
    setup, monkeypatch, tmp_path
):
    from lilo.engine import backend_http
    from lilo.telemetry import backend

    telemetry, exporter, _ = setup
    monkeypatch.setattr(backend_http, "provider", trainer.provider)

    class Executor(EchoExecutor):
        async def execute(self, model_id, kind, payload):
            with backend.phase("optimizer"):
                backend.count("lilo.padded_tokens", 16)
                backend.count("lilo.packed_microbatch_count", 2)
                backend.count("lilo.loss_tokens", 7, model_id=model_id)
            return await super().execute(model_id, kind, payload)

        async def persist_checkpoint(self, model_id, payload, snapshot):
            with backend.phase("checkpoint_write"):
                (tmp_path / "checkpoint.pt").write_bytes(b"12345")
            with backend.phase("checkpoint_commit"):
                pass
            backend.checkpoint_size(str(tmp_path))
            return await super().persist_checkpoint(model_id, payload, snapshot)

    async def run():
        client = backend_http.HttpBackendClient(
            "http://backend",
            transport=httpx.ASGITransport(
                app=backend_http.create_backend_app(Executor())
            ),
        )
        engine = Engine(client, observer=telemetry)
        await engine.accept_model(
            "model", {"base_model": "test/model", "parameterization": "full"}
        )
        rid = await engine.optim_step(
            {"model_id": "model", "seq_id": 1, "adam_params": {}}
        )
        result = await engine.retrieve_future(rid, 1)
        assert result.status.value == "complete"
        assert "telemetry" not in result.result
        save = await engine.save_weights(
            {"model_id": "model", "seq_id": 2, "path": "snapshot"}
        )
        assert (await engine.retrieve_future(save, 1)).status.value == "complete"
        await engine.close()
        await client.close()

    asyncio.run(run())
    spans = exporter.get_finished_spans()
    (physical,) = [s for s in spans if s.name == "lilo.trainer.optim_step"]
    (phase,) = [s for s in spans if s.name == "lilo.backend.optimizer"]
    (command,) = [s for s in spans if s.name == "lilo.command.optim_step"]
    assert phase.context.trace_id == physical.context.trace_id
    assert phase.parent.span_id == physical.context.span_id
    assert (
        physical.start_time <= phase.start_time <= phase.end_time <= physical.end_time
    )
    assert physical.attributes["lilo.padded_tokens"] == 16
    assert physical.attributes["lilo.packed_microbatch_count"] == 2
    assert (
        physical.attributes["lilo.loss_tokens"]
        == command.attributes["lilo.loss_tokens"]
        == 7
    )
    assert "lilo.padded_tokens" not in command.attributes
    assert phase.attributes["lilo.rank"] == 0

    (save,) = [s for s in spans if s.name == "lilo.command.save_weights"]
    (persist,) = [s for s in spans if s.name == "lilo.trainer.persist.save_weights"]
    assert (
        save.attributes["lilo.checkpoint_bytes"]
        == persist.attributes["lilo.checkpoint_bytes"]
        == 5
    )
    for name in ("checkpoint_write", "checkpoint_commit"):
        (phase,) = [s for s in spans if s.name == "lilo.backend." + name]
        assert phase.parent.span_id == persist.context.span_id
    assert "lilo.loss_tokens" not in save.attributes
