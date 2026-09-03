import asyncio
import importlib
import itertools
import json
import os
import time
from types import SimpleNamespace

import httpx
import pytest
import tinker
from tinker import types

from lilo.client import create_full_training_client
from lilo.control_plane import ControlPlane, create_control_plane_app
from lilo.engine import OperationKind
from lilo.providers import SamplingTask
from lilo.providers.local import (
    InMemoryKeyValueStore,
    LocalEnginePlatform,
    LocalSamplingTaskPlatform,
)
from tests.support import TinkerStubExecutor, TinkerStubSampler, serve

BASE_MODEL = "Qwen/Qwen3-8B"
DEFINITION = "qwen3_8b"
FULL_DEFINITION = "qwen3_8b_full"
API_KEY = "tml-test"
MAX_CONTEXT_LENGTH = 32_768
DEFINITIONS = (
    SimpleNamespace(
        DEFINITION_ID=DEFINITION,
        MODEL_NAME=BASE_MODEL,
        PARAMETERIZATION="lora",
        CATALOG_VISIBLE=True,
        MAX_CONTEXT_LENGTH=MAX_CONTEXT_LENGTH,
    ),
)
FULL_DEFINITIONS = (
    SimpleNamespace(
        DEFINITION_ID=FULL_DEFINITION,
        MODEL_NAME=BASE_MODEL,
        PARAMETERIZATION="full",
        CATALOG_VISIBLE=True,
        MAX_CONTEXT_LENGTH=MAX_CONTEXT_LENGTH,
    ),
)


LORA_METADATA = {
    "schema_version": 1,
    "base_model": BASE_MODEL,
    "engine_definition_id": DEFINITION,
    "parameterization": {"type": "lora"},
    "lora_config": {
        "rank": 32,
        "train_mlp": True,
        "train_attn": True,
        "train_unembed": True,
    },
}


async def lora_checkpoint_metadata(path: str) -> dict[str, object]:
    if not path.endswith("/weights/snapshot"):
        raise FileNotFoundError(path)
    return LORA_METADATA


async def full_checkpoint_metadata(path: str) -> dict[str, object]:
    if not path.endswith("/weights/snapshot"):
        raise FileNotFoundError(path)
    return {
        "schema_version": 1,
        "base_model": BASE_MODEL,
        "engine_definition_id": FULL_DEFINITION,
        "parameterization": {"type": "full"},
        "lora_config": None,
    }


@pytest.fixture(scope="module")
def base_url():
    plane = ControlPlane(
        InMemoryKeyValueStore(),
        LocalEnginePlatform(DEFINITION, TinkerStubExecutor),
        sampling_tasks=LocalSamplingTaskPlatform(TinkerStubSampler()),
        read_checkpoint_metadata=lora_checkpoint_metadata,
    )
    app = create_control_plane_app(
        plane,
        DEFINITIONS,
        api_key=API_KEY,
        retrieve_window=5.0,
    )
    with serve(app) as url:
        yield url


@pytest.fixture(scope="module")
def full_base_url():
    plane = ControlPlane(
        InMemoryKeyValueStore(),
        LocalEnginePlatform(FULL_DEFINITION, TinkerStubExecutor),
        read_checkpoint_metadata=full_checkpoint_metadata,
    )
    app = create_control_plane_app(
        plane,
        FULL_DEFINITIONS,
        api_key=API_KEY,
        retrieve_window=5.0,
    )
    with serve(app) as url:
        yield url


def test_real_sdk_training_loop(base_url: str) -> None:
    service = tinker.ServiceClient(base_url=base_url, api_key=API_KEY)

    capabilities = service.get_server_capabilities()
    assert [
        (m.model_name, m.max_context_length) for m in capabilities.supported_models
    ] == [(BASE_MODEL, MAX_CONTEXT_LENGTH)]

    training = service.create_lora_training_client(base_model=BASE_MODEL, rank=32)

    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={"target_tokens": [2, 3, 4], "weights": [1.0, 1.0, 1.0]},
    )
    forward_future = training.forward_backward([datum], "cross_entropy")
    optim_future = training.optim_step(types.AdamParams(learning_rate=1e-4))

    forward_result = forward_future.result(timeout=30)
    assert forward_result.metrics["loss:sum"] == 1.25

    optim_result = optim_future.result(timeout=30)
    assert optim_result.metrics == {"lr": 1e-4}


def test_real_sdk_recreates_lora_client_from_state(base_url: str) -> None:
    service = tinker.ServiceClient(base_url=base_url, api_key=API_KEY)
    training = service.create_lora_training_client(base_model=BASE_MODEL, rank=32)
    saved = training.save_state("lora-resume").result(timeout=30)
    assert saved.path == f"tinker://{training.model_id}/weights/snapshot"
    info = service.create_rest_client().get_weights_info_by_tinker_path(
        saved.path
    ).result(timeout=30)
    assert info.base_model == BASE_MODEL
    assert info.is_lora
    assert info.lora_rank == 32

    resumed = service.create_training_client_from_state(
        saved.path,
        user_metadata={"resumed": "true"},
    )
    assert ":train:" in resumed.model_id
    assert resumed.optim_step(types.AdamParams(learning_rate=1e-4)).result(
        timeout=30
    ).metrics == {"lr": 1e-4}


def test_real_sdk_recreates_full_client_from_state(full_base_url: str) -> None:
    service = tinker.ServiceClient(base_url=full_base_url, api_key=API_KEY)
    training = create_full_training_client(service, BASE_MODEL)
    saved = training.save_state("full-resume").result(timeout=30)
    info = service.create_rest_client().get_weights_info_by_tinker_path(
        saved.path
    ).result(timeout=30)
    assert info.base_model == BASE_MODEL
    assert not info.is_lora
    assert info.lora_rank is None

    resumed = service.create_training_client_from_state_with_optimizer(saved.path)
    assert ":train:" in resumed.model_id
    assert resumed.optim_step(types.AdamParams(learning_rate=1e-4)).result(
        timeout=30
    ).metrics == {"lr": 1e-4}


def test_real_sdk_sampling_loop(base_url: str) -> None:
    service = tinker.ServiceClient(base_url=base_url, api_key=API_KEY)
    sampling = service.create_sampling_client(base_model=BASE_MODEL)
    future = sampling.sample(
        prompt=types.ModelInput.from_ints([1, 2, 3]),
        num_samples=2,
        sampling_params=types.SamplingParams(max_tokens=3, seed=42),
    )
    result = future.result(timeout=30)
    assert [list(sequence.tokens) for sequence in result.sequences] == [
        [0, 1, 2],
        [1, 2, 3],
    ]
    assert sampling.get_base_model() == BASE_MODEL


def test_real_sdk_named_sampler_export_and_reload(base_url: str) -> None:
    service = tinker.ServiceClient(base_url=base_url, api_key=API_KEY)
    training = service.create_lora_training_client(base_model=BASE_MODEL)
    saved = training.save_weights_for_sampler("named-reload", ttl_seconds=60).result(
        timeout=30
    )
    assert saved.path == (
        f"tinker://{training.model_id}:train:0/sampler_weights/named-reload"
    )

    sampling = service.create_sampling_client(model_path=saved.path)
    result = sampling.sample(
        prompt=types.ModelInput.from_ints([1]),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=2),
    ).result(timeout=30)
    assert list(result.sequences[0].tokens) == [0, 1]
    assert sampling.get_base_model() == BASE_MODEL


def test_real_sdk_ephemeral_sampler_export(base_url: str) -> None:
    service = tinker.ServiceClient(base_url=base_url, api_key=API_KEY)
    training = service.create_lora_training_client(base_model=BASE_MODEL)
    sampling = training.save_weights_and_get_sampling_client()
    result = sampling.sample(
        prompt=types.ModelInput.from_ints([1]),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=2),
    ).result(timeout=30)
    assert list(result.sequences[0].tokens) == [0, 1]
    assert sampling.get_base_model() == BASE_MODEL


def test_real_sdk_sample_async(base_url: str) -> None:
    async def run() -> None:
        service = tinker.ServiceClient(base_url=base_url, api_key=API_KEY)
        sampling = await service.create_sampling_client_async(base_model=BASE_MODEL)
        result = await sampling.sample_async(
            prompt=types.ModelInput.from_ints([1, 2, 3]),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=2),
        )
        assert list(result.sequences[0].tokens) == [0, 1]

    asyncio.run(run())


def test_real_sdk_samples_base_model_of_full_only_definition() -> None:
    class RecordingTasks(LocalSamplingTaskPlatform):
        def __init__(self) -> None:
            super().__init__(TinkerStubSampler())
            self.submitted: list[SamplingTask] = []

        async def submit(self, task: SamplingTask) -> str:
            self.submitted.append(task)
            return await super().submit(task)

    tasks = RecordingTasks()
    ensured = []

    async def ensure_pool(session) -> None:
        ensured.append(session)

    plane = ControlPlane(
        InMemoryKeyValueStore(),
        LocalEnginePlatform(FULL_DEFINITION, TinkerStubExecutor),
        sampling_tasks=tasks,
        ensure_sampling_pool=ensure_pool,
    )
    app = create_control_plane_app(
        plane,
        FULL_DEFINITIONS,
        api_key=API_KEY,
        retrieve_window=5.0,
    )
    with serve(app) as url:
        service = tinker.ServiceClient(base_url=url, api_key=API_KEY)
        sampling = service.create_sampling_client(base_model=BASE_MODEL)
        result = sampling.sample(
            prompt=types.ModelInput.from_ints([1]),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=2),
        ).result(timeout=30)
    assert list(result.sequences[0].tokens) == [0, 1]
    (task,) = tasks.submitted
    assert task.engine_definition_id == FULL_DEFINITION
    assert task.model_id is None
    assert task.publish_version is None
    assert ensured and all(session.model_id is None for session in ensured)


def test_real_sdk_resubmits_lost_sample() -> None:
    class LoseFirstTask:
        def __init__(self) -> None:
            self.local = LocalSamplingTaskPlatform(TinkerStubSampler())
            self.submitted: list[SamplingTask] = []

        async def submit(self, task: SamplingTask) -> str:
            self.submitted.append(task)
            if len(self.submitted) == 1:
                return "lost"
            return await self.local.submit(task)

        async def retrieve(self, task_id: str, timeout: float = 0.0):
            if task_id == "lost":
                return None
            return await self.local.retrieve(task_id, timeout)

    tasks = LoseFirstTask()
    plane = ControlPlane(
        InMemoryKeyValueStore(),
        LocalEnginePlatform(DEFINITION, TinkerStubExecutor),
        sampling_tasks=tasks,
    )
    app = create_control_plane_app(
        plane,
        DEFINITIONS,
        api_key=API_KEY,
        retrieve_window=0.1,
    )
    with serve(app) as url:
        service = tinker.ServiceClient(base_url=url, api_key=API_KEY)
        sampling = service.create_sampling_client(base_model=BASE_MODEL)
        result = sampling.sample(
            prompt=types.ModelInput.from_ints([1]),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=1),
        ).result(timeout=30)
    assert list(result.sequences[0].tokens) == [0]
    assert [task.payload["seq_id"] for task in tasks.submitted] == [0, 1]


class FakeVolume:
    def reload(self) -> None:
        return None

    def commit(self) -> None:
        return None


def volume_plane(tmp_path, monkeypatch) -> tuple[ControlPlane, object, list[str]]:
    modal_app = importlib.import_module("lilo.providers.modal.app")
    root = tmp_path / "checkpoints"
    monkeypatch.setattr(modal_app, "CHECKPOINT_ROOT", str(root))
    monkeypatch.setattr(modal_app, "checkpoint_volume", FakeVolume())
    clock = itertools.count(1_700_000_000)
    loaded: list[str] = []

    class VolumeExecutor(TinkerStubExecutor):
        async def execute(self, model_id, kind, payload):
            if kind == OperationKind.LOAD_WEIGHTS:
                loaded.append(payload.uri)
            return await super().execute(model_id, kind, payload)

        async def persist_checkpoint(self, model_id, payload, snapshot):
            target = root / model_id / "weights" / payload.destination
            target.mkdir(parents=True)
            (target / "checkpoint_rank0.pt").write_bytes(b"\0" * 64)
            (target / "metadata.json").write_text(json.dumps(LORA_METADATA))
            stamp = next(clock)
            os.utime(target, (stamp, stamp))
            return {"path": str(target), "type": "save_weights"}

    plane = ControlPlane(
        InMemoryKeyValueStore(),
        LocalEnginePlatform(DEFINITION, VolumeExecutor),
        read_checkpoint_metadata=modal_app._read_checkpoint_metadata,
        list_checkpoints=modal_app._list_checkpoints,
        delete_checkpoint=modal_app._delete_checkpoint,
        checkpoint_root=str(root),
    )
    return plane, root, loaded


def test_real_sdk_lists_and_deletes_checkpoints(tmp_path, monkeypatch) -> None:
    plane, root, loaded = volume_plane(tmp_path, monkeypatch)
    app = create_control_plane_app(
        plane, DEFINITIONS, api_key=API_KEY, retrieve_window=5.0
    )
    with serve(app) as url:
        service = tinker.ServiceClient(base_url=url, api_key=API_KEY)
        rest = service.create_rest_client()
        training = service.create_lora_training_client(base_model=BASE_MODEL, rank=32)
        training.save_state("first").result(timeout=30)
        saved = training.save_state("second").result(timeout=30)

        listing = rest.list_checkpoints(training.model_id).result(timeout=30)
        assert [c.checkpoint_id for c in listing.checkpoints] == [
            "weights/second",
            "weights/first",
        ]
        newest = listing.checkpoints[0]
        assert newest.checkpoint_type == "training"
        assert newest.tinker_path == f"tinker://{training.model_id}/weights/second"
        assert saved.path == newest.tinker_path
        assert newest.size_bytes > 64
        assert newest.time.year >= 2023

        run = rest.get_training_run(training.model_id).result(timeout=30)
        assert (run.base_model, run.is_lora, run.lora_rank) == (BASE_MODEL, True, 32)
        assert run.last_checkpoint is not None
        assert run.last_checkpoint.tinker_path == newest.tinker_path
        runs = rest.list_training_runs().result(timeout=30)
        assert [r.training_run_id for r in runs.training_runs] == [training.model_id]
        assert runs.cursor.total_count == 1
        with pytest.raises(tinker.NotFoundError):
            rest.get_training_run("no-such-run").result(timeout=30)

        resumed = service.create_training_client_from_state(newest.tinker_path)
        assert resumed.optim_step(types.AdamParams(learning_rate=1e-4)).result(
            timeout=30
        ).metrics == {"lr": 1e-4}
        training.load_state(newest.tinker_path).result(timeout=30)
        assert loaded[-1] == str(root / training.model_id / "weights" / "second")

        with pytest.raises(tinker.APIStatusError) as archive:
            rest.get_checkpoint_archive_url_from_tinker_path(
                newest.tinker_path
            ).result(timeout=30)
        assert archive.value.status_code == 405
        assert archive.value.body["error"] == "unsupported"
        assert (
            f"modal volume get lilo-checkpoints /{training.model_id}/weights/second"
            in archive.value.body["message"]
        )
        with pytest.raises(tinker.NotFoundError):
            rest.get_checkpoint_archive_url_from_tinker_path(
                f"tinker://{training.model_id}/weights/missing"
            ).result(timeout=30)

        rest.delete_checkpoint_from_tinker_path(newest.tinker_path).result(timeout=30)
        remaining = rest.list_checkpoints(training.model_id).result(timeout=30)
        assert [c.checkpoint_id for c in remaining.checkpoints] == ["weights/first"]
        assert not (root / training.model_id / "weights" / "second").exists()
        with pytest.raises(tinker.NotFoundError):
            rest.delete_checkpoint_from_tinker_path(newest.tinker_path).result(
                timeout=30
            )
        with pytest.raises(tinker.BadRequestError):
            rest.delete_checkpoint(training.model_id, "weights/%2e%2e/first").result(
                timeout=30
            )
        assert (root / training.model_id / "weights" / "first").is_dir()


def test_real_sdk_lost_model_fails_fast(tmp_path, monkeypatch) -> None:
    plane, _, _ = volume_plane(tmp_path, monkeypatch)
    app = create_control_plane_app(
        plane, DEFINITIONS, api_key=API_KEY, retrieve_window=5.0
    )
    with serve(app) as url:
        service = tinker.ServiceClient(base_url=url, api_key=API_KEY)
        training = service.create_lora_training_client(base_model=BASE_MODEL, rank=32)
        [instance] = asyncio.run(plane.engines.list_instances())
        asyncio.run(plane.engines.mark_dead(instance.instance_id))

        started = time.monotonic()
        with pytest.raises(tinker.APIStatusError) as info:
            training.optim_step(types.AdamParams(learning_rate=1e-4)).result(
                timeout=120
            )
        assert time.monotonic() - started < 30
        assert info.value.status_code == 410
        assert info.value.body["error"] == "model_lost"


def test_error_table_over_http() -> None:
    async def explode(model_id):
        raise RuntimeError("volume offline")

    engines = LocalEnginePlatform(DEFINITION, TinkerStubExecutor)
    plane = ControlPlane(InMemoryKeyValueStore(), engines, list_checkpoints=explode)
    app = create_control_plane_app(
        plane, DEFINITIONS, api_key=API_KEY, retrieve_window=1.0
    )
    with (
        serve(app) as url,
        httpx.Client(base_url=url, headers={"x-api-key": API_KEY}) as client,
    ):
        session_id = client.post("/api/v1/create_session", json={}).json()["session_id"]
        create = {
            "session_id": session_id,
            "model_seq_id": 0,
            "base_model": BASE_MODEL,
            "lora_config": {"rank": 32},
        }
        created = client.post("/api/v1/create_model", json=create).json()
        model_id = created["model_id"]
        creation = client.post(
            "/api/v1/retrieve_future", json={"request_id": created["request_id"]}
        )
        assert creation.json()["type"] == "create_model"

        def status(response: httpx.Response) -> tuple[int, str]:
            return response.status_code, response.json()["error"]

        assert status(
            client.post(
                "/api/v1/create_model", json={**create, "lora_config": {"rank": 8}}
            )
        ) == (409, "conflict")
        assert status(
            client.post("/api/v1/session_heartbeat", json={"session_id": "nope"})
        ) == (404, "not_found")
        assert status(client.post("/api/v1/session_heartbeat", json={})) == (
            400,
            "invalid_request",
        )
        internal = httpx.get(
            f"{url}/api/v1/training_runs/{model_id}/checkpoints",
            headers=client.headers,
        )
        assert status(internal) == (500, "internal")
        assert len(internal.json()["error_id"]) == 32

        [instance] = asyncio.run(engines.list_instances())
        step = {"model_id": model_id, "seq_id": 1, "adam_params": {}}
        engines.client(instance.instance_id).draining = True
        assert status(client.post("/api/v1/optim_step", json=step)) == (
            429,
            "saturated",
        )
        engines.client(instance.instance_id).draining = False
        request_id = client.post("/api/v1/optim_step", json=step).json()["request_id"]
        assert (
            "metrics"
            in client.post(
                "/api/v1/retrieve_future", json={"request_id": request_id}
            ).json()
        )

        asyncio.run(engines.mark_dead(instance.instance_id))
        lost = client.post("/api/v1/optim_step", json={**step, "seq_id": 2})
        assert status(lost) == (410, "model_lost")
        failed = client.post(
            "/api/v1/retrieve_future", json={"request_id": created["request_id"]}
        )
        assert failed.status_code == 200
        assert failed.json()["category"] == "server"
        assert "lost" in failed.json()["error"]
