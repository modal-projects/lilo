import asyncio
import importlib
import json
from types import SimpleNamespace

import pytest

from lilo.errors import RecordNotFound
from lilo.providers.local import InMemoryKeyValueStore
from lilo.providers.modal.fft_pool import FFTPoolSpec

FULL_DEFINITION = "qwen3_5_9b_full_64k"


def test_definitions_exclude_stale_128k_definition() -> None:
    modal_app = importlib.import_module("lilo.providers.modal.app")
    assert "qwen3_5_9b_full_128k" not in {
        definition.DEFINITION_ID for definition in modal_app.DEFINITIONS
    }


def test_ensure_pool_deploys_pinned_base_pool(monkeypatch) -> None:
    modal_app = importlib.import_module("lilo.providers.modal.app")
    registry = InMemoryKeyValueStore()
    deployed = []

    async def ensure(spec: dict) -> str:
        deployed.append(spec)
        await registry.put(f"fft_pool:{FFTPoolSpec(**spec).app_name}", spec)
        return "https://gateway"

    monkeypatch.setattr(modal_app, "shared_kv", InMemoryKeyValueStore)
    monkeypatch.setattr(modal_app, "fft_pool_kv", lambda: registry)
    monkeypatch.setattr(modal_app, "ModalSessionKeyValueStores", SimpleNamespace)
    monkeypatch.setattr(
        modal_app,
        "ensure_fft_pool",
        SimpleNamespace(remote=SimpleNamespace(aio=ensure)),
    )
    session = SimpleNamespace(
        engine_definition_id=FULL_DEFINITION,
        model_id=None,
        publish_version=None,
        latest=False,
    )

    async def run() -> None:
        plane = modal_app._plane()
        await plane.ensure_sampling_pool(session)
        await plane.ensure_sampling_pool(session)

    asyncio.run(run())
    base = FFTPoolSpec.base(FULL_DEFINITION)
    assert deployed == [base.as_dict()]
    assert base.version == 0 and not base.latest
    touch = asyncio.run(registry.get(f"fft_pool_touch:{base.app_name}"))
    assert touch["touched_at"] > 0


def test_prepare_model_spawns_sized_latest_pool_for_full_models(monkeypatch) -> None:
    modal_app = importlib.import_module("lilo.providers.modal.app")
    spawned = []

    async def spawn(spec: dict) -> None:
        spawned.append(spec)

    monkeypatch.setattr(modal_app, "shared_kv", InMemoryKeyValueStore)
    monkeypatch.setattr(modal_app, "fft_pool_kv", InMemoryKeyValueStore)
    monkeypatch.setattr(modal_app, "ModalSessionKeyValueStores", SimpleNamespace)
    monkeypatch.setattr(
        modal_app,
        "ensure_fft_pool",
        SimpleNamespace(spawn=SimpleNamespace(aio=spawn)),
    )

    def model(definition_id: str, spec: dict) -> SimpleNamespace:
        return SimpleNamespace(
            engine_definition_id=definition_id,
            model_id="session:train:0",
            spec=spec,
        )

    async def run() -> None:
        plane = modal_app._plane()
        await plane.prepare_model(
            model(
                FULL_DEFINITION, {"rollout": {"min_containers": 8, "max_containers": 8}}
            )
        )
        await plane.prepare_model(model(FULL_DEFINITION, {}))

    asyncio.run(run())
    assert spawned == [
        FFTPoolSpec(
            FULL_DEFINITION,
            "session:train:0",
            True,
            0,
            min_containers=8,
            max_containers=8,
        ).as_dict(),
        FFTPoolSpec(FULL_DEFINITION, "session:train:0", True, 0).as_dict(),
    ]


def test_ensure_pool_sizes_latest_pool_from_model_rollout_config(monkeypatch) -> None:
    from lilo.control_plane.keys import model_key
    from lilo.control_plane.records import ModelRecord

    modal_app = importlib.import_module("lilo.providers.modal.app")
    kv = InMemoryKeyValueStore()
    registry = InMemoryKeyValueStore()
    deployed = []

    async def ensure(spec: dict) -> str:
        deployed.append(spec)
        await registry.put(f"fft_pool:{FFTPoolSpec.from_dict(spec).app_name}", spec)
        return "https://gateway"

    monkeypatch.setattr(modal_app, "shared_kv", lambda: kv)
    monkeypatch.setattr(modal_app, "fft_pool_kv", lambda: registry)
    monkeypatch.setattr(modal_app, "ModalSessionKeyValueStores", SimpleNamespace)
    monkeypatch.setattr(
        modal_app,
        "ensure_fft_pool",
        SimpleNamespace(remote=SimpleNamespace(aio=ensure)),
    )
    model_id = "session:train:0"
    model = ModelRecord(
        model_id=model_id,
        session_id="session",
        model_seq_id=0,
        engine_definition_id=FULL_DEFINITION,
        spec={"rollout": {"min_containers": 2, "max_containers": 8}},
        created_at=1.0,
    )

    def session(latest: bool, version: int) -> SimpleNamespace:
        return SimpleNamespace(
            engine_definition_id=FULL_DEFINITION,
            model_id=model_id,
            publish_version=version,
            latest=latest,
        )

    async def run() -> None:
        await kv.put(model_key(model_id), model.model_dump(mode="json"))
        plane = modal_app._plane()
        await plane.ensure_sampling_pool(session(True, 0))
        await plane.ensure_sampling_pool(session(True, 0))
        await plane.ensure_sampling_pool(session(False, 3))

    asyncio.run(run())
    assert deployed == [
        FFTPoolSpec(
            FULL_DEFINITION, model_id, True, 0, min_containers=2, max_containers=8
        ).as_dict(),
        FFTPoolSpec(FULL_DEFINITION, model_id, False, 3).as_dict(),
    ]


def test_execute_sample_routes_base_session_without_version(monkeypatch) -> None:
    modal_app = importlib.import_module("lilo.providers.modal.app")
    sampling = importlib.import_module("lilo.inference.sampling")
    specs = []

    async def gateway(spec: FFTPoolSpec) -> str:
        specs.append(spec)
        return "https://gateway"

    async def sample(task, gateway, *, data_parallel_size, **_):
        return {"gateway": gateway, "version": task["publish_version"]}

    monkeypatch.setenv("MODAL_PROXY_TOKEN_ID", "wk-a")
    monkeypatch.setenv("MODAL_PROXY_TOKEN_SECRET", "ws-a")
    monkeypatch.setattr(modal_app, "pool_gateway", gateway)
    monkeypatch.setattr(modal_app, "fft_pool_kv", InMemoryKeyValueStore)
    monkeypatch.setattr(sampling, "sample_task", sample)
    task = {
        "engine_definition_id": FULL_DEFINITION,
        "model_id": None,
        "publish_version": None,
        "latest": False,
    }
    assert asyncio.run(modal_app.execute_sample.local(task)) == {
        "gateway": "https://gateway",
        "version": None,
    }
    assert specs == [FFTPoolSpec.base(FULL_DEFINITION)]

def test_checkpoint_metadata_reader_reloads_existing_volume(
    tmp_path,
    monkeypatch,
) -> None:
    modal_app = importlib.import_module("lilo.providers.modal.app")
    target = tmp_path / "volume"
    root = tmp_path / "checkpoints"
    checkpoint = target / "model" / "weights" / "checkpoint"
    checkpoint.mkdir(parents=True)
    root.symlink_to(target, target_is_directory=True)
    metadata = {
        "schema_version": 1,
        "base_model": "Qwen/Qwen3-4B",
        "engine_definition_id": "qwen3_4b_lora32_16k",
        "parameterization": {"type": "lora"},
        "lora_config": {"rank": 32},
    }
    (checkpoint / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    reloads = []

    class Volume:
        def reload(self) -> None:
            reloads.append(True)

    monkeypatch.setattr(modal_app, "CHECKPOINT_ROOT", str(root))
    monkeypatch.setattr(modal_app, "checkpoint_volume", Volume())

    checkpoint_uri = root / "model" / "weights" / "checkpoint"
    assert (
        asyncio.run(modal_app._read_checkpoint_metadata(str(checkpoint_uri)))
        == metadata
    )
    assert reloads == [True]


def test_sample_touch_refreshes_pinned_pool_at_most_once_per_interval(
    monkeypatch,
) -> None:
    modal_app = importlib.import_module("lilo.providers.modal.app")
    spec = FFTPoolSpec("definition", "model", False, 3)
    clock = [1000.0]
    puts = []

    class Registry:
        async def put(self, key, value):
            puts.append((key, value["touched_at"]))

    monkeypatch.setattr(modal_app, "_pool_touches", {})
    monkeypatch.setattr(modal_app, "fft_pool_kv", Registry)
    monkeypatch.setattr(modal_app, "time", SimpleNamespace(time=lambda: clock[0]))

    async def run() -> None:
        await modal_app._touch_fft_pool(spec)
        await modal_app._touch_fft_pool(spec)
        clock[0] += modal_app.FFT_POOL_TOUCH_INTERVAL
        await modal_app._touch_fft_pool(spec)
        await modal_app._touch_fft_pool(FFTPoolSpec("definition", "model", True, 0))

    asyncio.run(run())

    key = f"fft_pool_touch:{spec.app_name}"
    assert puts == [(key, 1000.0), (key, 1060.0)]


def test_cleanup_redeploys_pool_touched_while_stopping(monkeypatch) -> None:
    modal_app = importlib.import_module("lilo.providers.modal.app")
    spec = FFTPoolSpec("definition", "model", False, 3)
    key = f"fft_pool:{spec.app_name}"
    registry = InMemoryKeyValueStore()
    events = []

    class Pool:
        def __init__(self, *args):
            pass

        async def discover_replicas_async(self):
            return []

    def stop(stopping):
        events.append(f"stop:{stopping.app_name}")
        asyncio.run(modal_app._touch_fft_pool(stopping))

    async def spawn(record):
        events.append(f"deploy:{FFTPoolSpec(**record).app_name}")

    monkeypatch.setattr(modal_app, "_pool_touches", {})
    monkeypatch.setattr(modal_app, "fft_pool_kv", lambda: registry)
    monkeypatch.setattr(modal_app, "shared_kv", InMemoryKeyValueStore)
    monkeypatch.setattr(modal_app, "ModalFlashPool", Pool)
    monkeypatch.setattr(modal_app, "stop_pool", stop)
    monkeypatch.setattr(
        modal_app, "ensure_fft_pool", SimpleNamespace(spawn=SimpleNamespace(aio=spawn))
    )

    async def run() -> tuple[str, ...]:
        await registry.put(key, {**spec.as_dict(), "touched_at": 0.0})
        await registry.put(f"fft_pool_touch:{spec.app_name}", {"touched_at": 1.0})
        return await modal_app._cleanup_fft_pools()

    assert asyncio.run(run()) == (spec.app_name,)
    assert asyncio.run(registry.get(key)) is None
    assert events == [f"stop:{spec.app_name}", f"deploy:{spec.app_name}"]


def test_cleaner_loses_models_on_removed_definitions(monkeypatch) -> None:
    from lilo.control_plane.keys import model_key, placement_key, trainer_demand_key
    from lilo.control_plane.records import ModelRecord

    modal_app = importlib.import_module("lilo.providers.modal.app")
    kv = InMemoryKeyValueStore()
    registry = InMemoryKeyValueStore()
    stopped = []
    monkeypatch.setattr(modal_app, "shared_kv", lambda: kv)
    monkeypatch.setattr(modal_app, "fft_pool_kv", lambda: registry)
    monkeypatch.setattr(
        modal_app, "stop_pool", lambda spec: stopped.append(spec.app_name)
    )
    pools = {}

    async def seed(definition_id: str, model_id: str) -> None:
        record = ModelRecord(
            model_id=model_id,
            session_id="session",
            model_seq_id=0,
            engine_definition_id=definition_id,
            spec={},
            created_at=1.0,
        )
        await kv.put(model_key(model_id), record.model_dump(mode="json"))
        await kv.put(placement_key(model_id), {"placed": True})
        await kv.put(trainer_demand_key(model_id), {"demand": True})
        pool = FFTPoolSpec(definition_id, model_id, True, 0)
        pools[model_id] = pool
        await registry.put(f"fft_pool:{pool.app_name}", pool.as_dict())

    async def run() -> None:
        await seed(FULL_DEFINITION, "live")
        await seed("removed_definition", "orphan")
        assert await modal_app._lose_undefined_models() == ("orphan",)
        assert await modal_app._cleanup_fft_pools() == (pools["orphan"].app_name,)
        assert await kv.get(placement_key("live")) is not None
        assert await kv.get(trainer_demand_key("live")) is not None
        assert await kv.get(placement_key("orphan")) is None
        assert await kv.get(trainer_demand_key("orphan")) is None

    asyncio.run(run())
    assert stopped == [pools["orphan"].app_name]
    assert modal_app.parameterization_for("removed_definition") is None


def test_checkpoint_volume_listing_and_delete(tmp_path, monkeypatch) -> None:
    modal_app = importlib.import_module("lilo.providers.modal.app")
    events: list[str] = []

    class Volume:
        def __init__(self, name: str) -> None:
            self.name = name

        def reload(self) -> None:
            events.append(f"reload:{self.name}")

        def commit(self) -> None:
            events.append(f"commit:{self.name}")

    checkpoints = tmp_path / "checkpoints"
    monkeypatch.setattr(modal_app, "CHECKPOINT_ROOT", str(checkpoints))
    monkeypatch.setattr(modal_app, "checkpoint_volume", Volume("ckpt"))

    lora = checkpoints / "model-a" / "weights" / "step-1"
    lora.mkdir(parents=True)
    (lora / "checkpoint_rank0.pt").write_bytes(b"a" * 10)
    (lora / "metadata.json").write_text(json.dumps({"base_model": "Qwen/Qwen3-4B"}))
    fft = checkpoints / "model-b" / "weights" / "latest"
    fft.mkdir(parents=True)
    (fft / "checkpoint_rank0.pt").write_bytes(b"b" * 7)
    (fft / "nested").mkdir()
    (fft / "nested" / "shard").write_bytes(b"c" * 3)
    (checkpoints / "model-a" / "weights" / "stray.txt").write_text("x")
    (checkpoints / "model-a" / "sampler_weights").mkdir()

    entries = asyncio.run(modal_app._list_checkpoints(None))
    assert sorted(
        (entry["model_id"], entry["name"], entry["size_bytes"], entry["metadata"])
        for entry in entries
    ) == [
        (
            "model-a",
            "step-1",
            10 + (lora / "metadata.json").stat().st_size,
            {"base_model": "Qwen/Qwen3-4B"},
        ),
        ("model-b", "latest", 10, None),
    ]
    assert {entry["path"] for entry in entries} == {str(lora), str(fft)}
    assert [
        entry["name"] for entry in asyncio.run(modal_app._list_checkpoints("model-b"))
    ] == ["latest"]
    assert asyncio.run(modal_app._list_checkpoints("model-c")) == []
    assert events == ["reload:ckpt"] * 3

    events.clear()
    asyncio.run(modal_app._delete_checkpoint(str(lora)))
    assert not lora.exists()
    assert events == ["reload:ckpt", "commit:ckpt"]
    with pytest.raises(RecordNotFound):
        asyncio.run(modal_app._delete_checkpoint(str(lora)))
    with pytest.raises(ValueError):
        asyncio.run(modal_app._delete_checkpoint(str(tmp_path / "elsewhere")))
