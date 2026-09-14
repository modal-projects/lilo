import importlib
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from lilo.engines import qwen3_5_4b_full_64k, qwen3_6_27b_full_64k
from lilo.run import Pool, stop_children


def test_pool_bounds():
    with pytest.raises(ValueError):
        Pool(min_containers=2, max_containers=1)


def test_cleanup_attempts_all_children_even_on_failure(monkeypatch):
    module = importlib.import_module("lilo.run")
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    calls = []
    def stop(name):
        calls.append(name)
        if name == "broken":
            raise RuntimeError("provider unavailable")
    with pytest.raises(ExceptionGroup):
        stop_children(["broken", "healthy"], stop)
    assert calls == ["broken"] * 3 + ["healthy"]


@pytest.mark.parametrize("warm,body_failure,drain_failure", [(True, False, False), (False, True, False), (True, False, True)])
def test_owned_children_stop_before_parent(monkeypatch, warm, body_failure, drain_failure):
    import modal
    module = importlib.import_module("lilo.run")
    scoped = importlib.import_module("lilo.providers.modal.scoped")
    events = []
    data = {}
    class Registry:
        def put(self, key, value): data[key] = value
        def get(self, key): return data.get(key)
    monkeypatch.setattr(modal.Dict, "from_name", lambda *a, **k: Registry())
    monkeypatch.setattr(modal.Dict.objects, "delete", lambda *a, **k: None)
    monkeypatch.setattr(modal.Secret, "from_name", lambda *a, **k: None)
    @contextmanager
    def parent():
        events.append("parent-start")
        try: yield
        finally: events.append("parent-stop")
    def manage(action):
        events.append(action)
        if action == "close" and drain_failure:
            raise RuntimeError("drain failure")
    def stop(children):
        events.extend("stop-" + child for child in children)
    monkeypatch.setattr(module, "stop_children", stop)
    monkeypatch.setattr(scoped, "build_app", lambda *a: (
        SimpleNamespace(run=parent), SimpleNamespace(get_web_url=lambda: "https://example.invalid"),
        SimpleNamespace(remote=manage), [], SimpleNamespace(remote=lambda: events.append("assets")),
        SimpleNamespace(object_id="im-test")))
    try:
        with module.run(engine=qwen3_5_4b_full_64k(), warm=warm) as (url, key):
            assert url == "https://example.invalid" and key.startswith("tml-")
            data["children"] = ["pinned"]
            if body_failure: raise ValueError("body failure")
    except (ValueError, ExceptionGroup):
        assert body_failure or drain_failure
    assert ("warm" in events) == warm
    assert events.index("stop-pinned") < events.index("parent-stop")


def test_recipes_keep_context_and_topology_together():
    small = qwen3_5_4b_full_64k()
    large = qwen3_6_27b_full_64k()
    small.validate()
    large.validate()
    assert small.training.fp32_lm_head
    assert small.training.optimizer.loss_scale == 1
    assert large.training.tensor_model_parallel_size == 4
    assert large.training.context_parallel_size == 2
    assert large.training.seq_length == 65536
    assert large.training.provider_overrides["recompute_granularity"] == "full"


def test_publication_targets_the_scoped_latest_route(monkeypatch):
    import modal
    from lilo.providers.modal.scoped_pool import publication_pool, ScopedFlashPool
    route = {'url': 'https://sampler.invalid', 'function_id': 'fu-scoped'}
    monkeypatch.setenv('LILO_SCOPED_REGISTRY', 'owned-run')
    monkeypatch.setattr(modal.Dict, 'from_name', lambda name: {'model:abc': route})
    pool = publication_pool('custom-engine', 'abc')
    assert isinstance(pool, ScopedFlashPool)
    assert pool.route == route
    assert pool.gateway_url() == route['url']


def test_shared_publication_keeps_existing_pool(monkeypatch):
    from lilo.providers.modal.scoped_pool import publication_pool
    from lilo.providers.modal.fft_pool import FFTLatestPool
    monkeypatch.delenv('LILO_SCOPED_REGISTRY', raising=False)
    assert isinstance(publication_pool('existing', 'abc'), FFTLatestPool)


def test_latest_minimum_updates_by_id_without_name_lookup(monkeypatch):
    import asyncio
    from modal.client import _Client
    from lilo.providers.modal.scoped_pool import set_minimum
    calls = []
    async def update(request): calls.append(request)
    async def client(): return SimpleNamespace(stub=SimpleNamespace(FunctionUpdateSchedulingParams=update))
    monkeypatch.setattr(_Client, 'from_env', client)
    set_minimum('fu-ephemeral', 1)
    assert calls[0].function_id == 'fu-ephemeral'
    assert calls[0].settings.min_containers == 1


def test_retry_finishes_preparation_without_creating_another_model():
    import asyncio
    from lilo.providers.modal.scoped_control import ScopedControlPlane
    from lilo.providers.local import InMemoryKeyValueStore, LocalEnginePlatform
    from tests.support import EchoExecutor
    prepared = []
    async def prepare(model):
        prepared.append(model.model_id)
        if len(prepared) == 1:
            raise RuntimeError('transient infrastructure error')
    async def check():
        plane = ScopedControlPlane(InMemoryKeyValueStore(),
            LocalEnginePlatform('test', EchoExecutor), prepare_model=prepare)
        session = await plane.create_session()
        args = dict(session_id=session.session_id, model_seq_id=0, definition_id='test', spec={'rank':32})
        with pytest.raises(RuntimeError): await plane.create_model(**args)
        result = await plane.create_model(**args)
        assert prepared == [result.model.model_id]*2
        assert not result.created
    asyncio.run(check())
