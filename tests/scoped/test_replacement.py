import asyncio
from types import SimpleNamespace

import httpx
import pytest
from stitch.sync import ConstraintUnmet
from stitch.types import VersionConstraint, VersionRef

from lilo.inference.scoped_sidecar import AssignedReconciler, AssignedSnapshotStore, assigned_app, _expected_run
from lilo.providers.modal.scoped_assignment import claim_model


class Registry:
    def __init__(self, values):
        self.values = values
        self.get = SimpleNamespace(aio=self.read)
        self.put = SimpleNamespace(aio=self.write)
    async def read(self, key): return self.values.get(key)
    async def write(self, key, value): self.values[key] = value


def test_replacement_requires_confirmed_loss_and_fences_previous_model():
    async def check():
        values = {'routes': [{}, {'url': 'latest'}]}
        registry = Registry(values)
        active = []
        deleted = []
        async def instances(_): return list(active)
        async def spawn(_): active.append('trainer')
        async def delete(key): deleted.append(key)
        engines = SimpleNamespace(active_instances=instances, spawn_instance=spawn)
        async def get(key): return None
        kv = SimpleNamespace(delete=delete, get=get)
        await claim_model(registry, kv, engines, 'engine', 'a')
        await claim_model(registry, kv, engines, 'engine', 'a')
        assert len(active) == 1
        with pytest.raises(ValueError, match='already active'):
            await claim_model(registry, kv, engines, 'engine', 'b')
        assert values['slot:0'] == 'a'
        active.clear()  # Provider has confirmed the trainer invocation ended.
        await claim_model(registry, kv, engines, 'engine', 'b')
        assert values['slot:0'] == 'b' and deleted == ['trainer_demand:a']
        assert len(active) == 1
        with pytest.raises(ValueError, match='replaced'):
            await claim_model(registry, kv, engines, 'engine', 'a')
    asyncio.run(check())


def test_retry_after_spawn_failure_keeps_assignment():
    async def check():
        registry = Registry({'routes': [{}, {'url': 'latest'}]})
        attempts = []
        async def instances(_): return []
        async def spawn(_):
            attempts.append(1)
            if len(attempts) == 1: raise RuntimeError('provider unavailable')
        engines = SimpleNamespace(active_instances=instances, spawn_instance=spawn)
        with pytest.raises(RuntimeError):
            await claim_model(registry, SimpleNamespace(get=registry.read), engines, 'engine', 'a')
        await claim_model(registry, SimpleNamespace(get=registry.read), engines, 'engine', 'a')
        assert registry.values['slot:0'] == 'a'
    asyncio.run(check())


def test_stitch_run_switch_drains_old_requests_and_gates_new_identity():
    async def check():
        events = []
        async def reset(): events.append('reset')
        async def pause(): events.append('pause')
        async def resume(): events.append('resume')
        engine = SimpleNamespace(reset=reset, pause=pause, resume=resume)
        reconciler = AssignedReconciler(store=None, engine=engine, run_id='a')
        reconciler.applied = VersionRef('a', 7)
        token = _expected_run.set('a')
        async with reconciler.admit(VersionConstraint(min_version=7)):
            switch = asyncio.create_task(reconciler._switch_run('b'))
            await asyncio.sleep(0)
            assert not switch.done() and events == []
        await switch
        assert events == ['pause', 'reset', 'resume']
        assert reconciler.applied == VersionRef('b', 0)
        with pytest.raises(ConstraintUnmet):
            async with reconciler.admit(VersionConstraint()): pass
        _expected_run.reset(token)
        token = _expected_run.set('b')
        with pytest.raises(ConstraintUnmet):
            async with reconciler.admit(VersionConstraint(min_version=1)): pass
        reconciler.applied = VersionRef('b', 1)
        async with reconciler.admit(VersionConstraint(min_version=1)) as served:
            assert served == VersionRef('b', 1)
        _expected_run.reset(token)
    asyncio.run(check())


def test_http_admission_rejects_old_handles_and_waits_for_run_switch():
    async def check():
        registry = Registry({'slot:0': 'b'})
        engine = SimpleNamespace(base_url=lambda: 'http://unused', blocked_routes=lambda: ())
        reconciler = AssignedReconciler(store=None, engine=engine, run_id='a')
        reconciler.applied = VersionRef('a', 999)
        app = assigned_app(reconciler, engine, registry)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            old = await client.post('/generate', json={'weight_run_id': 'a'})
            assert old.status_code == 410
            new = await client.post('/generate', json={'weight_run_id': 'b', 'weight_version': {'min_version': 1}})
            assert new.status_code == 409  # a/999 must never satisfy b/1.
    asyncio.run(check())


def test_assigned_store_changes_pointer_without_rewriting_pinned_history(tmp_path):
    from lilo.inference.full_bulletin import FFTSnapshotBulletin, PinnedFFTSnapshotStore
    board = FFTSnapshotBulletin(tmp_path)
    board.claim('a')
    values = {'slot:0': 'b'}
    store = AssignedSnapshotStore(board, 'a', values)
    store.refresh()
    assert store.read_pointer() == VersionRef('b', 0)
    assert PinnedFFTSnapshotStore(board, VersionRef('a', 0)).read_pointer() == VersionRef('a', 0)


def test_old_creation_retry_cannot_resurrect_a_placed_model():
    async def check():
        registry = Registry({'slot:0': 'a', 'placement:a': {'engine': 'dead'}})
        async def instances(_): return []
        async def spawn(_): raise AssertionError('must not resurrect lost model')
        engines = SimpleNamespace(active_instances=instances, spawn_instance=spawn)
        with pytest.raises(ValueError, match='trainer was lost'):
            await claim_model(registry, SimpleNamespace(get=registry.read), engines, 'engine', 'a')
    asyncio.run(check())


def test_cpu_run_switch_retires_after_drain_without_resetting_cache(monkeypatch):
    import lilo.inference.scoped_sidecar as module
    async def check():
        events = []
        async def reset(): raise AssertionError('CPU cache reset is unsupported')
        async def retire():
            events.append('retired')
            raise RuntimeError('simulated container termination')
        monkeypatch.setattr(module, 'retire_replica', retire)
        engine = SimpleNamespace(delta_update_mode='cpu', reset=reset)
        reconciler = AssignedReconciler(store=None, engine=engine, run_id='a')
        reconciler.applied = VersionRef('a', 1)
        async with reconciler.admit(VersionConstraint(min_version=1)):
            switch = asyncio.create_task(reconciler._switch_run('b'))
            await asyncio.sleep(0)
            assert not switch.done() and events == []
        with pytest.raises(RuntimeError, match='simulated container termination'):
            await switch
        assert events == ['retired']
        assert reconciler.applied == VersionRef('a', 1)  # Never relabel old weights.
    asyncio.run(check())
