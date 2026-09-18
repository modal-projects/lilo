"""Run against SGLang in the rollout image; no model or GPU is required.

Use the actual finished-output handler, abort response handler and LoRA registry.
The backend-only development environment does not install SGLang.
"""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("sglang")
from fastapi import HTTPException
from sglang.srt.lora.lora_registry import LoRARef, LoRARegistry
from sglang.srt.managers.io_struct import AbortReq, BatchTokenIDOutput, GenerateReqInput
from sglang.srt.managers.tokenizer_manager import (
    LoRARequestRef,
    ReqState,
    TokenizerManager,
)


def manager(registry):
    m = TokenizerManager.__new__(TokenizerManager)
    m.enable_lora = True
    m.lora_registry = registry
    m.rid_to_state = {}
    m.child_rid_to_logical_rid = {}
    m.logical_rid_to_child_rids = {}
    m.enable_metrics = False
    m.enable_trace = False
    m.disaggregation_mode = "null"
    m.lora_ref_cache = {}
    m.dump_requests_folder = None
    m.crash_dump_folder = None
    m.incremental_streaming_output = False
    m.server_args = SimpleNamespace(
        batch_notify_size=32, speculative_algorithm=None, max_loaded_loras=8
    )
    m.config_value = lambda _: "default"
    return m


def state(rid, adapter, stream):
    obj = SimpleNamespace(
        rid=rid,
        lora_path=adapter.lora_name,
        lora_id=adapter.lora_id,
        stream=stream,
        return_logprob=False,
    )
    times = SimpleNamespace(
        first_token_time=1,
        trace_ctx=SimpleNamespace(tracing_enable=False),
        set_finished_time=lambda: None,
        get_e2e_latency=lambda: 0,
    )
    return ReqState(
        out_list=[],
        finished=False,
        event=asyncio.Event(),
        obj=obj,
        time_stats=times,
        lora_ref=LoRARequestRef(adapter.lora_id),
    )


def output(rid, reason):
    fields = dict.fromkeys(BatchTokenIDOutput.__struct_fields__)
    fields.update(
        rids=[rid],
        finished_reasons=[reason],
        output_ids=[[]],
        prompt_tokens=[1],
        completion_tokens=[0],
        reasoning_tokens=[0],
        cached_tokens=[0],
        retraction_counts=[0],
    )
    return BatchTokenIDOutput(**fields)


@pytest.mark.parametrize("status", [200, 400, 499, 500, 503])
@pytest.mark.parametrize("stream", [False, True])
def test_finished_request_releases_once_and_allows_eviction(status, stream):
    async def scenario():
        registry = LoRARegistry()
        adapter = LoRARef(lora_name="version-1", lora_path="/adapter", pinned=False)
        await registry.register(adapter)
        await registry.acquire(adapter.lora_name)
        m = manager(registry)
        s = state("request", adapter, stream)
        m.rid_to_state[s.obj.rid] = s
        reason = (
            {"type": "length", "length": 0}
            if status == 200
            else {"type": "abort", "status_code": status, "message": "test abort"}
        )
        await m._handle_batch_output(output(s.obj.rid, reason))
        # The consumer sees the same completed response and handles its abort.
        if status == 400 and not stream:
            with pytest.raises(ValueError):
                await m._handle_abort_finish_reason(s.out_list[-1], s, stream)
        elif status in (499, 500, 503) and not stream:
            with pytest.raises(HTTPException) as error:
                await m._handle_abort_finish_reason(s.out_list[-1], s, stream)
            assert error.value.status_code == status
        else:
            await m._handle_abort_finish_reason(s.out_list[-1], s, stream)
        await asyncio.sleep(0)  # Drain the common completion handler's release task.
        assert registry._counters[adapter.lora_id].value() == 0
        victim = await registry.lru_lora_name(exclude_pinned=True)
        ident = await registry.unregister(victim)
        await asyncio.wait_for(registry.wait_for_unload(ident), timeout=0.2)

    asyncio.run(scenario())


def test_eviction_still_waits_for_another_live_request():
    async def scenario():
        registry = LoRARegistry()
        adapter = LoRARef(lora_name="version-1", lora_path="/adapter", pinned=False)
        await registry.register(adapter)
        await registry.acquire(adapter.lora_name)
        await registry.acquire(adapter.lora_name)
        m = manager(registry)
        failed = state("failed", adapter, False)
        live = state("live", adapter, False)
        m.rid_to_state.update(failed=failed, live=live)
        reason = {"type": "abort", "status_code": 503, "message": "queue full"}
        await m._handle_batch_output(output("failed", reason))
        with pytest.raises(HTTPException):
            await m._handle_abort_finish_reason(failed.out_list[-1], failed, False)
        await asyncio.sleep(0)
        assert registry._counters[adapter.lora_id].value() == 1
        ident = await registry.unregister(adapter.lora_name)
        eviction = asyncio.create_task(registry.wait_for_unload(ident))
        await asyncio.sleep(0)
        assert not eviction.done()
        await m._handle_batch_output(output("live", {"type": "length", "length": 0}))
        await asyncio.wait_for(eviction, timeout=0.2)

    asyncio.run(scenario())


def test_cpu_eviction_prefers_idle_without_evicting_the_new_adapter():
    async def scenario():
        registry = LoRARegistry()
        busy = LoRARef(lora_name="busy", lora_path="/busy", pinned=False)
        pinned = LoRARef(lora_name="pinned", lora_path="/pinned", pinned=True)
        idle = LoRARef(lora_name="idle", lora_path="/idle", pinned=False)
        new = LoRARef(lora_name="new", lora_path="/new", pinned=False)
        await registry.register(busy)
        await registry.acquire(busy.lora_name)
        for adapter in (pinned, idle, new):
            await registry.register(adapter)
        # Preserve ordinary strict LRU lookups; eviction specifically skips busy.
        assert await registry.lru_lora_name() == "busy"
        assert (
            await registry.lru_lora_name(exclude_pinned=True, exclude_names={"new"})
            == "idle"
        )
        ident = await registry.unregister("idle")
        await asyncio.wait_for(registry.wait_for_unload(ident), 0.2)
        # No idle old victim: fallback to the oldest busy adapter and wait for it.
        assert (
            await registry.lru_lora_name(exclude_pinned=True, exclude_names={"new"})
            == "busy"
        )
        ident = await registry.unregister("busy")
        eviction = asyncio.create_task(registry.wait_for_unload(ident))
        await asyncio.sleep(0)
        assert not eviction.done()
        await registry.release(busy.lora_id)
        await asyncio.wait_for(eviction, 0.2)
        assert (
            await registry.lru_lora_name(exclude_pinned=True, exclude_names={"new"})
            is None
        )

    asyncio.run(scenario())


@pytest.mark.parametrize("registers", [False, True])
def test_implicit_reload_delegates_to_snapshot_owner_and_checks_registration(
    monkeypatch, registers
):
    import httpx

    monkeypatch.setenv(
        "LILO_LORA_RELOAD_URL", "http://sidecar/internal/reload_lora_adapter"
    )

    async def scenario():
        registry = LoRARegistry()
        adapter = LoRARef(lora_name="known", lora_path="/immutable", pinned=False)
        m = manager(registry)
        m.lora_ref_cache[adapter.lora_name] = adapter
        calls = []

        class Client:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post(self, url, json):
                assert url == "http://sidecar/internal/reload_lora_adapter"
                assert json == {"lora_name": "known"}
                calls.append(True)
                if registers:
                    await registry.register(adapter)
                return httpx.Response(200, request=httpx.Request("POST", url))

        monkeypatch.setattr(httpx, "AsyncClient", Client)
        request = GenerateReqInput(input_ids=[1], rid="guarded", lora_path="known")
        request.normalize_batch_and_arguments()
        m._init_req_state(request)
        if not registers:
            with pytest.raises(ValueError, match="did not register"):
                await m._resolve_lora_path(request)
        else:
            await m._resolve_lora_path(request)
            assert request.lora_id == adapter.lora_id
            m._remove_req_state(request.rid)
            await drain(m)
            # A CPU-resident hit never calls the sidecar again.
            request = GenerateReqInput(input_ids=[1], rid="resident", lora_path="known")
            request.normalize_batch_and_arguments()
            m._init_req_state(request)
            await m._resolve_lora_path(request)
            m._remove_req_state(request.rid)
            await drain(m)
            assert registry._counters[adapter.lora_id].value() == 0
        m._remove_req_state(request.rid)
        await drain(m)
        assert calls == [True]

    asyncio.run(scenario())


@pytest.mark.parametrize("status", [None, 400, 499, 500, 503])
@pytest.mark.parametrize("stream", [False, True])
def test_queue_abort_releases_without_a_response_consumer(status, stream):
    async def scenario():
        registry = LoRARegistry()
        adapter = LoRARef(lora_name="version-1", lora_path="/adapter", pinned=False)
        await registry.register(adapter)
        await registry.acquire(adapter.lora_name)
        m = manager(registry)
        s = state("request", adapter, stream)
        m.rid_to_state[s.obj.rid] = s
        reason = (
            None
            if status is None
            else {"type": "abort", "status_code": status, "message": "queue abort"}
        )
        abort = AbortReq(rid=s.obj.rid, finished_reason=reason)
        m._handle_abort_req(abort)
        # A late duplicate abort or batch completion must not release again.
        m._handle_abort_req(abort)
        await m._handle_batch_output(output(s.obj.rid, {"type": "length", "length": 0}))
        await asyncio.sleep(0)
        assert registry._counters[adapter.lora_id].value() == 0
        ident = await registry.unregister(adapter.lora_name)
        await asyncio.wait_for(registry.wait_for_unload(ident), timeout=0.2)
        assert s.finished and s.event.is_set()

    asyncio.run(scenario())


def test_normal_completion_followed_by_abort_echo_releases_once():
    async def scenario():
        registry = LoRARegistry()
        adapter = LoRARef(lora_name="version-1", lora_path="/adapter", pinned=False)
        await registry.register(adapter)
        await registry.acquire(adapter.lora_name)
        m = manager(registry)
        s = state("request", adapter, False)
        m.rid_to_state[s.obj.rid] = s
        await m._handle_batch_output(output(s.obj.rid, {"type": "length", "length": 0}))
        m._handle_abort_req(AbortReq(rid=s.obj.rid))
        await asyncio.sleep(0)
        assert registry._counters[adapter.lora_id].value() == 0

    asyncio.run(scenario())


async def drain(m):
    while m.__dict__.get("_lora_tasks"):
        await asyncio.gather(*tuple(m._lora_tasks))


async def setup_request(*, paths="version-1", n=1):
    registry = LoRARegistry()
    names = {paths} if isinstance(paths, str) else set(paths) - {None}
    adapters = {}
    for name in names:
        adapter = LoRARef(lora_name=name, lora_path="/adapter", pinned=False)
        await registry.register(adapter)
        adapters[name] = adapter
    obj = GenerateReqInput(
        input_ids=[1, 2] if isinstance(paths, str) else [[1, 2] for _ in paths],
        lora_path=paths,
        sampling_params={"n": n},
        rid="parent",
    )
    obj.normalize_batch_and_arguments()
    m = manager(registry)
    lifecycles = m._init_req_state(obj)
    return m, obj, lifecycles, adapters


@pytest.mark.parametrize("dispatched", [False, True])
def test_exception_cleanup_waits_for_scheduler_if_dispatched(dispatched):
    async def scenario():
        m, obj, lifecycles, adapters = await setup_request()
        await m._resolve_lora_path(obj)
        s = m.rid_to_state[obj.rid]
        s.dispatched = dispatched
        aborts = []
        m._dispatch_to_scheduler = aborts.append
        m._discard_pending_req_states(obj, lifecycles)
        await drain(m)
        adapter = adapters["version-1"]
        assert m.lora_registry._counters[adapter.lora_id].value() == int(dispatched)
        ident = await m.lora_registry.unregister(adapter.lora_name)
        eviction = asyncio.create_task(m.lora_registry.wait_for_unload(ident))
        await asyncio.sleep(0)
        if dispatched:
            assert not eviction.done()
            assert len(aborts) == 1 and obj.rid in m.rid_to_state
            m._handle_abort_req(AbortReq(rid=obj.rid))
        await asyncio.wait_for(eviction, 0.2)
        m._discard_pending_req_states(obj, lifecycles)
        assert obj.rid not in m.rid_to_state
        await drain(m)

    asyncio.run(scenario())


def test_cleanup_before_acquisition_does_not_release():
    async def scenario():
        m, obj, lifecycles, adapters = await setup_request()
        m._discard_pending_req_states(obj, lifecycles)
        await drain(m)
        assert m.lora_registry._counters[adapters["version-1"].lora_id].value() == 0

    asyncio.run(scenario())


def test_cancelled_acquisition_releases_without_touching_reused_rid():
    async def scenario():
        m, obj, lifecycles, adapters = await setup_request()
        entered, resume = asyncio.Event(), asyncio.Event()
        acquire = m.lora_registry.acquire

        async def delayed(paths):
            ids = await acquire(paths)
            entered.set()
            await resume.wait()
            return ids

        m.lora_registry.acquire = delayed
        task = asyncio.create_task(m._resolve_lora_path(obj))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        m._discard_pending_req_states(obj, lifecycles)
        replacement = GenerateReqInput(input_ids=[1], rid=obj.rid)
        replacement.normalize_batch_and_arguments()
        m._init_req_state(replacement)
        new_state = m.rid_to_state[obj.rid]
        resume.set()
        await drain(m)
        assert m.rid_to_state[obj.rid] is new_state
        assert new_state.lora_ref is None
        assert m.lora_registry._counters[adapters["version-1"].lora_id].value() == 0

    asyncio.run(scenario())


def test_partial_batch_cleanup_releases_only_undispatched_references():
    async def scenario():
        m, obj, lifecycles, adapters = await setup_request(
            paths=["version-1", "version-1", "version-2", None]
        )
        await m._resolve_lora_path(obj)
        m.rid_to_state[obj.rid[0]].dispatched = True
        m._dispatch_to_scheduler = lambda _: None
        m._discard_pending_req_states(obj, lifecycles)
        await drain(m)
        assert m.lora_registry._counters[adapters["version-1"].lora_id].value() == 1
        assert m.lora_registry._counters[adapters["version-2"].lora_id].value() == 0
        assert list(m.rid_to_state) == [obj.rid[0]]
        m._handle_abort_req(AbortReq(rid=obj.rid[0]))
        await drain(m)
        assert m.lora_registry._counters[adapters["version-1"].lora_id].value() == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel", [False, True])
def test_parallel_samples_share_parent_reference_until_last_completion(cancel):
    async def scenario():
        import copy

        m, obj, lifecycles, adapters = await setup_request(n=3)
        await m._resolve_lora_path(obj)
        rid = obj.rid[0]
        ident = adapters["version-1"].lora_id
        # Prefix-cache child completes before the actual samples are submitted.
        child = copy.copy(obj[0])
        child.rid = "prefix"
        m._init_child_req_state(rid, child)
        m._remove_req_state(child.rid)
        await drain(m)
        assert m.lora_registry._counters[ident].value() == 1
        for i in range(3):
            child = copy.copy(obj[0])
            child.rid = f"child-{i}"
            m._init_child_req_state(rid, child)
            m.rid_to_state[child.rid].dispatched = True
        if cancel:
            m._dispatch_to_scheduler = lambda _: None
            m._discard_pending_req_states(obj, lifecycles)
        else:
            m._remove_req_state(rid)
        for i in range(3):
            m._handle_abort_req(AbortReq(rid=f"child-{i}"))
            await drain(m)
            assert m.lora_registry._counters[ident].value() == int(i < 2)
        assert not m.rid_to_state and not m.logical_rid_to_child_rids

    asyncio.run(scenario())


def test_late_abort_consumer_cannot_remove_reused_rid():
    async def scenario():
        m, obj, _, _ = await setup_request()
        await m._resolve_lora_path(obj)
        old = m.rid_to_state[obj.rid]
        reason = {"type": "abort", "status_code": 499, "message": "cancelled"}
        m._handle_abort_req(AbortReq(rid=obj.rid, finished_reason=reason))
        replacement = GenerateReqInput(input_ids=[1], rid=obj.rid)
        replacement.normalize_batch_and_arguments()
        m._init_req_state(replacement)
        new = m.rid_to_state[obj.rid]
        with pytest.raises(HTTPException):
            await m._handle_abort_finish_reason(old.out_list[-1], old, False)
        assert m.rid_to_state[obj.rid] is new
        await drain(m)

    asyncio.run(scenario())


def test_registry_acquire_pins_before_unregister_can_observe_zero():
    async def scenario():
        m, _, _, adapters = await setup_request()
        registry = m.lora_registry
        adapter = adapters["version-1"]
        counter = registry._counters[adapter.lora_id]
        entered, resume = asyncio.Event(), asyncio.Event()
        increment = counter.increment

        async def delayed(**kwargs):
            entered.set()
            await resume.wait()
            return await increment(**kwargs)

        counter.increment = delayed
        acquisition = asyncio.create_task(registry.acquire(adapter.lora_name))
        await entered.wait()
        removal = asyncio.create_task(registry.unregister(adapter.lora_name))
        await asyncio.sleep(0)
        assert not removal.done()
        resume.set()
        ident = await acquisition
        assert await removal == ident
        eviction = asyncio.create_task(registry.wait_for_unload(ident))
        await asyncio.sleep(0)
        assert not eviction.done()
        await registry.release(ident)
        await asyncio.wait_for(eviction, 0.2)

    asyncio.run(scenario())


def test_registry_cancelled_partial_acquisition_rolls_back():
    async def scenario():
        m, _, _, adapters = await setup_request(paths=["version-1", "version-2"])
        registry = m.lora_registry
        entered = asyncio.Event()

        async def blocked(**kwargs):
            entered.set()
            await asyncio.Event().wait()

        registry._counters[adapters["version-2"].lora_id].increment = blocked
        acquisition = asyncio.create_task(registry.acquire(["version-1", "version-2"]))
        await entered.wait()
        assert registry._counters[adapters["version-1"].lora_id].value() == 1
        acquisition.cancel()
        with pytest.raises(asyncio.CancelledError):
            await acquisition
        assert all(counter.value() == 0 for counter in registry._counters.values())

    asyncio.run(scenario())
