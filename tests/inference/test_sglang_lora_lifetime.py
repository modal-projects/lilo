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
from sglang.srt.managers.io_struct import AbortReq, BatchTokenIDOutput
from sglang.srt.managers.tokenizer_manager import ReqState, TokenizerManager


def manager(registry):
    m = TokenizerManager.__new__(TokenizerManager)
    m.enable_lora = True
    m.lora_registry = registry
    m.rid_to_state = {}
    m.child_rid_to_logical_rid = {}
    m.logical_rid_to_child_rids = {}
    m.enable_metrics = False
    m.dump_requests_folder = None
    m.crash_dump_folder = None
    m.incremental_streaming_output = False
    m.server_args = SimpleNamespace(batch_notify_size=32, speculative_algorithm=None)
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
        out_list=[], finished=False, event=asyncio.Event(), obj=obj, time_stats=times
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
