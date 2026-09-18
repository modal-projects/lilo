"""Real-GPU check of admission while CPU LRU eviction encounters a long decode.

Uses identical H200/config/workload for strict LRU and idle-first LRU. Only the
victim-selection predicate changes; limits remain 64 CPU / 4 GPU / 32 running.
The long request is aborted after measurement, so this is a latency comparison,
not a throughput comparison. No production files or deployments are modified.
"""

from pathlib import Path
import modal
from lilo.providers.modal.rollout_image import image

app = modal.App("lilo-lora-idle-eviction-benchmark")
bench_image = image.add_local_file(
    Path(__file__).resolve().parents[1]
    / "tests/inference/test_sglang_lora_lifetime.py",
    "/root/test_sglang_lora_lifetime.py",
)


@app.function(
    image=bench_image,
    gpu="H200",
    cpu=8,
    memory=65536,
    timeout=2400,
    volumes={
        "/assets": modal.Volume.from_name("lilo-model-assets"),
        "/bulletin": modal.Volume.from_name("lilo-snapshot-bulletin", version=2),
    },
)
async def validate(adapter_path: str):
    import asyncio
    import json
    import subprocess
    import time
    import httpx
    from lilo.inference.serving import start_sglang, terminate, wait_http

    subprocess.run(
        ["python", "-m", "pytest", "-q", "/root/test_sglang_lora_lifetime.py"],
        check=True,
    )
    registry_path = Path(
        "/sgl-workspace/sglang/python/sglang/srt/lora/lora_registry.py"
    )
    patched = registry_path.read_text()
    predicate = (
        "if not exclude_pinned or self._counters[lora_ref.lora_id].value() == 0:"
    )
    assert predicate in patched
    results = []
    try:
        for policy in ("strict-lru", "idle-first"):
            registry_path.write_text(
                patched.replace(predicate, "if True:")
                if policy == "strict-lru"
                else patched
            )
            proc = start_sglang(
                "/assets/Qwen3.5-9B",
                port=8001,
                context_length=65536,
                max_loras_per_batch=4,
                max_loaded_loras=64,
                max_lora_rank=32,
                max_running_requests=32,
                max_queued_requests=8,
                memory_fraction=0.8,
                schedule_policy="lpm",
                lora_target_modules=(
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "lm_head",
                ),
            )
            try:
                await asyncio.to_thread(
                    wait_http, "http://127.0.0.1:8001/health", proc, 900
                )
                async with httpx.AsyncClient(
                    base_url="http://127.0.0.1:8001", timeout=180
                ) as c:

                    async def load(name):
                        start = time.monotonic()
                        r = await c.post(
                            "/load_lora_adapter",
                            json={
                                "lora_name": name,
                                "lora_path": adapter_path,
                                "pinned": False,
                            },
                        )
                        r.raise_for_status()
                        assert r.json()["success"], r.text
                        return time.monotonic() - start

                    def payload(name, rid, n):
                        return {
                            "input_ids": [9707, 11, 358, 1079] * 256,
                            "lora_path": name,
                            "rid": rid,
                            "stream": True,
                            "sampling_params": {
                                "max_new_tokens": n,
                                "temperature": 0,
                                "ignore_eos": True,
                            },
                        }

                    for i in range(64):
                        await load(f"adapter-{i}")
                    # Warm both prefill/decode kernels before the measured work.
                    r = await c.post("/generate", json=payload("adapter-0", "warm", 64))
                    r.raise_for_status()
                    busy_started = asyncio.Event()
                    busy_finished = asyncio.Event()

                    async def long_decode():
                        async with c.stream(
                            "POST", "/generate", json=payload("adapter-0", "busy", 8192)
                        ) as r:
                            r.raise_for_status()
                            async for line in r.aiter_lines():
                                if line.startswith("data: {"):
                                    meta = json.loads(line[6:]).get("meta_info", {})
                                    if meta.get("completion_tokens", 0):
                                        busy_started.set()
                        busy_finished.set()

                    busy = asyncio.create_task(long_decode())
                    try:
                        await asyncio.wait_for(busy_started.wait(), 60)
                        # Touch every other adapter after the busy one, making
                        # that busy adapter the oldest without private APIs.
                        for i in range(1, 64):
                            r = await c.post(
                                "/generate",
                                json=payload(f"adapter-{i}", f"touch-{i}", 1),
                            )
                            r.raise_for_status()
                        assert not busy_finished.is_set(), (
                            "long decode finished before cache pressure"
                        )
                        start = time.monotonic()
                        load_s = await load("new-publication")
                        first_token_s = None
                        async with c.stream(
                            "POST",
                            "/generate",
                            json=payload("new-publication", "measured", 64),
                        ) as r:
                            r.raise_for_status()
                            async for line in r.aiter_lines():
                                if line.startswith("data: {"):
                                    meta = json.loads(line[6:]).get("meta_info", {})
                                    if (
                                        meta.get("completion_tokens", 0)
                                        and first_token_s is None
                                    ):
                                        first_token_s = time.monotonic() - start
                        models = (await c.get("/v1/models")).json()["data"][1:]
                        entry = {
                            "policy": policy,
                            "load_s": load_s,
                            "admission_to_first_token_s": first_token_s,
                            "busy_finished": busy_finished.is_set(),
                            "cpu_adapters": len(models),
                        }
                        results.append(entry)
                        print("EVICTION_RESULT " + json.dumps(entry), flush=True)
                    finally:
                        await c.post("/abort_request", json={"rid": "busy"})
                        await asyncio.wait_for(busy, 30)
            finally:
                terminate(proc)
    finally:
        registry_path.write_text(patched)
    return results


@app.local_entrypoint()
def main(
    adapter_path: str = "/bulletin/733f0f3932f05be7d75191cea39b8894/weight_v000001",
    prefix_only: bool = False,
):
    import json

    result = (
        prefix_probe.remote(adapter_path)
        if prefix_only
        else validate.remote(adapter_path)
    )
    path = Path("scripts/results/lora-admission") / (
        "prefix-cache.json" if prefix_only else "idle-eviction.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(result)


@app.function(
    image=bench_image,
    gpu="H200",
    cpu=8,
    memory=65536,
    timeout=1200,
    volumes={
        "/assets": modal.Volume.from_name("lilo-model-assets"),
        "/bulletin": modal.Volume.from_name("lilo-snapshot-bulletin", version=2),
    },
)
async def prefix_probe(adapter_path: str):
    import asyncio
    import httpx
    from lilo.inference.serving import start_sglang, terminate, wait_http

    proc = start_sglang(
        "/assets/Qwen3.5-9B",
        port=8001,
        context_length=65536,
        max_loras_per_batch=4,
        max_loaded_loras=64,
        max_lora_rank=32,
        max_running_requests=32,
        max_queued_requests=8,
        memory_fraction=0.8,
        schedule_policy="lpm",
        lora_target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ),
    )
    results = []
    try:
        await asyncio.to_thread(wait_http, "http://127.0.0.1:8001/health", proc, 900)
        async with httpx.AsyncClient(
            base_url="http://127.0.0.1:8001", timeout=180
        ) as c:
            r = await c.post(
                "/load_lora_adapter",
                json={"lora_name": "probe", "lora_path": adapter_path, "pinned": False},
            )
            r.raise_for_status()
            assert r.json()["success"], r.text
            for adapter in (None, "probe"):
                for length in (1024, 1025, 2048, 2049, 4096, 4097):
                    r = await c.post("/flush_cache")
                    r.raise_for_status()
                    prompt = ([9707, 11, 358, 1079] * ((length + 3) // 4))[:length]
                    for repeat in range(3):
                        r = await c.post(
                            "/generate",
                            json={
                                "input_ids": prompt,
                                "lora_path": adapter,
                                "sampling_params": {
                                    "max_new_tokens": 16,
                                    "temperature": 0,
                                    "ignore_eos": True,
                                },
                            },
                        )
                        r.raise_for_status()
                        meta = r.json()["meta_info"]
                        results.append(
                            {
                                "adapter": adapter,
                                "prompt_tokens": length,
                                "repeat": repeat,
                                "cached_tokens": meta.get("cached_tokens"),
                                "completion_tokens": meta.get("completion_tokens"),
                                "e2e_latency": meta.get("e2e_latency"),
                            }
                        )
    finally:
        terminate(proc)
    return results
