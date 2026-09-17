"""Short GPU regression: queue rejections/aborts followed by LoRA eviction.

Run: PYTHONPATH=src modal run scripts/validate_lora_eviction.py --adapter-path ...
Uses a separate one-GPU app and existing immutable adapter files.
"""

from pathlib import Path
import modal
from lilo.providers.modal.rollout_image import image

app = modal.App("lilo-lora-eviction-regression")
validation_image = image.add_local_file(
    Path(__file__).resolve().parents[1]
    / "tests/inference/test_sglang_lora_lifetime.py",
    "/root/test_sglang_lora_lifetime.py",
)


@app.function(image=validation_image, timeout=180)
def unit_checks():
    import subprocess

    subprocess.run(
        ["python", "-m", "pytest", "-q", "/root/test_sglang_lora_lifetime.py"],
        check=True,
    )
    return {"status": "passed"}


@app.function(
    image=validation_image,
    gpu="H200",
    timeout=1200,
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
    from lilo.inference.serving import start_sglang, wait_http, terminate

    subprocess.run(
        ["python", "-m", "pytest", "-q", "/root/test_sglang_lora_lifetime.py"],
        check=True,
    )
    assert Path(adapter_path, "adapter_model.safetensors").is_file()
    proc = start_sglang(
        "/assets/Qwen3.5-9B",
        port=8001,
        context_length=65536,
        max_loras_per_batch=2,
        max_loaded_loras=2,
        max_lora_rank=32,
        max_running_requests=1,
        max_queued_requests=4,
        memory_fraction=0.8,
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
        await asyncio.to_thread(wait_http, "http://127.0.0.1:8001/health", proc, 900)
        async with httpx.AsyncClient(
            base_url="http://127.0.0.1:8001", timeout=120
        ) as c:
            loads = []

            async def load(name):
                start = time.monotonic()
                r = await c.post(
                    "/load_lora_adapter",
                    json={
                        "lora_name": name,
                        "lora_path": adapter_path,
                        "pinned": False,
                    },
                    timeout=30,
                )
                r.raise_for_status()
                assert r.json()["success"], r.text
                models = (await c.get("/v1/models")).json()["data"][1:]
                assert len(models) <= 2
                loads.append(
                    {
                        "name": name,
                        "seconds": time.monotonic() - start,
                        "registered": len(models),
                    }
                )

            def payload(name, rid, n):
                return {
                    "input_ids": [9707, 11, 358, 1079],
                    "lora_path": name,
                    "rid": rid,
                    "sampling_params": {
                        "max_new_tokens": n,
                        "temperature": 0,
                        "ignore_eos": True,
                    },
                }

            await load("validation-0")
            r = await c.post("/generate", json=payload("validation-0", "warmup", 8))
            r.raise_for_status()
            # Reject after acquiring an adapter but before scheduler dispatch.
            invalid = payload("validation-0", "over-context", 1)
            invalid["input_ids"] = [1] * 65537
            r = await c.post("/generate", json=invalid, timeout=30)
            assert r.status_code == 400, (r.status_code, r.text[:300])
            # Prefix caching plus n child requests must retain one logical pin.
            parallel = payload("validation-0", "parallel", 8)
            parallel["sampling_params"]["n"] = 3
            r = await c.post("/generate", json=parallel, timeout=30)
            r.raise_for_status()
            assert len(r.json()) == 3
            # Close a streaming HTTP consumer while its scheduler request is live.
            disconnected = payload("validation-0", "disconnected", 4096)
            disconnected["stream"] = True
            async with c.stream("POST", "/generate", json=disconnected) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        break
            # Abort a parallel group while its prefix/children are in flight.
            parallel_abort = payload("validation-0", "parallel-abort", 4096)
            parallel_abort["sampling_params"]["n"] = 3
            pending = asyncio.create_task(c.post("/generate", json=parallel_abort))
            await asyncio.sleep(1)
            (
                await c.post("/abort_request", json={"rid": "parallel-abort"})
            ).raise_for_status()
            await asyncio.wait_for(pending, 30)
            tasks = [
                asyncio.create_task(
                    c.post(
                        "/generate", json=payload("validation-0", f"pressure-{i}", 4096)
                    )
                )
                for i in range(12)
            ]
            await asyncio.sleep(2)
            await asyncio.gather(
                *(
                    c.post("/abort_request", json={"rid": f"pressure-{i}"})
                    for i in range(12)
                )
            )
            responses = await asyncio.wait_for(asyncio.gather(*tasks), timeout=30)
            statuses = [r.status_code for r in responses]
            assert 503 in statuses, statuses
            assert any(code in (499, 500) for code in statuses), statuses
            # Force repeated evictions immediately instead of accumulating 256 publications.
            for i in range(1, 9):
                await load(f"validation-{i}")
                r = await c.post(
                    "/generate",
                    json=payload(f"validation-{i}", f"after-{i}", 8),
                    timeout=30,
                )
                r.raise_for_status()
            # A sidecar may remember an evicted name; SGLang must reload its cached ref.
            r = await c.post(
                "/generate", json=payload("validation-0", "reload-old", 8), timeout=30
            )
            r.raise_for_status()
            result = {
                "status": "passed",
                "pressure_statuses": statuses,
                "over_context_rejection": "passed",
                "parallel_samples": 3,
                "stream_disconnect": "passed",
                "parallel_abort": "passed",
                "loads": loads,
                "old_version_reload": "passed",
            }
            print(json.dumps(result))
            return result
    finally:
        terminate(proc)


@app.local_entrypoint()
def main(adapter_path: str = "", unit_only: bool = False):
    import json

    if not unit_only and not adapter_path:
        raise ValueError("Provide --adapter-path for GPU validation or use --unit-only")
    result = unit_checks.remote() if unit_only else validate.remote(adapter_path)
    out = Path(__file__).resolve().parent / "results/lora-eviction-validation"
    out.mkdir(parents=True, exist_ok=True)
    (out / ("unit-result.json" if unit_only else "gpu-result.json")).write_text(
        json.dumps(result, indent=2) + "\n"
    )
