"""Paired H200 admission/TTFT benchmark under publication and CPU eviction churn.

PYTHONPATH=src modal run --env kailash-dev scripts/benchmark_lora_admission.py

Measures the first real token in SGLang's SSE stream, even though the existing
sidecar buffers the public response. No synthetic sleeps or changed engine
concurrency. Uses immutable existing snapshots READ ONLY, with local manifests
to simulate 12 publishers. The benchmark never advances training pointers.
"""

from pathlib import Path

import modal

from lilo.providers.modal.rollout_image import image

app = modal.App("lilo-lora-admission-benchmark")
bench_image = image.add_local_file(
    Path(__file__).parent / "benchmarks/lora_sidecar_baseline.py", "/root/baseline.py"
).add_local_file(
    Path(__file__).resolve().parents[1]
    / "tests/inference/test_sglang_lora_lifetime.py",
    "/root/test_sglang_lora_lifetime.py",
)
bulletin_volume = modal.Volume.from_name("lilo-snapshot-bulletin", version=2)
result_volume = modal.Volume.from_name(
    "lilo-inference-latency-results", create_if_missing=True
)


@app.function(
    image=bench_image,
    gpu="H200",
    cpu=8,
    memory=65536,
    timeout=3600,
    volumes={
        "/assets": modal.Volume.from_name("lilo-model-assets"),
        "/bulletin": bulletin_volume,
        "/results": result_volume,
    },
)
async def benchmark(source_run: str, rounds: int = 8):
    import asyncio
    import importlib.util
    import json
    import random
    import subprocess
    import time
    from contextlib import asynccontextmanager

    import httpx
    import uvicorn
    from fastapi import Request
    from fastapi.responses import Response
    from stitch.types import VersionRef

    from lilo.inference.bulletin import PEFT_FILES, SnapshotBulletin
    from lilo.inference.lora_sidecar import create_app
    from lilo.inference.serving import start_sglang, terminate, wait_http

    subprocess.run(
        ["python", "-m", "pytest", "-q", "/root/test_sglang_lora_lifetime.py"],
        check=True,
    )

    spec = importlib.util.spec_from_file_location("baseline", "/root/baseline.py")
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    snapshots = sorted(Path("/bulletin", source_run).glob("weight_v*/snapshot.json"))
    assert len(snapshots) >= 12 * rounds, len(snapshots)
    # Identical source bytes and access order across A/B, including adapter rank.
    sources = [path.parent for path in snapshots[: 12 * rounds]]
    # The simulated publisher must not read from the consumer's mount while its
    # background reload is in progress. Real trainers use a separate mount.
    source_manifests = [
        json.loads((source / "snapshot.json").read_text()) for source in sources
    ]
    result_dir = Path("/results") / str(time.time_ns())
    result_dir.mkdir()
    print(f"RESULT_DIR {result_dir.name}", flush=True)
    root = Path("/tmp/admission-bulletin")
    root.mkdir()
    rng = random.Random(7341)
    prompt = [rng.randrange(1000, 20000) for _ in range(1024)]
    records, loads, refreshes, resolutions = {}, [], [], []

    @asynccontextmanager
    async def expose_reload_callback(sidecar, factory):
        if factory is baseline.create_app:
            # The frozen baseline predates the safety callback. Preserve its
            # original single-lock behavior for the old-version revisit only.
            async def reload_adapter(request: Request):
                name = (await request.json())["lora_name"]
                async with sidecar.state.adapter_lock:
                    response = await sidecar.state.client.post(
                        "/load_lora_adapter",
                        json={
                            "lora_name": name,
                            "lora_path": sidecar.state.registered_adapters[name],
                            "pinned": False,
                        },
                    )
                return Response(
                    response.content,
                    status_code=response.status_code,
                    media_type=response.headers.get("content-type"),
                )

            sidecar.add_api_route(
                "/internal/reload_lora_adapter", reload_adapter, methods=["POST"]
            )
        server = uvicorn.Server(
            uvicorn.Config(
                sidecar,
                host="127.0.0.1",
                port=8002,
                lifespan="off",
                log_level="warning",
                access_log=False,
            )
        )
        task = asyncio.create_task(server.serve())
        try:
            while not server.started:
                if task.done():
                    await task
                    raise RuntimeError("callback server did not start")
                await asyncio.sleep(0.01)
            yield
        finally:
            server.should_exit = True
            await task

    class Bulletin(SnapshotBulletin):
        async def refresh(self):
            start = time.monotonic()
            await super().refresh()
            refreshes.append(time.monotonic() - start)

        def resolve(self, ref):
            start = time.monotonic()
            try:
                return super().resolve(ref)
            finally:
                resolutions.append(time.monotonic() - start)

    class TokenStream(httpx.AsyncByteStream):
        def __init__(self, wrapped, record):
            self.wrapped, self.record = wrapped, record

        async def __aiter__(self):
            pending = b""
            async for chunk in self.wrapped:
                pending += chunk
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    if (
                        not line.startswith(b"data: ")
                        or line.strip() == b"data: [DONE]"
                    ):
                        continue
                    item = json.loads(line[6:])
                    meta = item.get("meta_info", {})
                    if meta.get("completion_tokens", 0) > 0:
                        self.record.setdefault("first_token", time.monotonic())
                    self.record["meta"] = meta
                yield chunk

        async def aclose(self):
            await self.wrapped.aclose()

    class MeasurementTransport(httpx.AsyncBaseTransport):
        def __init__(self):
            self.wrapped = httpx.AsyncHTTPTransport()

        async def handle_async_request(self, request):
            start = time.monotonic()
            body = json.loads(request.content) if request.content else {}
            if request.url.path == "/generate":
                record = records[body["rid"].split(":admit-")[0]]
                record.setdefault("generate_send", start)
                record["generate_attempts"] = record.get("generate_attempts", 0) + 1
            response = await self.wrapped.handle_async_request(request)
            if request.url.path == "/generate":
                response.stream = TokenStream(response.stream, record)
            elif request.url.path == "/load_lora_adapter":
                loads.append(
                    {"seconds": time.monotonic() - start, "name": body["lora_name"]}
                )
            return response

        async def aclose(self):
            await self.wrapped.aclose()

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
        lora_reload_url="http://127.0.0.1:8002/internal/reload_lora_adapter",
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
    result = {
        "config": {
            "baseline_commit": "7600e5bf757c39266a5ba7b3999ba7e6a8e0fb14",
            "gpu": "H200",
            "model": "Qwen3.5-9B",
            "prompt_tokens": 1024,
            "output_tokens": 64,
            "ignore_eos": True,
            "clients": 12,
            "rounds": rounds,
            "requests_per_adapter": 4,
            "request_concurrency": 16,
            "max_running_requests": 32,
            "max_queued_requests": 8,
            "max_loras_per_batch": 4,
            "max_loaded_loras": 64,
            "source_run": source_run,
            "measurement": "sidecar entry to first SGLang SSE token; excludes gateway",
        },
        "phases": [],
    }
    try:
        await asyncio.to_thread(wait_http, "http://127.0.0.1:8001/health", proc, 900)
        async with httpx.AsyncClient(
            base_url="http://127.0.0.1:8001", timeout=180
        ) as engine:

            async def reset():
                models = (await engine.get("/v1/models")).json()["data"][1:]
                for model in models:
                    response = await engine.post(
                        "/unload_lora_adapter", json={"lora_name": model["id"]}
                    )
                    response.raise_for_status()
                response = await engine.post("/flush_cache")
                response.raise_for_status()

            async def phase(label, factory, count):
                await reset()
                records.clear()
                loads.clear()
                refreshes.clear()
                resolutions.clear()
                bulletin = Bulletin(root, refresh=bulletin_volume.reload)
                sidecar = factory(
                    bulletin, "http://127.0.0.1:8001", transport=MeasurementTransport()
                )
                phase_start = time.monotonic()
                residency = []
                async with (
                    sidecar.router.lifespan_context(sidecar),
                    expose_reload_callback(sidecar, factory),
                    httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=sidecar),
                        base_url="http://sidecar",
                        timeout=300,
                    ) as client,
                ):

                    async def sample(client_index, version, sample_index, repeat):
                        rid = (
                            f"{label}-{client_index}-{version}-{sample_index}-{repeat}"
                        )
                        record = records[rid] = {
                            "start": time.monotonic(),
                            "repeat": repeat,
                        }
                        response = await client.post(
                            "/generate",
                            json={
                                "input_ids": prompt,
                                "rid": rid,
                                "stream": True,
                                "weight_run_id": f"{label}-client-{client_index}",
                                "weight_version": {"min_version": version},
                                "sampling_params": {
                                    "max_new_tokens": 64,
                                    "temperature": 0,
                                    "ignore_eos": True,
                                },
                            },
                        )
                        record["finish"] = time.monotonic()
                        record["status"] = response.status_code
                        if response.is_error:
                            record["error"] = response.text[:500]
                        record["admission_s"] = (
                            record.get("generate_send", record["finish"])
                            - record["start"]
                        )
                        if "first_token" in record:
                            record["ttft_s"] = record["first_token"] - record["start"]
                            record["engine_ttft_s"] = (
                                record["first_token"] - record["generate_send"]
                            )
                        record["e2e_s"] = record["finish"] - record["start"]

                    for version in range(1, count + 1):
                        # Immutable manifests with real adapter weights from the
                        # volume; publishing advances each client's local latest.
                        for client_index in range(12):
                            ref = VersionRef(f"{label}-client-{client_index}", version)
                            target = bulletin.snapshot_dir(ref)
                            target.mkdir(parents=True)
                            source = sources[(version - 1) * 12 + client_index]
                            manifest = dict(
                                source_manifests[(version - 1) * 12 + client_index]
                            )
                            manifest["ref"] = ref.identity
                            for name in PEFT_FILES:
                                (target / name).symlink_to(source / name)
                            (target / "snapshot.json").write_text(json.dumps(manifest))
                            bulletin.advance(ref)
                        for offset in range(0, 12, 4):
                            # Warm repeats expose prefix reuse independently of
                            # first-use validation and registration.
                            for repeat in (False, True):
                                await asyncio.gather(
                                    *(
                                        sample(c, version, n, repeat)
                                        for c in range(offset, offset + 4)
                                        for n in range(4)
                                    )
                                )
                        models = (await engine.get("/v1/models")).json()["data"][1:]
                        residency.append(len(models))
                        print(
                            f"{label} publication={version} cpu_adapters={len(models)} requests={len(records)}",
                            flush=True,
                        )
                    # Revisit an evicted old version: exercises SGLang's implicit
                    # reload using names already known to Lilo.
                    if count > 5:
                        for client_index in range(4):
                            payload = {
                                "input_ids": prompt,
                                "rid": f"{label}-revisit-{client_index}",
                                "stream": True,
                                "weight_run_id": f"{label}-client-{client_index}",
                                "weight_version": {"exact_version": 1},
                                "sampling_params": {
                                    "max_new_tokens": 64,
                                    "temperature": 0,
                                    "ignore_eos": True,
                                },
                            }
                            record = records[payload["rid"]] = {
                                "start": time.monotonic(),
                                "revisit": True,
                            }
                            r = await client.post("/generate", json=payload)
                            record.update(status=r.status_code, finish=time.monotonic())
                return {
                    "label": label,
                    "wall_s": time.monotonic() - phase_start,
                    "records": dict(records),
                    "loads": list(loads),
                    "refresh_s": list(refreshes),
                    "resolve_s": list(resolutions),
                    "cpu_residency": residency,
                }

            await phase("warmup", create_app, 1)
            for label, factory in (
                ("baseline-a", baseline.create_app),
                ("optimized-a", create_app),
                ("optimized-b", create_app),
                ("baseline-b", baseline.create_app),
            ):
                data = await phase(label, factory, rounds)
                result["phases"].append(data)
                (result_dir / "benchmark.json").write_text(json.dumps(result))
                await asyncio.to_thread(result_volume.commit)
                print(
                    "PHASE_SUMMARY "
                    + json.dumps(
                        {
                            "label": label,
                            "wall_s": data["wall_s"],
                            "complete": sum(
                                r.get("meta", {}).get("completion_tokens") == 64
                                for r in data["records"].values()
                            ),
                            "total": len(data["records"]),
                        }
                    ),
                    flush=True,
                )
            await reset()
    finally:
        terminate(proc)
    return result


@app.local_entrypoint()
def main(source_run: str = "733f0f3932f05be7d75191cea39b8894", rounds: int = 8):
    import json

    result = benchmark.remote(source_run, rounds)
    path = Path("scripts/results/lora-admission/benchmark-v3.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(f"Saved {path}")
