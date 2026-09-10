from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

import modal

APP_NAME = "lilo-multiturn-kv-ablation-v3"
MODEL = "Qwen/Qwen3-0.6B"
PROXY_PORT = 8000
SGLANG_PORT = 8001
TIMEOUT = 20 * 60

image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.17")
    .entrypoint([])
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "SGLANG_DISABLE_CUDNN_CHECK": "1",
        }
    )
)
client_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_python_source("lilo")
)
app = modal.App(APP_NAME)
proxy_secret = modal.Secret.from_name(
    "lilo-proxy",
    required_keys=["MODAL_PROXY_TOKEN_ID", "MODAL_PROXY_TOKEN_SECRET"],
)


def _wait_ready(
    process: subprocess.Popen,
    port: int,
    path: str = "health",
) -> None:
    deadline = time.monotonic() + TIMEOUT
    error: BaseException | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"SGLang exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/{path}",
                timeout=5,
            ) as response:
                if response.status < 500:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            error = exc
        time.sleep(2)
    raise TimeoutError(f"service on port {port} did not become ready: {error}")


def proxy_app():
    import uuid

    import httpx
    from fastapi import FastAPI
    from fastapi.responses import Response

    proxy = FastAPI()
    replica_id = uuid.uuid4().hex
    print(json.dumps({"proxy_replica_id": replica_id}), flush=True)

    @proxy.get("/ready")
    async def ready() -> Response:
        return Response(status_code=204)

    @proxy.get("/health")
    async def health() -> Response:
        try:
            async with httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{SGLANG_PORT}",
                timeout=5,
            ) as client:
                response = await client.get("/health")
            status_code = response.status_code
        except httpx.TransportError:
            status_code = 503
        return Response(
            content=json.dumps({"status": "ok", "replica_id": replica_id}),
            status_code=status_code,
            media_type="application/json",
            headers={"X-Lilo-Replica-ID": replica_id},
        )

    @proxy.post("/generate")
    async def generate(body: dict) -> Response:
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{SGLANG_PORT}",
            timeout=TIMEOUT,
        ) as client:
            response = await client.post(
                "/generate",
                json=body,
            )
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type"),
            headers={"X-Lilo-Replica-ID": replica_id},
        )

    return proxy


@app.server(
    image=image,
    gpu="L4",
    min_containers=2,
    max_containers=2,
    target_concurrency=2,
    scaledown_window=5 * 60,
    startup_timeout=TIMEOUT,
    exit_grace_period=30,
    port=PROXY_PORT,
    routing_region="us-west",
)
class Server:
    @modal.enter()
    def start(self) -> None:
        self.sglang = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sglang.launch_server",
                "--model-path",
                MODEL,
                "--host",
                "0.0.0.0",
                "--port",
                str(SGLANG_PORT),
                "--context-length",
                "4096",
                "--mem-fraction-static",
                "0.75",
                "--max-running-requests",
                "8",
                "--schedule-policy",
                "lpm",
                "--disable-cuda-graph",
                "--skip-server-warmup",
            ],
            start_new_session=True,
        )
        self.proxy = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "validate_multiturn_cache_affinity_modal:proxy_app",
                "--factory",
                "--host",
                "0.0.0.0",
                "--port",
                str(PROXY_PORT),
            ],
            start_new_session=True,
        )
        _wait_ready(self.proxy, PROXY_PORT, "ready")

    @modal.exit()
    def stop(self) -> None:
        for process in (self.proxy, self.sglang):
            if process.poll() is not None:
                continue
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)


async def _wait_for_pool(
    gateway: str,
    headers: dict[str, str],
) -> set[str]:
    import httpx

    deadline = time.monotonic() + TIMEOUT
    async with httpx.AsyncClient(
        base_url=gateway,
        headers=headers,
        timeout=30,
    ) as client:
        while time.monotonic() < deadline:
            responses = await asyncio.gather(
                *(
                    client.get(
                        "/health",
                        headers={"Modal-Session-ID": f"pool-ready-{index}"},
                    )
                    for index in range(16)
                ),
                return_exceptions=True,
            )
            replicas = set()
            for response in responses:
                if not isinstance(response, httpx.Response):
                    continue
                try:
                    replica_id = response.json().get("replica_id")
                except ValueError:
                    continue
                if response.status_code == 200 and replica_id:
                    replicas.add(str(replica_id))
            if all(
                isinstance(response, httpx.Response) and response.status_code == 200
                for response in responses
            ) and len(replicas) >= 2:
                print(
                    json.dumps(
                        {"live_modal_replicas": sorted(replicas)},
                        sort_keys=True,
                    ),
                    flush=True,
                )
                return replicas
            await asyncio.sleep(2)
    raise TimeoutError("the fixed two-replica pool did not become ready")


async def _run_trajectory(
    gateway: str,
    *,
    arm: str,
    trajectory: int,
    use_affinity: bool,
    turns: int,
    initial_prompt_tokens: int,
    output_tokens: int,
    headers: dict[str, str],
) -> dict[str, object]:
    from lilo.inference.sampling import sample_task

    arm_offset = 0 if use_affinity else 200
    prompt = [100 + arm_offset + trajectory]
    prompt.extend(
        2000 + ((arm_offset * 31 + trajectory * 997 + index) % 5000)
        for index in range(initial_prompt_tokens - 1)
    )
    hits = []
    eligible = []
    prompt_lengths = []
    request_durations = []

    for turn in range(turns):
        expected_reusable = 0 if turn == 0 else len(prompt) - 1
        payload = {
            "num_samples": 1,
            "prompt": {
                "chunks": [
                    {
                        "type": "encoded_text",
                        "tokens": list(prompt),
                    }
                ]
            },
            "sampling_params": {
                "max_tokens": output_tokens,
                "temperature": 0.0,
                "seed": 1234 + trajectory * turns + turn,
            },
        }
        if use_affinity:
            payload["cache_affinity_key"] = f"trajectory-{trajectory}"
        started = time.perf_counter()
        result = await sample_task(
            {
                "request_id": f"modal-ablation:{arm}:{trajectory}:{turn}",
                "sampling_session_id": "modal-validation-sampler",
                "model_id": None,
                "publish_version": None,
                "latest": False,
                "payload": payload,
            },
            gateway,
            context_length=4096,
            headers=headers,
        )
        request_duration = time.perf_counter() - started
        sequence = result["sequences"][0]
        generated = list(sequence["tokens"])
        cached_tokens = int(result["prompt_cache_hit_tokens"])
        record = {
            "arm": arm,
            "trajectory": trajectory,
            "turn": turn,
            "prompt_tokens": len(prompt),
            "expected_reusable_tokens": expected_reusable,
            "prompt_cache_hit_tokens": cached_tokens,
            "input_cache_hit_rate": cached_tokens / len(prompt),
            "request_latency_ms": request_duration * 1000,
            "output_tokens": len(generated),
        }
        print(json.dumps(record, sort_keys=True), flush=True)
        hits.append(cached_tokens)
        eligible.append(expected_reusable)
        prompt_lengths.append(len(prompt))
        request_durations.append(request_duration)
        prompt.extend(generated)
        prompt.append(1200 + trajectory * turns + turn)

    eligible_hits = sum(
        min(hit, reusable)
        for hit, reusable in zip(hits[1:], eligible[1:], strict=True)
    )
    eligible_tokens = sum(eligible[1:])
    full_hit_turns = sum(
        hit >= reusable
        for hit, reusable in zip(hits[1:], eligible[1:], strict=True)
    )
    return {
        "arm": arm,
        "trajectory": trajectory,
        "turns": turns,
        "reusable_prefix_tokens": eligible_tokens,
        "reusable_cache_hit_tokens": eligible_hits,
        "reusable_prefix_hit_rate": eligible_hits / eligible_tokens,
        "all_prompt_tokens": sum(prompt_lengths),
        "all_prompt_cache_hit_tokens": sum(hits),
        "overall_input_cache_hit_rate": sum(hits) / sum(prompt_lengths),
        "full_hit_turns": full_hit_turns,
        "request_durations_seconds": request_durations,
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _aggregate(arm: str, trajectories: list[dict]) -> dict:
    eligible_tokens = sum(item["reusable_prefix_tokens"] for item in trajectories)
    hit_tokens = sum(item["reusable_cache_hit_tokens"] for item in trajectories)
    all_prompt_tokens = sum(item["all_prompt_tokens"] for item in trajectories)
    all_hit_tokens = sum(
        item["all_prompt_cache_hit_tokens"] for item in trajectories
    )
    request_durations = [
        duration
        for item in trajectories
        for duration in item["request_durations_seconds"]
    ]
    warm_request_durations = [
        duration
        for item in trajectories
        for duration in item["request_durations_seconds"][1:]
    ]
    return {
        "arm": arm,
        "trajectories": len(trajectories),
        "reusable_prefix_tokens": eligible_tokens,
        "reusable_cache_hit_tokens": hit_tokens,
        "reusable_prefix_hit_rate": hit_tokens / eligible_tokens,
        "all_prompt_tokens": all_prompt_tokens,
        "all_prompt_cache_hit_tokens": all_hit_tokens,
        "overall_input_cache_hit_rate": all_hit_tokens / all_prompt_tokens,
        "full_hit_turns": sum(item["full_hit_turns"] for item in trajectories),
        "measured_turns": sum(item["turns"] - 1 for item in trajectories),
        "mean_request_latency_ms": 1000
        * sum(request_durations)
        / len(request_durations),
        "p50_request_latency_ms": 1000 * _percentile(request_durations, 0.5),
        "p95_request_latency_ms": 1000 * _percentile(request_durations, 0.95),
        "mean_warm_turn_latency_ms": 1000
        * sum(warm_request_durations)
        / len(warm_request_durations),
    }


async def _validate(
    gateway: str,
    *,
    trajectories: int,
    turns: int,
    initial_prompt_tokens: int,
    output_tokens: int,
    minimum_hit_rate: float,
    minimum_lift: float,
    headers: dict[str, str],
) -> None:
    replicas = await _wait_for_pool(gateway, headers)
    results_by_arm = {"no_affinity": [], "affinity": []}
    arm_configs = (("no_affinity", False), ("affinity", True))
    for trajectory in range(trajectories):
        ordered_arms = (
            arm_configs if trajectory % 2 == 0 else tuple(reversed(arm_configs))
        )
        for arm, use_affinity in ordered_arms:
            results_by_arm[arm].append(
                await _run_trajectory(
                    gateway,
                    arm=arm,
                    trajectory=trajectory,
                    use_affinity=use_affinity,
                    turns=turns,
                    initial_prompt_tokens=initial_prompt_tokens,
                    output_tokens=output_tokens,
                    headers=headers,
                )
            )
    arms = {
        arm: _aggregate(arm, results)
        for arm, results in results_by_arm.items()
    }

    baseline_rate = arms["no_affinity"]["reusable_prefix_hit_rate"]
    affinity_rate = arms["affinity"]["reusable_prefix_hit_rate"]
    baseline_latency = arms["no_affinity"]["mean_warm_turn_latency_ms"]
    affinity_latency = arms["affinity"]["mean_warm_turn_latency_ms"]
    summary = {
        "replicas": sorted(replicas),
        "arms": arms,
        "absolute_reusable_prefix_hit_rate_lift": affinity_rate - baseline_rate,
        "absolute_overall_input_cache_hit_rate_lift": (
            arms["affinity"]["overall_input_cache_hit_rate"]
            - arms["no_affinity"]["overall_input_cache_hit_rate"]
        ),
        "warm_turn_latency_speedup": baseline_latency / affinity_latency,
        "warm_turn_latency_reduction": 1 - affinity_latency / baseline_latency,
        "minimum_required_affinity_reusable_prefix_hit_rate": minimum_hit_rate,
        "minimum_required_lift": minimum_lift,
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if affinity_rate < minimum_hit_rate:
        raise RuntimeError(
            f"affinity reusable-prefix hit rate {affinity_rate:.3f} "
            f"is below {minimum_hit_rate:.3f}"
        )
    if affinity_rate - baseline_rate < minimum_lift:
        raise RuntimeError(
            f"affinity lift {affinity_rate - baseline_rate:.3f} "
            f"is below {minimum_lift:.3f}"
        )


@app.function(
    image=client_image,
    secrets=[proxy_secret],
    timeout=TIMEOUT,
)
async def validate_remote(
    gateway: str,
    trajectories: int,
    turns: int,
    initial_prompt_tokens: int,
    output_tokens: int,
    minimum_hit_rate: float,
    minimum_lift: float,
) -> None:
    await _validate(
        gateway,
        trajectories=trajectories,
        turns=turns,
        initial_prompt_tokens=initial_prompt_tokens,
        output_tokens=output_tokens,
        minimum_hit_rate=minimum_hit_rate,
        minimum_lift=minimum_lift,
        headers={
            "Modal-Key": os.environ["MODAL_PROXY_TOKEN_ID"],
            "Modal-Secret": os.environ["MODAL_PROXY_TOKEN_SECRET"],
        },
    )


@app.local_entrypoint()
async def main(
    trajectories: int = 8,
    turns: int = 4,
    initial_prompt_tokens: int = 512,
    output_tokens: int = 8,
    minimum_hit_rate: float = 0.8,
    minimum_lift: float = 0.1,
) -> None:
    gateway = await Server.get_url.aio()
    await validate_remote.remote.aio(
        gateway,
        trajectories,
        turns,
        initial_prompt_tokens,
        output_tokens,
        minimum_hit_rate,
        minimum_lift,
    )
