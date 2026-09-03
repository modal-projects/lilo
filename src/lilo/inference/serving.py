from __future__ import annotations

import math
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Sequence


def start_sglang(
    model_path: str,
    *,
    port: int,
    context_length: int,
    max_loras_per_batch: int,
    max_loaded_loras: int,
    max_lora_rank: int,
    max_running_requests: int,
    max_queued_requests: int | None = None,
    tensor_parallel_size: int = 1,
    expert_parallel_size: int = 1,
    expert_tensor_parallel_size: int = 1,
    parallel_world_size: int | None = None,
    trust_remote_code: bool = False,
    lora_target_modules: Sequence[str] = ("all",),
    enable_lora: bool = True,
    enable_cpu_weight_cache: bool = False,
    cpu_weight_cache_max_compile_group_gb: float | None = None,
    memory_fraction: float = 0.85,
    schedule_policy: str = "fcfs",
) -> subprocess.Popen:
    if enable_lora and max_loaded_loras < max_loras_per_batch:
        raise ValueError("max_loaded_loras must be at least max_loras_per_batch")
    if tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be at least 1")
    if max_queued_requests is not None and max_queued_requests < 1:
        raise ValueError("max_queued_requests must be at least 1")
    if expert_parallel_size < 1:
        raise ValueError("expert_parallel_size must be at least 1")
    if expert_tensor_parallel_size < 1:
        raise ValueError("expert_tensor_parallel_size must be at least 1")
    if enable_lora and not lora_target_modules:
        raise ValueError("lora_target_modules must not be empty")
    if (
        cpu_weight_cache_max_compile_group_gb is not None
        and not enable_cpu_weight_cache
    ):
        raise ValueError(
            "cpu_weight_cache_max_compile_group_gb requires enable_cpu_weight_cache"
        )
    if not 0 < memory_fraction < 1:
        raise ValueError("memory_fraction must be between 0 and 1")
    moe_parallel_size = expert_parallel_size * expert_tensor_parallel_size
    world_size = parallel_world_size or math.lcm(
        tensor_parallel_size,
        moe_parallel_size,
    )
    if world_size < 1:
        raise ValueError("parallel_world_size must be at least 1")
    if world_size % tensor_parallel_size:
        raise ValueError(
            "parallel_world_size must be divisible by tensor_parallel_size"
        )
    if expert_parallel_size > 1 and world_size % moe_parallel_size:
        raise ValueError("parallel_world_size must be divisible by EP times expert TP")
    attention_data_parallel_size = world_size // tensor_parallel_size
    moe_data_parallel_size = (
        world_size // moe_parallel_size if expert_parallel_size > 1 else 1
    )
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tp-size",
        str(world_size),
    ]
    if attention_data_parallel_size > 1:
        command.extend(["--dp-size", str(attention_data_parallel_size)])
        command.append("--enable-dp-attention")
    if expert_parallel_size > 1:
        command.extend(["--ep-size", str(expert_parallel_size)])
    if moe_data_parallel_size > 1:
        command.extend(["--moe-dp-size", str(moe_data_parallel_size)])
    if trust_remote_code:
        command.append("--trust-remote-code")
    if enable_cpu_weight_cache:
        command.append("--enable-cpu-weight-cache")
    if cpu_weight_cache_max_compile_group_gb is not None:
        command.extend(
            [
                "--cpu-weight-cache-max-compile-group-gb",
                str(cpu_weight_cache_max_compile_group_gb),
            ]
        )
    command.extend(
        [
            "--context-length",
            str(context_length),
            "--mem-fraction-static",
            str(memory_fraction),
            "--weight-loader-disable-mmap",
        ]
    )
    if enable_lora:
        command.extend(
            [
                "--enable-lora",
                "--max-loras-per-batch",
                str(max_loras_per_batch),
                "--max-loaded-loras",
                str(max_loaded_loras),
                "--max-lora-rank",
                str(max_lora_rank),
                "--lora-target-modules",
                *lora_target_modules,
            ]
        )
    command.extend(
        [
            "--max-running-requests",
            str(max_running_requests),
            "--schedule-policy",
            schedule_policy,
        ]
    )
    if max_queued_requests is not None:
        command.extend(
            [
                "--max-queued-requests",
                str(max_queued_requests),
            ]
        )
    return subprocess.Popen(command, start_new_session=True)


def start_fft_sidecar(
    *,
    port: int,
    sglang_port: int,
    model_path: str,
    bulletin_root: str,
    bulletin_volume: str,
    run_id: str,
    pinned_version: int | None = None,
) -> subprocess.Popen:
    command = [
        sys.executable,
        "-m",
        "lilo.inference.fft_sidecar",
        "--port",
        str(port),
        "--upstream-url",
        f"http://127.0.0.1:{sglang_port}",
        "--base-checkpoint-dir",
        model_path,
        "--bulletin-root",
        bulletin_root,
        "--bulletin-volume",
        bulletin_volume,
        "--run-id",
        run_id,
    ]
    if pinned_version is not None:
        command.extend(
            [
                "--pinned-version",
                str(pinned_version),
            ]
        )
    return subprocess.Popen(command, start_new_session=True)


def wait_http(url: str, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    error: BaseException | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"process exited with code {process.returncode}: {url}")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status < 500:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            error = exc
        time.sleep(2)
    raise TimeoutError(f"{url}: {error}")


def terminate(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)


def supervise_children(
    first: subprocess.Popen,
    second: subprocess.Popen,
) -> threading.Thread:
    """Terminate the sibling when either serving process exits."""

    def monitor() -> None:
        while True:
            if first.poll() is not None:
                terminate(second)
                return
            if second.poll() is not None:
                terminate(first)
                return
            time.sleep(1)

    thread = threading.Thread(
        target=monitor,
        name="tinker-rollout-child-supervisor",
        daemon=True,
    )
    thread.start()
    return thread
