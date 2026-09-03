from unittest.mock import Mock, patch

from lilo.inference.serving import (
    start_fft_sidecar,
    start_sglang,
    supervise_children,
)


def test_full_sglang_uses_cpu_weight_cache() -> None:
    with patch("subprocess.Popen") as popen:
        start_sglang(
            "/model",
            port=8001,
            context_length=1024,
            max_loras_per_batch=1,
            max_loaded_loras=1,
            max_lora_rank=1,
            max_running_requests=64,
            enable_lora=False,
            enable_cpu_weight_cache=True,
            cpu_weight_cache_max_compile_group_gb=16,
            schedule_policy="lpm",
        )
    command = popen.call_args.args[0]
    assert "--enable-lora" not in command
    assert "--enable-cpu-weight-cache" in command
    assert command[command.index("--cpu-weight-cache-max-compile-group-gb") + 1] == "16"
    assert command[command.index("--schedule-policy") + 1] == "lpm"


def test_sglang_supports_hybrid_moe_parallelism() -> None:
    with patch("subprocess.Popen") as popen:
        start_sglang(
            "/model",
            port=8001,
            context_length=65_536,
            max_loras_per_batch=1,
            max_loaded_loras=1,
            max_lora_rank=1,
            max_running_requests=64,
            tensor_parallel_size=1,
            expert_parallel_size=4,
            parallel_world_size=4,
            memory_fraction=0.9,
            enable_lora=False,
        )
    command = popen.call_args.args[0]
    assert command[command.index("--tp-size") + 1] == "4"
    assert command[command.index("--dp-size") + 1] == "4"
    assert command[command.index("--ep-size") + 1] == "4"
    assert command[command.index("--mem-fraction-static") + 1] == "0.9"
    assert "--enable-dp-attention" in command


def test_child_supervisor_terminates_sibling() -> None:
    exited = Mock()
    exited.poll.return_value = 1
    sibling = Mock()
    sibling.poll.return_value = None

    with patch("lilo.inference.serving.terminate") as terminate:
        supervisor = supervise_children(exited, sibling)
        supervisor.join(timeout=1)

    assert not supervisor.is_alive()
    terminate.assert_called_once_with(sibling)


def test_fft_sidecar_receives_store_scope() -> None:
    with patch("subprocess.Popen") as popen:
        start_fft_sidecar(
            port=8000,
            sglang_port=8001,
            model_path="/model",
            bulletin_root="/bulletin",
            bulletin_volume="bulletin",
            run_id="run-a",
            pinned_version=7,
        )
    command = popen.call_args.args[0]
    assert command[command.index("--base-checkpoint-dir") + 1] == "/model"
    assert command[command.index("--run-id") + 1] == "run-a"
    assert command[command.index("--pinned-version") + 1] == "7"
