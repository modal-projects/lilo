# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "modal-training-gym @ git+https://github.com/modal-projects/training-gym.git@dda10c3b27bc85839ae9b6d1a006595cf11e5953",
# ]
# ///
"""Arm B: Miles 9B with calculate_per_token_loss propagated to the model config.

Reproduces slim-hill-371ae1cfcb8e (Qwen3.5-9B LoRA, 16k ctx, 360 pinned
LongRLVR prompts) on a patched Miles checkout (/home/ubuntu/repos/miles,
branch ``graddiff-pertoken``) so the Bridge provider gets
``provider.calculate_per_token_loss = args.calculate_per_token_loss``.

Delta vs slim-hill: ``docker_image=radixark/miles:dev-202609050049`` —
slim-hill's recorded image is unknown; the default dev-202608120325 ships
sglang without ``--gated-launch-port``, so the ft rollout-cell gates
(``http://<host>:20276``…) never bound and blocked rollout startup.
dev-202609050049 ships sglang 0.5.19.dev56+ga359142 with gated launch.
FT stays enabled as in slim-hill.

Run: MODAL_SERVER_URL=https://api.modal.com MODAL_PROFILE= \
    MODAL_ENVIRONMENT=micah-dev uv run scripts/graddiff/e2e_miles_pertoken.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from modal_training_gym import (
    DatasetConfig,
    MilesRecipe,
    Qwen3_5_9B,
    TrainConfig,
    WandbConfig,
)

MODEL_NAME = "Qwen/Qwen3.5-9B"
CONTEXT_LENGTH = 16_384
MAX_GENERATION_TOKENS = 4_096
MAX_PROMPT_TOKENS = CONTEXT_LENGTH - MAX_GENERATION_TOKENS

# pinned_prompts.json is copied into the image as /tmp/pinned_prompts.json via
# MilesRecipe.patch_files; write()/rows() run inside that image. Storing the
# payload on the dataset object would bloat the cloudpickled function past
# Modal's 64KiB serialized-function limit.
PINNED_PROMPTS_LOCAL = "/home/ubuntu/work/modal_dl/pinned_prompts.json"
PINNED_PROMPTS_REMOTE = "/tmp/pinned_prompts.json"
PINNED_PROMPTS_SHA256 = (
    "4a54952d6ccb807ea0d74b05153ce275fd683cf3b43dac5abd7f280d97a9008f"
)
NUM_PROMPTS = 360

LOCAL_MILES = "/home/ubuntu/repos/miles"
DOCKER_IMAGE = "radixark/miles:dev-202609050049"

_SECTION_PATTERNS = {
    "answer": re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE),
    "useful_chunks": re.compile(
        r"<useful_chunks>(.*?)</useful_chunks>",
        re.DOTALL | re.IGNORECASE,
    ),
}
_CHUNK_PATTERN = re.compile(r"\d+")


def _single_section(response: str, name: str) -> str | None:
    matches = _SECTION_PATTERNS[name].findall(response)
    return matches[0].strip() if len(matches) == 1 else None


def _token_f1(candidate: str, reference: str) -> float:
    cand = Counter(candidate.split())
    ref = Counter(reference.split())
    overlap = sum((cand & ref).values())
    if overlap == 0:
        return 0.0
    if sum(cand.values()) == 0 or sum(ref.values()) == 0:
        return 0.0
    precision = overlap / sum(cand.values())
    recall = overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def _score_response(
    response: str,
    *,
    ground_truth: str,
    reference_chunks: Sequence[int],
) -> dict[str, float]:
    answer = _single_section(response, "answer")
    useful_chunks = _single_section(response, "useful_chunks")
    if answer is None or useful_chunks is None:
        return {
            "reward": 0.0,
            "answer_f1": 0.0,
            "chunk_precision": 0.0,
            "chunk_recall": 0.0,
            "chunk_f2": 0.0,
            "format": 0.0,
        }

    predicted = {int(chunk) for chunk in _CHUNK_PATTERN.findall(useful_chunks)}
    reference = set(reference_chunks)
    overlap = len(predicted & reference)
    chunk_precision = overlap / len(predicted) if predicted else 0.0
    chunk_recall = overlap / len(reference) if reference else 0.0
    denominator = 4 * chunk_precision + chunk_recall
    chunk_f2 = 5 * chunk_precision * chunk_recall / denominator if denominator else 0.0
    answer_f1 = _token_f1(answer, ground_truth)
    reward = answer_f1 + 0.1 * chunk_f2 + 0.9 * answer_f1 * chunk_f2
    return {
        "reward": reward,
        "answer_f1": answer_f1,
        "chunk_precision": chunk_precision,
        "chunk_recall": chunk_recall,
        "chunk_f2": chunk_f2,
        "format": 1.0,
    }


def _reward_one(sample: Any) -> float:
    try:
        label = json.loads(str(sample.label))
        result = _score_response(
            sample.response,
            ground_truth=str(label["ground_truth"]),
            reference_chunks=label["ref_chunks"],
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        result = {
            "reward": 0.0,
            "answer_f1": 0.0,
            "chunk_precision": 0.0,
            "chunk_recall": 0.0,
            "chunk_f2": 0.0,
            "format": 0.0,
        }
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    metadata["longrlvr"] = result
    sample.metadata = metadata
    return float(result["reward"])


async def longrlvr_reward(args, samples, **kwargs):
    del args, kwargs
    if isinstance(samples, list):
        return [_reward_one(sample) for sample in samples]
    return _reward_one(samples)


class PinnedPromptDataset(DatasetConfig):
    def __init__(self, remote_path: str = PINNED_PROMPTS_REMOTE) -> None:
        self._remote_path = remote_path

    def input_key(self) -> str:
        return "prompt"

    def label_key(self) -> str:
        return "label"

    def output_format(self) -> str:
        return "parquet"

    def apply_chat_template(self) -> bool:
        return True

    def rows(self):
        data = Path(self._remote_path).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != PINNED_PROMPTS_SHA256:
            raise RuntimeError(
                f"pinned_prompts.json sha256 {digest} != expected "
                f"{PINNED_PROMPTS_SHA256}"
            )
        rows_in = json.loads(data)["rows"]
        if len(rows_in) != NUM_PROMPTS:
            raise RuntimeError(
                f"expected {NUM_PROMPTS} pinned prompts, got {len(rows_in)}"
            )
        for row in rows_in:
            yield {
                "prompt": row["prompt"],
                "label": json.dumps(
                    {
                        "question": row["question"],
                        "ground_truth": row["ground_truth"],
                        "ref_chunks": [int(c) for c in row["ref_chunks"]],
                    },
                    ensure_ascii=False,
                ),
            }

    def write(self, path: str) -> None:
        from datasets import Dataset

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        Dataset.from_list(list(self.rows())).to_parquet(destination)


def build_config(*, steps: int) -> TrainConfig:
    dataset = PinnedPromptDataset()
    recipe = MilesRecipe(
        name="miles-9b-16k-pertoken",
        gpu_type="H100",
        docker_image=DOCKER_IMAGE,
        local_miles=LOCAL_MILES,
        patch_files=[PINNED_PROMPTS_LOCAL],
        image_run_commands=[
            (
                "cd /root/Megatron-LM && git fetch --depth 200 origin"
                " miles-main && git checkout -f 73b54618f"
            ),
            (
                "pip install --no-deps"
                " 'git+https://github.com/radixark/Megatron-Bridge.git"
                "@40b930897717941cfe2bd9806f417e28bf1bfa65'"
            ),
        ],
        miles_model_name="qwen3.5-9B",
        async_mode=True,
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=1,
        colocate=False,
        megatron_to_hf_mode="bridge",
        num_rollout=steps,
        rollout_batch_size=16,
        n_samples_per_prompt=8,
        rollout_max_response_len=MAX_GENERATION_TOKENS,
        rollout_temperature=1.0,
        rollout_shuffle=True,
        rollout_top_p=1.0,
        global_batch_size=128,
        over_sampling_batch_size=24,
        dynamic_sampling_filter_path=(
            "miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std"
        ),
        balance_data=True,
        custom_rm_function=longrlvr_reward,
        lr=1e-4,
        lr_decay_style="constant",
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        optimizer="adam",
        use_distributed_optimizer=True,
        optimizer_cpu_offload=False,
        use_precision_aware_optimizer=False,
        use_tis=True,
        advantage_estimator="grpo",
        use_kl_loss=False,
        kl_loss_coef=0.0,
        kl_coef=0.0,
        entropy_coef=0.0,
        eps_clip=0.2,
        eps_clip_high=0.28,
        calculate_per_token_loss=True,
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        sequence_parallel=True,
        lora_rank=32,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=(
            "language_model.decoder.layers.*.self_attention.linear_qkv,"
            "language_model.decoder.layers.*.self_attention.linear_proj,"
            "language_model.decoder.layers.*.mlp.linear_fc1,"
            "language_model.decoder.layers.*.mlp.linear_fc2"
        ),
        no_gradient_accumulation_fusion=True,
        attention_backend="flash",
        attention_softmax_in_fp32=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        accumulate_allreduce_grads_in_fp32=True,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        qkv_format="thd",
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=16_384,
        sglang_ep_size=1,
        sglang_mem_fraction_static=0.8,
        sglang_max_running_requests=32,
        sglang_cuda_graph_bs=[1, 2, 4, 8, 16, 24, 32],
        sglang_reasoning_parser="qwen3",
        sglang_lora_backend="triton",
        sglang_lora_use_virtual_experts=True,
        apply_chat_template_kwargs={"enable_thinking": False},
        update_weight_buffer_size=2 * 1024**3,
        use_fault_tolerance=True,
        rollout_health_check_first_wait=1_800,
        save="/checkpoints/miles-9b-16k-pertoken",
        save_interval=5,
        n_samples_per_eval_prompt=2,
        eval_max_response_len=4096,
        eval_top_p=1.0,
        environment={
            "PYTHONPATH": "/root:/root/Megatron-LM:/root/miles",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
        },
        image_env={
            # Miles' wandb_utils names runs after --wandb-group when the
            # random suffix is disabled; WANDB_NAME only applies if any
            # wandb.init path leaves name unset.
            "WANDB_NAME": "miles-9b-16k-pertoken-30",
        },
        metrics=WandbConfig(
            project="miles-lora-longcontext",
            group="graddiff-e2e",
            exp_name="miles-9b-16k-pertoken-30",
            disable_random_suffix=True,
        ),
        extra_config={
            "fully_async": True,
            "max_seq_len": CONTEXT_LENGTH,
            "max_weight_staleness": 1,
            "pause_generation_mode": "in_place",
            "update_weight_transfer_mode": "broadcast",
            "rollout_max_context_len": CONTEXT_LENGTH,
            "rollout_max_prompt_len": MAX_PROMPT_TOKENS,
            "rollout_seed": 42,
            "clip_grad": 1.0,
            "log_probs_chunk_size": 4_096,
            "sglang_max_lora_rank": 32,
            "sglang_max_loras_per_batch": 8,
        },
    )
    model = Qwen3_5_9B(model_name=MODEL_NAME)
    return TrainConfig(model=model, dataset=dataset, recipe=recipe)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Arm B: Miles 9B per-token-loss e2e run."
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")

    config = build_config(steps=args.steps)
    if args.dry_run:
        print("Miles per-token 9B configuration validated")
        print(config.recipe)
        return
    run = config.launch()
    print(f"Training run: {run.training_run_id}")
    print(f"Modal app: {run.modal_app_url}")
    print(f"Function call: {run.function_call_id}")


if __name__ == "__main__":
    main()
