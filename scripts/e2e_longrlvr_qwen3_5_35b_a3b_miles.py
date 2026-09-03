# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "modal-training-gym @ git+https://github.com/modal-projects/training-gym.git@59ebbf71668d0326837fe4cc2111a1522fae151b",
# ]
# ///
"""Run 64K LongRLVR on Qwen3.5-35B-A3B with Miles."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from modal_training_gym import (
    DatasetConfig,
    MilesRecipe,
    Qwen3_6_35B,
    TrainConfig,
)

MODEL_NAME = "Qwen/Qwen3.5-35B-A3B"
DATASET_NAME = "Guanzheng/LongRLVR-Data"
CONTEXT_LENGTH = 65_536
MAX_GENERATION_TOKENS = 8_192
MAX_PROMPT_TOKENS = CONTEXT_LENGTH - MAX_GENERATION_TOKENS
GROUP_SIZE = 8
GROUPS_PER_BATCH = 16
SOURCE_GROUP_MULTIPLIER = 1.5
LEARNING_RATE = 1e-6
SEED = 0

_SECTION_PATTERNS = {
    "answer": re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE),
    "useful_chunks": re.compile(
        r"<useful_chunks>(.*?)</useful_chunks>",
        re.DOTALL | re.IGNORECASE,
    ),
}
_CHUNK_PATTERN = re.compile(r"<CHUNK_(\d+)>", re.IGNORECASE)
_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
_CONCISE_INSTRUCTION = (
    "Keep the reasoning concise and reserve enough tokens to always emit "
    "<useful_chunks>...</useful_chunks> and <answer>...</answer>."
)


def _single_section(response: str, name: str) -> str | None:
    matches = _SECTION_PATTERNS[name].findall(response)
    return matches[0].strip() if len(matches) == 1 else None


def _token_f1(candidate: str, reference: str) -> float:
    candidate_tokens = Counter(_TOKEN_PATTERN.findall(candidate.casefold()))
    reference_tokens = Counter(_TOKEN_PATTERN.findall(reference.casefold()))
    if not candidate_tokens or not reference_tokens:
        return 0.0
    overlap = sum((candidate_tokens & reference_tokens).values())
    if not overlap:
        return 0.0
    precision = overlap / sum(candidate_tokens.values())
    recall = overlap / sum(reference_tokens.values())
    return 2 * precision * recall / (precision + recall)


def _concise_prompt(prompt: list[dict[str, str]]) -> list[dict[str, str]]:
    messages = [dict(message) for message in prompt]
    if not messages:
        return messages
    content = messages[-1].get("content")
    if not isinstance(content, str) or _CONCISE_INSTRUCTION in content:
        return messages
    marker = "\n\nDocument:"
    messages[-1]["content"] = (
        content.replace(marker, f"\n{_CONCISE_INSTRUCTION}{marker}", 1)
        if marker in content
        else f"{_CONCISE_INSTRUCTION}\n\n{content}"
    )
    return messages


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
    chunk_f2 = (
        5 * chunk_precision * chunk_recall / denominator if denominator else 0.0
    )
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


class LongRLVRDataset(DatasetConfig):
    dataset_id = "longrlvr-qwen35-64k"
    input_key = "prompt"
    label_key = "label"
    output_format = "parquet"
    apply_chat_template = True
    always_prepare = True
    writes_eval_paths = False

    def __init__(
        self,
        *,
        num_prompts: int,
        seed: int = SEED,
        max_prompt_tokens: int = MAX_PROMPT_TOKENS,
    ) -> None:
        self.num_prompts = num_prompts
        self.seed = seed
        self.max_prompt_tokens = max_prompt_tokens
        super().__init__()

    def prepare(
        self,
        path: str,
        eval_paths: dict[str, str] | None = None,
    ) -> None:
        del eval_paths
        from datasets import Dataset, load_dataset
        from transformers import AutoTokenizer

        dataset = load_dataset(
            DATASET_NAME,
            split="train",
            streaming=True,
        ).shuffle(
            seed=self.seed,
            buffer_size=max(256, 4 * self.num_prompts),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
        )
        rows: list[dict[str, Any]] = []
        seen_questions: set[str] = set()
        for row in dataset:
            prompt = row.get("prompt")
            reward_model = row.get("reward_model") or {}
            extra_info = row.get("extra_info") or {}
            question = str(extra_info.get("question") or "")
            ground_truth = str(reward_model.get("ground_truth") or "")
            reference_chunks = extra_info.get("ref_chunks") or []
            if (
                not isinstance(prompt, list)
                or not question
                or question in seen_questions
                or not ground_truth
                or not reference_chunks
            ):
                continue
            prompt = _concise_prompt(prompt)
            prompt_tokens = tokenizer.apply_chat_template(
                prompt,
                tokenize=True,
                add_generation_prompt=True,
            )
            if len(prompt_tokens) > self.max_prompt_tokens:
                continue
            seen_questions.add(question)
            rows.append(
                {
                    "prompt": prompt,
                    "label": json.dumps(
                        {
                            "question": question,
                            "ground_truth": ground_truth,
                            "ref_chunks": [
                                int(chunk) for chunk in reference_chunks
                            ],
                        },
                        ensure_ascii=False,
                    ),
                }
            )
            if len(rows) == self.num_prompts:
                break
        if len(rows) != self.num_prompts:
            raise RuntimeError(
                f"found only {len(rows)} usable prompts; expected {self.num_prompts}"
            )
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        Dataset.from_list(rows).to_parquet(destination)


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


def build_config(*, steps: int, seed: int = SEED) -> TrainConfig:
    source_groups_per_batch = int(GROUPS_PER_BATCH * SOURCE_GROUP_MULTIPLIER)
    dataset = LongRLVRDataset(
        num_prompts=steps * source_groups_per_batch,
        seed=seed,
    )
    recipe = MilesRecipe(
        name="miles-longrlvr-qwen35",
        gpu_type="H200",
        miles_model_name="qwen3.5-35B-A3B",
        async_mode=True,
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
        rollout_num_gpus=8,
        colocate=False,
        ref_load="/checkpoints/Qwen3.5-35B-A3B_torch_dist",
        megatron_to_hf_mode="",
        conversion_tensor_model_parallel_size=2,
        conversion_pipeline_model_parallel_size=1,
        conversion_expert_model_parallel_size=2,
        conversion_expert_tensor_parallel_size=1,
        num_rollout=steps,
        rollout_batch_size=GROUPS_PER_BATCH,
        n_samples_per_prompt=GROUP_SIZE,
        rollout_max_response_len=MAX_GENERATION_TOKENS,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        global_batch_size=GROUPS_PER_BATCH * GROUP_SIZE,
        over_sampling_batch_size=source_groups_per_batch,
        dynamic_sampling_filter_path=(
            "miles.rollout.filter_hub.dynamic_sampling_filters."
            "check_reward_nonzero_std"
        ),
        balance_data=True,
        custom_rm_function=longrlvr_reward,
        lr=LEARNING_RATE,
        lr_decay_style="constant",
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        optimizer="adam",
        use_tis=True,
        advantage_estimator="grpo",
        use_kl_loss=False,
        kl_loss_coef=0.0,
        kl_coef=0.0,
        entropy_coef=0.0,
        eps_clip=0.2,
        eps_clip_high=0.28,
        calculate_per_token_loss=True,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        context_parallel_size=2,
        expert_model_parallel_size=8,
        expert_tensor_parallel_size=1,
        sequence_parallel=True,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=32_768,
        optimizer_cpu_offload=True,
        overlap_cpu_optimizer_d2h_h2d=True,
        use_precision_aware_optimizer=True,
        rollout_num_gpus_per_engine=8,
        sglang_ep_size=8,
        sglang_mem_fraction_static=0.7,
        sglang_max_running_requests=32,
        sglang_cuda_graph_bs=[1, 2, 4, 8, 16, 24, 32],
        sglang_reasoning_parser="qwen3",
        attention_backend="flash",
        apply_chat_template_kwargs={"enable_thinking": True},
        update_weight_buffer_size=2 * 1024**3,
        use_fault_tolerance=True,
        rollout_health_check_first_wait=1_800,
        save_interval=steps,
        capture_trace=True,
        trace_sample_limit=16,
        environment={
            "PYTHONPATH": "/root:/root/Megatron-LM:/root/miles",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
        },
        extra_config={
            "fully_async": True,
            "max_seq_len": CONTEXT_LENGTH,
            "max_weight_staleness": 1,
            "pause_generation_mode": "in_place",
            "update_weight_transfer_mode": "broadcast",
            "rollout_max_context_len": CONTEXT_LENGTH,
            "rollout_max_prompt_len": MAX_PROMPT_TOKENS,
            "log_probs_chunk_size": 4_096,
        },
    )
    model = Qwen3_6_35B(model_name=MODEL_NAME)
    return TrainConfig(model=model, dataset=dataset, recipe=recipe)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 64K LongRLVR with Miles."
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")

    config = build_config(steps=args.steps, seed=args.seed)
    if args.dry_run:
        print("Miles LongRLVR configuration validated")
        print(config.recipe)
        return
    run = config.launch(prepare_inputs=True)
    print(f"Training run: {run.training_run_id}")
    print(f"Modal app: {run.modal_app_url}")
    print(f"Function call: {run.function_call_id}")


if __name__ == "__main__":
    main()
