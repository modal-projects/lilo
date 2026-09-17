# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "modal-training-gym @ git+https://github.com/modal-projects/training-gym.git@bdfd80d3bcd726953899a73088cd9fc0bd98939f",
# ]
# ///
"""LongRLVR, Qwen3.8-27B LoRA (rank 32), Miles via training-gym.

Context-length ladder (16k -> 64k -> 128k -> 256k) for the Miles + Modal
baseline. Everything except model, context length, prompt cap and topology is
identical to the Qwen3.5-9B 16k launcher.

Examples:
    uv run scripts/e2e_longrlvr_qwen3_8_27b_miles_lora.py --context-k 16 --steps 1
    uv run scripts/e2e_longrlvr_qwen3_8_27b_miles_lora.py --context-k 64 --steps 5 --cp 2
    uv run scripts/e2e_longrlvr_qwen3_8_27b_miles_lora.py --context-k 256 --steps 5 --actor-nodes 2 --tp 2 --cp 8 --pad-to-tokens 250000
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from modal_training_gym import (
    DatasetConfig,
    MilesRecipe,
    Qwen3_8_27B,
    TrainConfig,
    WandbConfig,
)

MODEL_NAME = "Qwen/Qwen3.8-27B"
MILES_MODEL_NAME = "qwen3.8-27B"
MILES_IMAGE = "radixark/miles:dev-202609151226"
DATASET_NAME = "Guanzheng/LongRLVR-Data"
MAX_GENERATION_TOKENS = 4_096
WANDB_PROJECT = "miles-lora-longcontext"
WANDB_GROUP = "baseline-miles-27b"
# Anchored below decoder.layers so the adapter stays off the vision tower / MTP
# block; covers softmax attention, MLP, and the GDN in_proj/out_proj layers.
_LAYERS = "language_model.decoder.layers.*"
TARGET_MODULES = ",".join(
    [
        f"{_LAYERS}.self_attention.linear_qkv",
        f"{_LAYERS}.self_attention.linear_proj",
        f"{_LAYERS}.mlp.linear_fc1",
        f"{_LAYERS}.mlp.linear_fc2",
        f"{_LAYERS}.self_attention.in_proj",
        f"{_LAYERS}.self_attention.out_proj",
    ]
)
GROUP_SIZE = 8
GROUPS_PER_BATCH = 16
SOURCE_GROUP_MULTIPLIER = 1.5
LEARNING_RATE = 1e-4
SEED = 0

_SECTION_PATTERNS = {
    "answer": re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE),
    "useful_chunks": re.compile(
        r"<useful_chunks>(.*?)</useful_chunks>",
        re.DOTALL | re.IGNORECASE,
    ),
}
_CHUNK_PATTERN = re.compile(r"<CHUNK_(\d+)>", re.IGNORECASE)
_CHUNK_BLOCK = re.compile(r"<CHUNK_(\d+)>(.*?)</CHUNK_\1>", re.DOTALL)
_DOC_MARKER = "Document:"
_QUESTION_MARKER = "\n\nQuestion:"
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


def _split_document(content: str) -> tuple[str, list[str], str] | None:
    """Return (head, chunk_bodies, tail) or None if the prompt isn't in the LongRLVR shape."""
    doc_start = content.find(_DOC_MARKER)
    q_start = content.rfind(_QUESTION_MARKER)
    if doc_start < 0 or q_start < 0 or q_start < doc_start:
        return None
    head = content[: doc_start + len(_DOC_MARKER)]
    body = content[doc_start + len(_DOC_MARKER) : q_start]
    tail = content[q_start:]
    blocks = _CHUNK_BLOCK.findall(body)
    if not blocks or [int(i) for i, _ in blocks] != list(range(len(blocks))):
        return None
    return head, [b for _, b in blocks], tail


def _render_document(head: str, chunks: Sequence[str], tail: str) -> str:
    body = "\n".join(f"<CHUNK_{i}>{c}</CHUNK_{i}>" for i, c in enumerate(chunks))
    return f"{head}\n {body}\n{tail}"


def _pad_with_distractors(
    chunks: list[str],
    ref_chunks: list[int],
    pool: Sequence[list[str]],
    *,
    target_chars: int,
    rng: random.Random,
) -> tuple[list[str], list[int]]:
    """Interleave whole distractor documents (from `pool`) around the real document until
    total chars >= target_chars. The real doc stays contiguous and in order; ref_chunks are
    renumbered to their new positions. Deterministic given rng."""
    real_chars = sum(len(c) for c in chunks)
    docs: list[list[str]] = []
    total = real_chars
    order = list(range(len(pool)))
    rng.shuffle(order)
    for idx in order:
        if total >= target_chars:
            break
        doc = pool[idx]
        need = target_chars - total
        doc_chars = sum(len(c) for c in doc)
        if doc_chars > need:
            take: list[str] = []
            acc = 0
            for c in doc:
                if acc >= need:
                    break
                take.append(c)
                acc += len(c)
            doc = take
            doc_chars = acc
        if doc:
            docs.append(doc)
            total += doc_chars
    insert_at = rng.randint(0, len(docs))
    docs.insert(insert_at, chunks)
    offset = sum(len(d) for d in docs[:insert_at])
    merged = [c for d in docs for c in d]
    return merged, [offset + r for r in ref_chunks]


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


class LongRLVRDataset(DatasetConfig):
    def __init__(
        self,
        *,
        num_prompts: int,
        max_prompt_tokens: int,
        min_prompt_tokens: int = 0,
        seed: int = SEED,
        pad_to_tokens: int | None = None,
    ) -> None:
        self.num_prompts = num_prompts
        self.seed = seed
        self.max_prompt_tokens = max_prompt_tokens
        self.min_prompt_tokens = min_prompt_tokens
        self.pad_to_tokens = pad_to_tokens
        super().__init__()

    def input_key(self) -> str:
        return "prompt"

    def label_key(self) -> str:
        return "label"

    def output_format(self) -> str:
        return "parquet"

    def apply_chat_template(self) -> bool:
        return True

    def cache_key(self) -> str | None:
        return None

    def rows(self) -> Iterable[dict[str, Any]]:
        raise NotImplementedError("LongRLVRDataset materializes via write()")

    def write(self, path: str) -> None:
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
        pool: list[list[str]] = []
        if self.pad_to_tokens is not None:
            pool_dataset = load_dataset(
                DATASET_NAME,
                split="train",
                streaming=True,
            ).shuffle(
                seed=self.seed,
                buffer_size=max(256, 4 * self.num_prompts),
            )
            for pool_row in itertools.islice(pool_dataset, 96):
                pool_prompt = pool_row.get("prompt")
                if not isinstance(pool_prompt, list) or not pool_prompt:
                    continue
                pool_prompt = _concise_prompt(pool_prompt)
                pool_content = pool_prompt[-1].get("content")
                if not isinstance(pool_content, str):
                    continue
                parsed_pool = _split_document(pool_content)
                if parsed_pool is not None:
                    pool.append(parsed_pool[1])
                    if len(pool) == 64:
                        break
        rows: list[dict[str, Any]] = []
        prompt_lengths: list[int] = []
        base_prompt_lengths: list[int] = []
        seen_questions: set[str] = set()
        scanned = 0
        too_long = 0
        too_short = 0
        unparsable = 0
        pad_failed = 0
        for row in dataset:
            scanned += 1
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
            if self.pad_to_tokens is None:
                # Cheap character-level prefilter; ~2% of LongRLVR prompts fit in
                # 12k tokens, so tokenizing every 40k-token prompt is the bottleneck.
                prompt_chars = sum(len(str(m.get("content", ""))) for m in prompt)
                if prompt_chars > 6 * self.max_prompt_tokens:
                    too_long += 1
                    continue
                prompt_text = tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)[
                    "input_ids"
                ]
                if len(prompt_tokens) > self.max_prompt_tokens:
                    too_long += 1
                    continue
                ref_ints = [int(chunk) for chunk in reference_chunks]
                label_extra: dict[str, Any] = {}
            else:
                ref_ints = [int(chunk) for chunk in reference_chunks]
                content = prompt[-1].get("content")
                if not isinstance(content, str):
                    unparsable += 1
                    continue
                parsed = _split_document(content)
                if parsed is None:
                    unparsable += 1
                    continue
                head, chunks, tail = parsed
                pool_for_row = [
                    distractors
                    for distractors in pool
                    if distractors is not chunks and distractors[:1] != chunks[:1]
                ]
                rng = random.Random(f"{self.seed}:{question}")
                base_text = tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                base_tokens = len(
                    tokenizer(base_text, add_special_tokens=False)["input_ids"]
                )
                if base_tokens > self.max_prompt_tokens:
                    too_long += 1
                    continue
                ratio = len(base_text) / max(base_tokens, 1)
                target_chars = int(self.pad_to_tokens * ratio)
                padded_chunks, new_refs = _pad_with_distractors(
                    chunks,
                    ref_ints,
                    pool_for_row,
                    target_chars=target_chars,
                    rng=rng,
                )
                prompt_tokens = []
                for _ in range(8):
                    padded_prompt = [dict(message) for message in prompt]
                    padded_prompt[-1]["content"] = _render_document(
                        head, padded_chunks, tail
                    )
                    prompt_text = tokenizer.apply_chat_template(
                        padded_prompt,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)[
                        "input_ids"
                    ]
                    if len(prompt_tokens) <= self.max_prompt_tokens:
                        prompt = padded_prompt
                        break
                    real_offset = new_refs[0] - ref_ints[0]
                    real_end = real_offset + len(chunks)
                    if real_end < len(padded_chunks):
                        padded_chunks.pop()
                    elif real_offset > 0:
                        padded_chunks.pop(0)
                        new_refs = [ref - 1 for ref in new_refs]
                    else:
                        break
                else:
                    prompt_tokens = []
                if len(prompt_tokens) > self.max_prompt_tokens or not prompt_tokens:
                    pad_failed += 1
                    continue
                label_extra = {
                    "base_prompt_tokens": base_tokens,
                }
            if len(prompt_tokens) < self.min_prompt_tokens:
                too_short += 1
                continue
            if self.pad_to_tokens is not None:
                base_prompt_lengths.append(base_tokens)
            seen_questions.add(question)
            prompt_lengths.append(len(prompt_tokens))
            rows.append(
                {
                    "prompt": prompt,
                    "label": json.dumps(
                        {
                            "question": question,
                            "ground_truth": ground_truth,
                            "ref_chunks": (
                                new_refs if self.pad_to_tokens is not None else ref_ints
                            ),
                            "prompt_tokens": len(prompt_tokens),
                            **label_extra,
                        },
                        ensure_ascii=False,
                    ),
                }
            )
            if len(rows) == self.num_prompts:
                break
        if len(rows) != self.num_prompts:
            raise RuntimeError(
                f"found only {len(rows)} usable prompts; expected {self.num_prompts} "
                f"(scanned {scanned}, too_long {too_long}, too_short {too_short}, "
                f"unparsable {unparsable}, pad_failed {pad_failed})"
            )
        quantiles = (
            statistics.quantiles(prompt_lengths, n=4) if len(prompt_lengths) > 1 else []
        )
        print(
            "LongRLVR prompt-length distribution "
            f"(cap {self.max_prompt_tokens}, floor {self.min_prompt_tokens}, "
            f"pad_to={self.pad_to_tokens}): "
            f"n={len(prompt_lengths)} scanned={scanned} too_long={too_long} "
            f"too_short={too_short} unparsable={unparsable} pad_failed={pad_failed} "
            f"min={min(prompt_lengths)} "
            f"mean={statistics.fmean(prompt_lengths):.0f} "
            f"quartiles={[round(q) for q in quantiles]} max={max(prompt_lengths)}"
            + (
                f" base_mean={statistics.fmean(base_prompt_lengths):.0f}"
                if self.pad_to_tokens is not None
                else ""
            ),
            flush=True,
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


class Topology:
    """Trainer/rollout topology for one context-length rung."""

    def __init__(
        self,
        *,
        gpu_type: str = "H200",
        actor_num_nodes: int = 1,
        tp: int = 4,
        cp: int = 1,
        rollout_num_gpus: int = 8,
        rollout_num_gpus_per_engine: int = 1,
        max_tokens_per_gpu: int | None = None,
        sglang_mem_fraction_static: float = 0.8,
        sglang_max_running_requests: int = 32,
    ) -> None:
        self.gpu_type = gpu_type
        self.actor_num_nodes = actor_num_nodes
        self.tp = tp
        self.cp = cp
        self.rollout_num_gpus = rollout_num_gpus
        self.rollout_num_gpus_per_engine = rollout_num_gpus_per_engine
        self.max_tokens_per_gpu = max_tokens_per_gpu
        self.sglang_mem_fraction_static = sglang_mem_fraction_static
        self.sglang_max_running_requests = sglang_max_running_requests

    @property
    def dp(self) -> int:
        return self.actor_num_nodes * 8 // (self.tp * self.cp)

    def describe(self) -> str:
        return (
            f"{self.actor_num_nodes}x{self.gpu_type}:8 trainer "
            f"TP{self.tp}xCP{self.cp}xDP{self.dp} + "
            f"{self.rollout_num_gpus} rollout GPUs "
            f"({self.rollout_num_gpus // self.rollout_num_gpus_per_engine} engines "
            f"x TP{self.rollout_num_gpus_per_engine})"
        )


def build_config(
    *,
    context_length: int,
    steps: int,
    topology: Topology,
    min_prompt_fraction: float = 0.0,
    pad_to_tokens: int | None = None,
    seed: int = SEED,
    use_wandb: bool = True,
    run_suffix: str = "",
) -> TrainConfig:
    max_prompt_tokens = context_length - MAX_GENERATION_TOKENS
    min_prompt_tokens = int(min_prompt_fraction * max_prompt_tokens)
    ctx_k = context_length // 1024
    source_groups_per_batch = int(GROUPS_PER_BATCH * SOURCE_GROUP_MULTIPLIER)
    dataset = LongRLVRDataset(
        num_prompts=steps * source_groups_per_batch,
        max_prompt_tokens=max_prompt_tokens,
        min_prompt_tokens=min_prompt_tokens,
        seed=seed,
        pad_to_tokens=pad_to_tokens,
    )
    # With CP a sequence is sharded across cp ranks, so the per-GPU token
    # budget only has to hold context_length / cp tokens.
    max_tokens_per_gpu = topology.max_tokens_per_gpu or context_length // topology.cp
    run_name = f"miles-27b-{ctx_k}k-{steps}{run_suffix}"
    recipe = MilesRecipe(
        name=f"miles-longrlvr-qwen38-27b-lora-{ctx_k}k",
        gpu_type=topology.gpu_type,
        docker_image=MILES_IMAGE,
        miles_model_name=MILES_MODEL_NAME,
        async_mode=True,
        actor_num_nodes=topology.actor_num_nodes,
        actor_num_gpus_per_node=8,
        rollout_num_gpus=topology.rollout_num_gpus,
        rollout_num_gpus_per_engine=topology.rollout_num_gpus_per_engine,
        colocate=False,
        megatron_to_hf_mode="bridge",
        lora_rank=32,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=TARGET_MODULES,
        no_gradient_accumulation_fusion=True,
        sglang_lora_backend="triton",
        num_rollout=steps,
        rollout_batch_size=GROUPS_PER_BATCH,
        n_samples_per_prompt=GROUP_SIZE,
        rollout_max_response_len=MAX_GENERATION_TOKENS,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        global_batch_size=GROUPS_PER_BATCH * GROUP_SIZE,
        over_sampling_batch_size=source_groups_per_batch,
        dynamic_sampling_filter_path=(
            "miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std"
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
        tensor_model_parallel_size=topology.tp,
        pipeline_model_parallel_size=1,
        context_parallel_size=topology.cp,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        sequence_parallel=True,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=max_tokens_per_gpu,
        sglang_mem_fraction_static=topology.sglang_mem_fraction_static,
        sglang_max_running_requests=topology.sglang_max_running_requests,
        sglang_cuda_graph_bs=[1, 2, 4, 8, 16, 24, 32],
        sglang_reasoning_parser="qwen3",
        attention_backend="flash",
        apply_chat_template_kwargs={"enable_thinking": False},
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
            # In-process inductor compile; the subprocess pool can hang TP ranks.
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "PYTHONFAULTHANDLER": "1",
        },
        extra_config={
            "fully_async": True,
            "max_seq_len": context_length,
            "max_weight_staleness": 1,
            "pause_generation_mode": "in_place",
            "update_weight_transfer_mode": "broadcast",
            "rollout_max_context_len": context_length,
            "rollout_max_prompt_len": max_prompt_tokens,
            "log_probs_chunk_size": 4_096,
            "sglang_max_lora_rank": 32,
        },
        metrics=WandbConfig(
            project=WANDB_PROJECT,
            entity="modal-labs",
            group=WANDB_GROUP,
            exp_name=run_name,
            modal_wandb_secret_name="wandb-secret",
        )
        if use_wandb
        else None,
    )
    model = Qwen3_8_27B(model_name=MODEL_NAME)
    return TrainConfig(model=model, dataset=dataset, recipe=recipe)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LongRLVR Qwen3.8-27B LoRA with Miles at a given context length."
    )
    parser.add_argument("--context-k", type=int, default=16, choices=[16, 64, 128, 256])
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--gpu-type",
        default="H200",
        help="Modal GPU type for trainer AND rollout nodes. 'H200' is exact; plain 'H100' "
        "may be auto-upgraded to H200 by Modal, use 'H100!' to forbid that.",
    )
    parser.add_argument("--actor-nodes", type=int, default=1)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--rollout-gpus", type=int, default=8)
    parser.add_argument("--rollout-gpus-per-engine", type=int, default=1)
    parser.add_argument("--max-tokens-per-gpu", type=int, default=None)
    parser.add_argument("--sglang-mem-fraction", type=float, default=0.8)
    parser.add_argument("--sglang-max-running-requests", type=int, default=32)
    parser.add_argument(
        "--pad-to-tokens",
        type=int,
        default=None,
        help="Pad each prompt with distractor documents from other LongRLVR rows up to "
        "~this many tokens (deterministic given --seed). Must be <= prompt cap.",
    )
    parser.add_argument(
        "--min-prompt-fraction",
        type=float,
        default=None,
        help="Drop prompts shorter than this fraction of the prompt cap "
        "(default 0 at 16k, 0.5 at >=64k, or 0.9 when padding).",
    )
    parser.add_argument("--run-suffix", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if (args.actor_nodes * 8) % (args.tp * args.cp):
        parser.error("tp*cp must divide the trainer GPU count")
    if args.rollout_gpus % args.rollout_gpus_per_engine:
        parser.error("rollout-gpus must be a multiple of rollout-gpus-per-engine")

    context_length = args.context_k * 1024
    min_prompt_fraction = (
        args.min_prompt_fraction
        if args.min_prompt_fraction is not None
        else (
            0.9
            if args.pad_to_tokens is not None
            else (0.0 if args.context_k <= 16 else 0.5)
        )
    )
    max_prompt_tokens = context_length - MAX_GENERATION_TOKENS
    if args.pad_to_tokens is not None and args.pad_to_tokens > max_prompt_tokens:
        parser.error("--pad-to-tokens must be <= the prompt cap")
    topology = Topology(
        gpu_type=args.gpu_type,
        actor_num_nodes=args.actor_nodes,
        tp=args.tp,
        cp=args.cp,
        rollout_num_gpus=args.rollout_gpus,
        rollout_num_gpus_per_engine=args.rollout_gpus_per_engine,
        max_tokens_per_gpu=args.max_tokens_per_gpu,
        sglang_mem_fraction_static=args.sglang_mem_fraction,
        sglang_max_running_requests=args.sglang_max_running_requests,
    )
    config = build_config(
        context_length=context_length,
        steps=args.steps,
        topology=topology,
        min_prompt_fraction=min_prompt_fraction,
        pad_to_tokens=args.pad_to_tokens,
        seed=args.seed,
        use_wandb=not args.no_wandb,
        run_suffix=args.run_suffix,
    )
    print(
        f"Context {context_length} (prompt cap {context_length - MAX_GENERATION_TOKENS}, "
        f"prompt floor {int(min_prompt_fraction * (context_length - MAX_GENERATION_TOKENS))}, "
        f"pad_to={args.pad_to_tokens})"
    )
    print(
        f"Topology: {topology.describe()}, max_tokens_per_gpu="
        f"{config.recipe.max_tokens_per_gpu}"
    )
    if args.dry_run:
        print("Miles LongRLVR configuration validated")
        print(config.recipe)
        return
    run = config.launch()
    print(f"Training run: {run.training_run_id}")
    print(f"Modal app: {run.modal_app_url}")
    print(f"Function call: {run.function_call_id}")


if __name__ == "__main__":
    main()
