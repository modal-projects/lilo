# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "datasets",
#   "tinker>=0.24,<0.25",
#   "tinker-cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@c8ed9c764b59161391156f980102d82f05014765",
# ]
# ///

from __future__ import annotations

import inspect
import math
import re
import textwrap
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import chz
import tinker
from datasets import load_dataset
from longrlvr_comparison_common import DATASET as DATASET_NAME
from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    Env,
    EnvGroupBuilder,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

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
    precision = overlap / sum(candidate_tokens.values())
    recall = overlap / sum(reference_tokens.values())
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def _concise_prompt(prompt: list[renderers.Message]) -> list[renderers.Message]:
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


def _with_rollout_worker_count(original, worker_count: int):
    """Compile the pinned Cookbook async loop with independent rollout workers."""
    source = textwrap.dedent(inspect.getsource(inspect.unwrap(original)))
    replacements = {
        "maxsize=config.async_config.groups_per_batch": f"maxsize={worker_count}",
        "_AsyncCounter(config.async_config.groups_per_batch)": (
            f"_AsyncCounter({worker_count})"
        ),
        "range(config.async_config.groups_per_batch)": f"range({worker_count})",
    }
    expected = {
        "maxsize=config.async_config.groups_per_batch": 1,
        "_AsyncCounter(config.async_config.groups_per_batch)": 1,
        "range(config.async_config.groups_per_batch)": 2,
    }
    for old, new in replacements.items():
        if source.count(old) != expected[old]:
            raise RuntimeError(f"unexpected Cookbook async loop shape for {old!r}")
        source = source.replace(old, new)
    stale_requeue_marker = """\
                if (
                    i_batch - wrapped_trajectory_group.sampling_client_step
                    > config.async_config.max_steps_off_policy
                ):
                    if dataloader_done_event.is_set():
                        logger.info(
                            f"[training_loop] Step {i_batch}: Samples are too stale, "
                            "discarding (dataloader done)"
                        )
                    else:
                        logger.info(
                            f"[training_loop] Step {i_batch}: Samples are too stale, requeuing"
                        )
                        asyncio.create_task(
                            env_group_builders_queue.put(
                                wrapped_trajectory_group.env_group_builder
                            ),
                            name="requeue_stale_sample_task",
                        )
                    return False
"""
    if source.count(stale_requeue_marker) != 1:
        raise RuntimeError("unexpected Cookbook stale-sample shape")
    source = source.replace(
        stale_requeue_marker,
        """\
                if (
                    i_batch - wrapped_trajectory_group.sampling_client_step
                    > config.async_config.max_steps_off_policy
                ):
                    logger.info(
                        f"[training_loop] Step {i_batch}: Samples are too stale, "
                        "discarding"
                    )
                    return False
""",
    )
    valid_group_marker = """\
            else:
                trajectory_groups_queue.put_nowait(
                    WrappedTrajectoryGroup(
                        trajectory_group=trajectory_group,
                        env_group_builder=env_group_builder,
                        sampling_client_step=sampling_client_step_copy,
                        metrics=worker_metrics,
                    )
                )
"""
    if source.count(valid_group_marker) != 1:
        raise RuntimeError("unexpected Cookbook valid-group worker shape")
    source = source.replace(
        valid_group_marker,
        valid_group_marker
        + """\
                # Keep at most one valid group per worker and sampler version.
                # Invalid/constant groups retry immediately to preserve headroom.
                while (
                    sampling_client_step == sampling_client_step_copy
                    and not dataloader_done_event.is_set()
                ):
                    await asyncio.sleep(0.1)
""",
    )
    namespace = dict(inspect.unwrap(original).__globals__)
    filename = inspect.getsourcefile(original) or "<cookbook>"
    exec(compile(source, filename, "exec"), namespace)  # noqa: S102
    return namespace["do_async_training"]


@dataclass(frozen=True)
class Reward:
    total: float
    answer_f1: float
    chunk_precision: float
    chunk_recall: float
    chunk_f2: float
    format_valid: float


def score_response(
    response: str,
    *,
    ground_truth: str,
    reference_chunks: Sequence[int],
) -> Reward:
    answer = _single_section(response, "answer")
    useful_chunks = _single_section(response, "useful_chunks")
    if answer is None or useful_chunks is None:
        return Reward(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    predicted = {int(chunk) for chunk in _CHUNK_PATTERN.findall(useful_chunks)}
    reference = set(reference_chunks)
    overlap = len(predicted & reference)
    chunk_precision = overlap / len(predicted) if predicted else 0.0
    chunk_recall = overlap / len(reference) if reference else 0.0
    denominator = 4 * chunk_precision + chunk_recall
    chunk_f2 = 5 * chunk_precision * chunk_recall / denominator if denominator else 0.0
    answer_f1 = _token_f1(answer, ground_truth)

    # LongRLVR uses a binary LLM judge for the answer. This local lexical F1
    # keeps the experiment self-contained while retaining its F2 grounding reward.
    total = answer_f1 + 0.1 * chunk_f2 + 0.9 * answer_f1 * chunk_f2
    return Reward(
        total=total,
        answer_f1=answer_f1,
        chunk_precision=chunk_precision,
        chunk_recall=chunk_recall,
        chunk_f2=chunk_f2,
        format_valid=1.0,
    )


def longrlvr_reward(
    response: str,
    *,
    ground_truth: str,
    reference_chunks: Sequence[int],
) -> dict[str, float]:
    reward = score_response(
        response,
        ground_truth=ground_truth,
        reference_chunks=reference_chunks,
    )
    return {
        "reward": reward.total,
        "answer_f1": reward.answer_f1,
        "chunk_precision": reward.chunk_precision,
        "chunk_recall": reward.chunk_recall,
        "chunk_f2": reward.chunk_f2,
        "format": reward.format_valid,
    }


class LongRLVREnv(Env):
    def __init__(
        self,
        prompt: list[renderers.Message],
        ground_truth: str,
        reference_chunks: tuple[int, ...],
        renderer: renderers.Renderer,
    ) -> None:
        self.prompt = prompt
        self.ground_truth = ground_truth
        self.reference_chunks = reference_chunks
        self.renderer = renderer

    async def initial_observation(
        self,
    ) -> tuple[Observation, StopCondition]:
        return (
            self.renderer.build_generation_prompt(self.prompt),
            self.renderer.get_stop_sequences(),
        )

    async def step(
        self,
        action: Action,
        *,
        extra: ActionExtra | None = None,
    ) -> StepResult:
        del extra
        message, _ = self.renderer.parse_response(action)
        response = renderers.get_text_content(message)
        reward = score_response(
            response,
            ground_truth=self.ground_truth,
            reference_chunks=self.reference_chunks,
        )
        return StepResult(
            reward=reward.total,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.renderer.get_stop_sequences(),
            metrics={
                "answer_f1": reward.answer_f1,
                "chunk_precision": reward.chunk_precision,
                "chunk_recall": reward.chunk_recall,
                "chunk_f2": reward.chunk_f2,
                "format": reward.format_valid,
            },
            logs={
                "answer": _single_section(response, "answer") or "",
                "ground_truth": self.ground_truth,
            },
        )


class LongRLVRDataset(RLDataset):
    def __init__(
        self,
        *,
        batch_size: int,
        group_size: int,
        num_groups: int,
        model_name: str,
        renderer_name: str,
        max_prompt_tokens: int,
        seed: int,
    ) -> None:
        tokenizer = get_tokenizer(model_name)
        self.tokenizer = tokenizer
        self.renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
        self.batch_size = batch_size
        self.group_size = group_size
        self.rows = self._load_rows(
            num_groups=num_groups,
            max_prompt_tokens=max_prompt_tokens,
            seed=seed,
        )

    def _load_rows(
        self,
        *,
        num_groups: int,
        max_prompt_tokens: int,
        seed: int,
    ) -> list[dict[str, Any]]:
        dataset = load_dataset(DATASET_NAME, split="train", streaming=True).shuffle(
            seed=seed,
            buffer_size=max(256, 4 * num_groups),
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
            prompt_chars = sum(
                len(str(message.get("content", ""))) for message in prompt
            )
            if prompt_chars > 6 * max_prompt_tokens:
                continue
            rendered = self.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            input_ids = self.tokenizer(rendered, add_special_tokens=False)["input_ids"]
            if len(input_ids) > max_prompt_tokens:
                continue
            seen_questions.add(question)
            rows.append(
                {
                    "prompt": prompt,
                    "ground_truth": ground_truth,
                    "reference_chunks": tuple(int(chunk) for chunk in reference_chunks),
                }
            )
            if len(rows) == num_groups:
                return rows
        raise RuntimeError(
            f"found only {len(rows)} usable rows; expected {num_groups} "
            f"with prompts no longer than {max_prompt_tokens} tokens"
        )

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.rows))
        if start >= end:
            raise IndexError(index)
        return [
            ProblemGroupBuilder(
                env_thunk=partial(
                    LongRLVREnv,
                    row["prompt"],
                    row["ground_truth"],
                    row["reference_chunks"],
                    self.renderer,
                ),
                num_envs=self.group_size,
                dataset_name="longrlvr",
            )
            for row in self.rows[start:end]
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.rows) / self.batch_size)


@chz.chz
class LongRLVRDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    group_size: int
    num_groups: int
    model_name: str
    renderer_name: str
    max_prompt_tokens: int
    seed: int = 0

    async def __call__(self) -> tuple[LongRLVRDataset, None]:
        return (
            LongRLVRDataset(
                batch_size=self.batch_size,
                group_size=self.group_size,
                num_groups=self.num_groups,
                model_name=self.model_name,
                renderer_name=self.renderer_name,
                max_prompt_tokens=self.max_prompt_tokens,
                seed=self.seed,
            ),
            None,
        )
