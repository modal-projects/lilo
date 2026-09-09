# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "datasets",
#   "tinker>=0.24,<0.25",
#   "tinker-cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@c8ed9c764b59161391156f980102d82f05014765",
# ]
# ///

from __future__ import annotations

import argparse
import asyncio
import inspect
import math
import re
import subprocess
import sys
import textwrap
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import patch

import chz
import tinker
from datasets import load_dataset
from grouped_tinker_completer import GroupedTinkerTokenCompleter
from tinker_cookbook import checkpoint_utils, renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.rl.train import AsyncConfig, Config
from tinker_cookbook.rl.train import main as train_main
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

MODEL_NAME = "Qwen/Qwen3.5-35B-A3B"
RENDERER_NAME = "qwen3_5"
DATASET_NAME = "Guanzheng/LongRLVR-Data"
BASE_URL = "https://modal-labs-kailash-dev--lilo-server.us-west.modal.run"

CONTEXT_LENGTH = 65_536
MAX_GENERATION_TOKENS = 8_192
GROUP_SIZE = 8
GROUPS_PER_BATCH = 16
SOURCE_GROUP_MULTIPLIER = 1.5
MAX_STEPS = 10
LEARNING_RATE = 1e-6
GRAD_CLIP_NORM = 1.0
SEED = 0
SCRIPT_PATH = Path(__file__).resolve()

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
    exec(compile(source, filename, "exec"), namespace)
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
            rendered = self.renderer.build_generation_prompt(prompt)
            if len(rendered.to_ints()) > max_prompt_tokens:
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


async def run(
    *,
    steps: int,
    group_size: int,
    groups_per_batch: int,
    source_group_multiplier: float,
    max_generation_tokens: int,
    base_url: str,
    log_path: str,
    seed: int,
) -> None:
    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
    from tinker_cookbook.rl import rollouts as rl_rollouts
    from tinker_cookbook.rl import train as rl_train

    create_adam_params = tinker.AdamParams
    save_checkpoint = checkpoint_utils.save_checkpoint_async

    def clipped_adam_params(*args, **kwargs):
        kwargs.setdefault("grad_clip_norm", GRAD_CLIP_NORM)
        return create_adam_params(*args, **kwargs)

    async def create_full_training(
        service,
        base_model: str,
        rank: int = 32,
        seed: int | None = None,
        train_mlp: bool = True,
        train_attn: bool = True,
        train_unembed: bool = True,
        user_metadata: dict[str, str] | None = None,
    ):
        del rank, seed, train_mlp, train_attn, train_unembed
        from lilo.client import create_full_training_client_async

        return await create_full_training_client_async(
            service,
            base_model,
            user_metadata=user_metadata,
        )

    async def save_supported_checkpoint(*args, **kwargs):
        # Cookbook's kind="both" waits for state persistence and publication.
        # This benchmark publishes sampler weights only, without recovery
        # checkpoints. To retain state saves while overlapping persistence, see:
        # docs/full-fine-tunes.md#persist-checkpoints-without-blocking-every-training-step
        if kwargs.get("kind") == "both":
            kwargs["kind"] = "sampler"
        return await save_checkpoint(*args, **kwargs)

    source_groups_per_batch = math.ceil(groups_per_batch * source_group_multiplier)
    do_async_training = _with_rollout_worker_count(
        rl_train.do_async_training,
        source_groups_per_batch,
    )
    dataset_builder = LongRLVRDatasetBuilder(
        batch_size=source_groups_per_batch,
        group_size=group_size,
        num_groups=steps * source_groups_per_batch,
        model_name=MODEL_NAME,
        renderer_name=RENDERER_NAME,
        max_prompt_tokens=CONTEXT_LENGTH - max_generation_tokens,
        seed=seed,
    )
    config = Config(
        learning_rate=LEARNING_RATE,
        dataset_builder=dataset_builder,
        model_name=MODEL_NAME,
        recipe_name="longrlvr",
        renderer_name=RENDERER_NAME,
        max_tokens=max_generation_tokens,
        log_path=log_path,
        eval_every=0,
        save_every=0,
        base_url=base_url,
        loss_fn="importance_sampling",
        remove_constant_reward_groups=True,
        compute_post_kl=False,
        num_groups_to_log=1,
        rollout_json_export=True,
        async_config=AsyncConfig(
            max_steps_off_policy=1,
            groups_per_batch=groups_per_batch,
        ),
        max_steps=steps,
    )
    with (
        patch.object(
            tinker.ServiceClient,
            "create_lora_training_client_async",
            create_full_training,
        ),
        patch.object(tinker, "AdamParams", clipped_adam_params),
        patch.object(
            checkpoint_utils,
            "save_checkpoint_async",
            save_supported_checkpoint,
        ),
        patch.object(
            rl_rollouts,
            "TinkerTokenCompleter",
            partial(GroupedTinkerTokenCompleter, group_size=group_size),
        ),
        patch.object(
            rl_train,
            "do_async_training",
            do_async_training,
        ),
    ):
        await train_main(config)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 64K LongRLVR on the lilo Qwen3.5-35B full backend."
    )
    parser.add_argument("--steps", type=int, default=MAX_STEPS)
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE)
    parser.add_argument("--groups-per-batch", type=int, default=GROUPS_PER_BATCH)
    parser.add_argument(
        "--source-group-multiplier",
        type=float,
        default=SOURCE_GROUP_MULTIPLIER,
    )
    parser.add_argument(
        "--max-generation-tokens",
        type=int,
        default=MAX_GENERATION_TOKENS,
    )
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--log-path", type=Path)
    parser.add_argument(
        "--detach",
        action="store_true",
        help="run in a detached process and write stdout/stderr beside the log directory",
    )
    args = parser.parse_args()

    if not 0 < args.max_generation_tokens < CONTEXT_LENGTH:
        parser.error(
            f"--max-generation-tokens must be between 1 and {CONTEXT_LENGTH - 1}"
        )
    if args.steps <= 0 or args.group_size <= 1 or args.groups_per_batch <= 0:
        parser.error(
            "steps/groups-per-batch must be positive and group-size must exceed 1"
        )
    if args.source_group_multiplier < 1:
        parser.error("--source-group-multiplier must be at least 1")

    stamp = time.strftime("%Y%m%d%H%M%S")
    log_path = args.log_path or Path(
        f"scripts/results/longrlvr_qwen3_5_35b_a3b_full_64k.{stamp}"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if args.detach:
        output_log = log_path.with_name(f"{log_path.name}.log")
        command = [
            sys.executable,
            str(SCRIPT_PATH),
            "--steps",
            str(args.steps),
            "--group-size",
            str(args.group_size),
            "--groups-per-batch",
            str(args.groups_per_batch),
            "--source-group-multiplier",
            str(args.source_group_multiplier),
            "--max-generation-tokens",
            str(args.max_generation_tokens),
            "--base-url",
            args.base_url,
            "--seed",
            str(args.seed),
            "--log-path",
            str(log_path),
        ]
        with output_log.open("a", encoding="utf-8") as stream:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        print(f"detached pid={process.pid} log={output_log} run={log_path}")
        return

    asyncio.run(
        run(
            steps=args.steps,
            group_size=args.group_size,
            groups_per_batch=args.groups_per_batch,
            source_group_multiplier=args.source_group_multiplier,
            max_generation_tokens=args.max_generation_tokens,
            base_url=args.base_url,
            log_path=str(log_path),
            seed=args.seed,
        )
    )
    print(log_path)


if __name__ == "__main__":
    main()
