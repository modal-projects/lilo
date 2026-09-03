from __future__ import annotations

import math
import re
from collections.abc import Sequence
from functools import partial
from typing import Any

import chz
from datasets import load_dataset
from tinker_cookbook import renderers
from tinker_cookbook.recipes.math_rl.math_env import MathEnv, safe_grade
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

DAPO_DATASET = "open-r1/DAPO-Math-17k-Processed"
_ANSWER_PATTERN = re.compile(r"(?i)(?:^|\n)\s*Answer\s*:\s*(.+?)\s*\Z")
_TERMINAL_TOKEN_PATTERN = re.compile(r"(?:<\|endoftext\|>|<\|im_end\|>)\s*\Z")
_USER_STOP_PATTERN = re.compile(r"(?:\r?\n)*User:\s*\Z")


def extract_dapo_answer(text: str) -> str:
    text = _TERMINAL_TOKEN_PATTERN.sub("", text).rstrip()
    match = _ANSWER_PATTERN.search(text)
    if match is None:
        raise ValueError("No terminal DAPO answer found")
    answer = match.group(1).strip()
    if answer.startswith("$") and answer.endswith("$") and len(answer) > 1:
        answer = answer[1:-1].strip()
    return answer


def extract_stopped_dapo_answer(text: str, *, stopped_on_user: bool) -> str:
    if not stopped_on_user:
        raise ValueError("DAPO response did not stop on the next user turn")
    return extract_dapo_answer(text)


class DAPOMathEnv(MathEnv):
    _stopped_on_user = False

    @classmethod
    def question_suffix(cls) -> str:
        return ""

    @property
    def stop_condition(self) -> list[str]:
        # Match the role marker itself so generation cannot continue into a
        # fabricated prompt, regardless of how many newlines precede it.
        return ["User:"]

    async def step(
        self,
        action: Action,
        *,
        extra: ActionExtra | None = None,
    ) -> StepResult:
        text = str(self.renderer.tokenizer.decode(action))
        self._stopped_on_user = _USER_STOP_PATTERN.search(text) is not None
        if text.endswith("User:") and not text.endswith("\n\nUser:"):
            # RoleColonRenderer recognizes only its canonical two-newline
            # delimiter. Keep the sampled action and logprobs unchanged in the
            # trajectory, but normalize the text used for parsing and scoring.
            text = text[: -len("User:")].rstrip("\n") + "\n\nUser:"
            action = self.renderer.tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        return await super().step(action, extra=extra)

    def check_format(self, sample_str: str) -> bool:
        try:
            extract_dapo_answer(sample_str)
            return True
        except ValueError:
            return False

    def check_answer(self, sample_str: str) -> bool:
        try:
            answer = extract_stopped_dapo_answer(
                sample_str,
                stopped_on_user=self._stopped_on_user,
            )
        except ValueError:
            return False
        return safe_grade(answer, self.answer, self.grader, self.timeout)


class DAPOMathDataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        seed: int = 0,
    ):
        self.ds = load_dataset(DAPO_DATASET, split="train").shuffle(seed=seed)
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min(batch_start + self.batch_size, len(self.ds))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            builder
            for row in self.ds.select(range(batch_start, batch_end))
            if (builder := self._make_env_group_builder(row)) is not None
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    def _make_env_group_builder(
        self,
        row: dict[str, Any],
    ) -> ProblemGroupBuilder | None:
        prompt = row.get("source_prompt")
        if isinstance(prompt, list) and prompt:
            problem = prompt[0].get("content", "")
        else:
            problem = row.get("prompt", "")
        answer = row.get("solution") or row.get("reward_model", {}).get(
            "ground_truth", ""
        )
        if not problem or not answer:
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                DAPOMathEnv,
                str(problem),
                str(answer),
                self.renderer,
                convo_prefix=None,
            ),
            num_envs=self.group_size,
            dataset_name="dapo-math-17k",
        )


@chz.chz
class DAPOMathDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    seed: int = 0

    async def __call__(self) -> tuple[DAPOMathDataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        return (
            DAPOMathDataset(
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                seed=self.seed,
            ),
            None,
        )
