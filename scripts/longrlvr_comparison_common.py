from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MODEL_NAME = "Qwen/Qwen3.5-9B"
DATASET = "Guanzheng/LongRLVR-Data"
CONTEXT_LENGTH = 16_384
MAX_PROMPT_TOKENS = 12_288
MAX_TOKENS = 4_096
RENDERER_NAME = "qwen3_5_disable_thinking"
GROUP_SIZE = 8
GROUPS_PER_BATCH = 16
SOURCE_GROUPS_PER_BATCH = 24
ROLLOUT_WORKERS = 24
MAX_STEPS = 5
SAVE_EVERY = 1
LEARNING_RATE = 1e-4
ADAM_BETAS = (0.9, 0.95)
ADAM_EPS = 1e-8
WEIGHT_DECAY = 0.0
GRAD_CLIP = 0.0
TEMPERATURE = 1.0
DATASET_SEED = 0
MAX_STEPS_OFF_POLICY = 1
LOSS_FN = "ppo"
LOSS_FN_CONFIG = {"clip_low_threshold": 0.8, "clip_high_threshold": 1.28}
WANDB_PROJECT = "miles-lora-longcontext"
WANDB_ENTITY = "modal-labs"
WANDB_GROUP = "phase1-lilo"


def experiment_config(*, max_steps: int = MAX_STEPS) -> dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "dataset": DATASET,
        "context_length": CONTEXT_LENGTH,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "max_tokens": MAX_TOKENS,
        "group_size": GROUP_SIZE,
        "groups_per_batch": GROUPS_PER_BATCH,
        "source_groups_per_batch": SOURCE_GROUPS_PER_BATCH,
        "rollout_workers": ROLLOUT_WORKERS,
        "max_steps": max_steps,
        "save_every": SAVE_EVERY,
        "learning_rate": LEARNING_RATE,
        "adam_betas": ADAM_BETAS,
        "adam_eps": ADAM_EPS,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "temperature": TEMPERATURE,
        "dataset_seed": DATASET_SEED,
        "max_steps_off_policy": MAX_STEPS_OFF_POLICY,
        "loss_fn": LOSS_FN,
        "loss_fn_config": LOSS_FN_CONFIG,
        "grpo_std_normalization": True,
        "remove_constant_reward_groups": True,
        "rollout_mode": "cohort",
        "train_attn": True,
        "train_mlp": True,
        "train_unembed": False,
        "lora_rank": 32,
        "lora_seed": None,
        "wandb_project": WANDB_PROJECT,
        "wandb_entity": WANDB_ENTITY,
        "wandb_group": WANDB_GROUP,
    }


def write_step_metrics(path: str | Path, metrics: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
