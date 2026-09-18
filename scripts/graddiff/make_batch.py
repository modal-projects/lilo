# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "datasets",
#   "wandb",
#   "tinker>=0.24,<0.25",
#   "tinker-cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@c8ed9c764b59161391156f980102d82f05014765",
# ]
# ///
"""Regenerate the step-0 pinned-prompt batch through the Lilo sampler.

Mirrors scripts/run_longrlvr_lilo_lora.py's step-0 data path (pinned prompts,
grouped sampler, matched advantages) and dumps the assembled datums to the
``lilo-graddiff`` Modal volume for gradient comparison against the trainer-side
dump produced by ``lilo.backends.miles_runtime.graddiff``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import modal

APP_NAME = "lilo-graddiff-client"
VOLUME_NAME = "lilo-graddiff"
VOLUME_ROOT = "/graddiff"
PINNED_PROMPT_PATH = "/root/pinned/pinned_prompts.json"

MODEL_ID = "qwen3_5_9b_miles_lora_16k_dp2"
MODEL_NAME = "Qwen/Qwen3.5-9B"
RENDERER_NAME = "qwen3_5_disable_thinking"
BASE_URL = os.environ.get(
    "TINKER_BASE_URL",
    "https://modal-labs-micah-dev--lilo-dp2-server.us-west.modal.run",
)
BATCH_SIZE = 24
GROUP_SIZE = 8
NUM_GROUPS = 360
MAX_PROMPT_TOKENS = 12_288
MAX_TOKENS = 4_096
TEMPERATURE = 1.0
SEED = 0
KEEP_GROUPS = 16

SCRIPTS_DIR = Path(__file__).resolve().parents[1]

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
secrets = [modal.Secret.from_name("lilo-api", required_keys=["TINKER_API_KEY"])]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "datasets",
        "wandb",
        "tinker>=0.24,<0.25",
        "tinker-cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@c8ed9c764b59161391156f980102d82f05014765",
        "torch",
    )
    .add_local_dir(str(SCRIPTS_DIR), remote_path="/root/scripts")
    .add_local_file(
        str(Path.home() / "work/modal_dl/pinned_prompts.json"),
        PINNED_PROMPT_PATH,
    )
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    volumes={VOLUME_ROOT: volume},
    secrets=secrets,
    timeout=4 * 60 * 60,
    cpu=4,
    memory=32 * 1024,
)
def make_batch() -> dict:
    sys.path.insert(0, "/root/scripts")

    import tinker
    from grouped_tinker_completer import GroupedTinkerTokenCompleter
    from longrlvr_dataset import LongRLVRDatasetBuilder
    from run_longrlvr_lilo_lora import _matched_advantage_stats
    from tinker_cookbook.rl import rollouts as rl_rollouts
    from tinker_cookbook.rl.data_processing import (
        assemble_training_data,
        remove_constant_reward_groups,
    )
    from tinker_cookbook.tokenizer_utils import get_tokenizer, register_tokenizer

    if MODEL_ID != MODEL_NAME:
        register_tokenizer(MODEL_ID, lambda: get_tokenizer(MODEL_NAME))

    # Base-model rollouts ride a fresh rank-32 LoRA on the Lilo sampler: the
    # adapter's B matrix is zero-initialized, so sampled weights are identical
    # to the base model.
    service = tinker.ServiceClient(base_url=BASE_URL)
    training_client = service.create_lora_training_client(
        base_model=MODEL_ID,
        rank=32,
        seed=SEED,
        train_mlp=True,
        train_attn=True,
        train_unembed=False,
    )
    sampling_client = training_client.save_weights_and_get_sampling_client(
        name="graddiff-step0"
    )

    async def _build_dataset():
        builder = LongRLVRDatasetBuilder(
            batch_size=BATCH_SIZE,
            group_size=GROUP_SIZE,
            num_groups=NUM_GROUPS,
            model_name=MODEL_NAME,
            renderer_name=RENDERER_NAME,
            max_prompt_tokens=MAX_PROMPT_TOKENS,
            seed=SEED,
            prompt_file=PINNED_PROMPT_PATH,
        )
        dataset, _ = await builder()
        return dataset

    dataset = asyncio.run(_build_dataset())

    kept_groups = []
    prompt_meta = []
    batch_index = 0
    rolled = 0
    while len(kept_groups) < KEEP_GROUPS:
        env_group_builders = dataset.get_batch(batch_index)

        async def _rollout_all(builders):
            completer = GroupedTinkerTokenCompleter(
                sampling_client,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                group_size=GROUP_SIZE,
            )

            async def _one(builder):
                return await rl_rollouts.do_group_rollout(builder, completer)

            return await asyncio.gather(*[_one(builder) for builder in builders])

        rolled_groups = asyncio.run(_rollout_all(env_group_builders))
        rolled += len(rolled_groups)
        survivors = remove_constant_reward_groups(list(rolled_groups))
        survivor_ids = {id(group) for group in survivors}
        for offset, group in enumerate(rolled_groups):
            if id(group) in survivor_ids:
                kept_groups.append(group)
                prompt_meta.append(
                    {
                        "group_idx": len(kept_groups) - 1,
                        "source_batch": batch_index,
                        "source_offset": offset,
                        "rewards": group.get_total_rewards(),
                    }
                )
            if len(kept_groups) >= KEEP_GROUPS:
                break
        batch_index += 1

    kept_groups = kept_groups[:KEEP_GROUPS]
    prompt_meta = prompt_meta[:KEEP_GROUPS]

    advantages, total_tokens, std_before, std_after = _matched_advantage_stats(
        kept_groups,
        std_normalize=True,
        per_token_scale=True,
    )
    datums, datum_meta = assemble_training_data(kept_groups, advantages)

    def _tensor_list(tensor_data):
        return list(tensor_data.data)

    out_datums = []
    for datum, meta in zip(datums, datum_meta, strict=True):
        inputs = datum.loss_fn_inputs
        target_tokens = _tensor_list(inputs["target_tokens"])
        out_datums.append(
            {
                "group_idx": meta["group_idx"],
                "traj_idx": meta["traj_idx"],
                "input_tokens": datum.model_input.to_ints(),
                "target_tokens": target_tokens,
                "sampled_logprobs": _tensor_list(inputs["logprobs"]),
                "advantages": _tensor_list(inputs["advantages"]),
                "mask": _tensor_list(inputs["mask"]),
                "reward": prompt_meta[meta["group_idx"]]["rewards"][meta["traj_idx"]],
                "response_len": sum(1 for m in inputs["mask"].data if m != 0.0),
            }
        )

    prompt_sha256 = hashlib.sha256(Path(PINNED_PROMPT_PATH).read_bytes()).hexdigest()
    rewards_all = [r for g in kept_groups for r in g.get_total_rewards()]
    payload = {
        "meta": {
            "prompt_sha256": prompt_sha256,
            "groups": len(kept_groups),
            "group_size": GROUP_SIZE,
            "total_tokens": total_tokens,
            "std_normalize": True,
            "per_token_scale": True,
            "std_before": std_before,
            "std_after": std_after,
            "base_url": BASE_URL,
            "model_id": MODEL_ID,
            "rolled_groups": rolled,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "groups": prompt_meta,
        "datums": out_datums,
    }

    out_path = Path(VOLUME_ROOT) / "batch" / "batch.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))
    volume.commit()

    summary = {
        "n_datums": len(out_datums),
        "total_tokens": total_tokens,
        "reward_mean": sum(rewards_all) / len(rewards_all),
        "groups_kept": len(kept_groups),
        "groups_rolled": rolled,
        "fraction_filtered": 1.0 - len(kept_groups) / rolled,
        "response_len_mean": sum(d["response_len"] for d in out_datums)
        / len(out_datums),
        "std_before": std_before,
        "std_after": std_after,
        "prompt_sha256": prompt_sha256,
        "path": str(out_path),
    }
    print(json.dumps(summary, indent=2))
    return {"summary": summary, "payload": payload}


@app.local_entrypoint()
def main() -> None:
    result = make_batch.remote()
    out_dir = Path.home() / "work/graddiff"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "batch.json").write_text(json.dumps(result["payload"]))
    print(json.dumps(result["summary"], indent=2))
    print(f"wrote {out_dir / 'batch.json'}")
