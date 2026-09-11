from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import os
import random
import statistics
import time
import uuid
from pathlib import Path

import httpx
import modal
import tinker
from lilo.client import create_full_training_client_async
from tinker import types

from codegolf.config import Config
from codegolf.judge import judge
from codegolf.pipeline import RolloutBuffer
from codegolf.reward import advantages, datum, extract_code, row_score
from codegolf.store import Store

log = logging.getLogger(__name__)


async def gather_work(*awaitables):
    """Finish or cancel every sibling before clients can be replaced on recovery."""
    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(work) for work in awaitables]
    return [task.result() for task in tasks]


async def retry_read(fn, store, kind, attempts=5):
    for attempt in range(attempts):
        try:
            return await fn()
        except Exception as exc:
            await store.event(
                kind, attempt=attempt, error=f"{type(exc).__name__}: {exc}"
            )
            if attempt + 1 == attempts:
                raise
            await asyncio.sleep(min(60, 2**attempt * 5))


async def release(training):
    if training is None:
        return
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                os.environ["TINKER_BASE_URL"] + "/api/v1/unload_model",
                headers={"X-API-Key": os.environ["TINKER_API_KEY"]},
                json={"model_id": training.model_id},
            )
            response.raise_for_status()
    finally:
        training.holder.close()


async def train(root: Path, data: Path, app: modal.App, cfg: Config, commit=None):
    store = Store(root, commit)
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    spec = {"config": dataclasses.asdict(cfg), "dataset_sha256": digest}
    await store.prepare(spec)
    all_problems = json.loads(data.read_text())["problems"]
    semaphore = asyncio.Semaphore(getattr(cfg, "judge_concurrency", 16))

    async def verify(code, problem):
        async with semaphore:
            return await retry_read(
                lambda: judge(code, problem["tests"], app), store, "judge_retry"
            )

    validated = store.read("validated.json")
    if validated is None:
        results = await gather_work(*(verify(p["reference"], p) for p in all_problems))
        validated = [
            p["id"] for p, r in zip(all_problems, results, strict=True) if r["passed"]
        ]
        await store.write("validated.json", validated)
        await store.event(
            "reference_validation", accepted=len(validated), total=len(all_problems)
        )
    problems = [p for p in all_problems if p["id"] in validated]
    random.Random(cfg.seed).shuffle(problems)
    evaluation, training_problems = (
        problems[: cfg.eval_problems],
        problems[cfg.eval_problems :],
    )
    if len(training_problems) < cfg.prompts_per_step:
        raise ValueError("Too few validated training problems")
    await store.write(
        "split.json",
        {
            "train": [p["id"] for p in training_problems],
            "eval": [p["id"] for p in evaluation],
        },
    )
    training = None
    for recovery in range(30):
        buffer = None
        state = await store.resume()
        step = state["step"]
        completed = store.read("complete.json")
        if completed is not None and completed["step"] >= cfg.steps:
            return state
        service = tinker.ServiceClient(
            base_url=os.environ["TINKER_BASE_URL"], api_key=os.environ["TINKER_API_KEY"]
        )
        try:
            await store.event("trainer_create", recovery=recovery, checkpoint=state)
            training = await create_full_training_client_async(
                service,
                cfg.model,
                rollout={
                    "min_containers": getattr(cfg, "rollout_min_replicas", 1),
                    "max_containers": getattr(cfg, "rollout_max_replicas", 2),
                    "scaledown_window": 1200,
                },
            )
            await store.write(
                "live.json", {"model_id": training.model_id, "step": step}
            )
            if state["path"]:
                await (
                    await training.load_state_with_optimizer_async(state["path"])
                ).result_async()
                await store.event(
                    "trainer_restored",
                    step=step,
                    path=state["path"],
                    model_id=training.model_id,
                )
            tokenizer = await asyncio.to_thread(training.get_tokenizer)
            sampling = await training.save_weights_and_get_sampling_client_async()

            async def group(problem, n, tag, sampler=None):
                sampler = sampler or sampling  # noqa: B023
                # All groups finish before the enclosing loop changes these clients.
                prompt = tokenizer.apply_chat_template(  # noqa: B023
                    [
                        {
                            "role": "system",
                            "content": "Solve the programming problem in Python 3. Make the correct program as short as possible in UTF-8 bytes. Read standard input and write standard output. Output only executable Python code, without explanation or markdown.",
                        },
                        {"role": "user", "content": problem["statement"]},
                    ],
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )

                async def sample():
                    return await sampler.sample_async(  # noqa: B023
                        prompt=types.ModelInput.from_ints(prompt),
                        num_samples=n,
                        sampling_params=types.SamplingParams(
                            max_tokens=cfg.max_tokens, temperature=1.0, top_p=1.0
                        ),
                    )

                response = await retry_read(sample, store, "sampling_retry")
                rows = []
                for sequence in response.sequences:
                    tokens = list(sequence.tokens)
                    lp = list(sequence.logprobs or [])
                    if not tokens or len(tokens) != len(lp):
                        raise RuntimeError("Invalid sampled tokens/logprobs")
                    text = tokenizer.decode(tokens, skip_special_tokens=True)  # noqa: B023
                    code = extract_code(text)
                    rows.append(
                        {
                            "tokens": tokens,
                            "logprobs": lp,
                            "text": text,
                            "code": code,
                            "bytes": len(code.encode()),
                            "truncated": sequence.stop_reason == "length",
                        }
                    )
                judgments = await gather_work(
                    *(verify(r["code"], problem) for r in rows)
                )
                for row, result in zip(rows, judgments, strict=True):
                    row.update(result)
                    row["reward"] = row_score(row, dataclasses.asdict(cfg))
                record = {"problem_id": problem["id"], "prompt": prompt, "rows": rows}
                if tag is not None:
                    await store.write(
                        f"rollouts/{tag}/{hashlib.sha256(problem['id'].encode()).hexdigest()[:16]}.json",
                        record,
                    )
                return record

            async def evaluate(at):
                records = await gather_work(
                    *(group(p, 1, f"eval-{at:04d}") for p in evaluation)
                )
                await store.write(f"eval/{at:04d}.json", summarize(records, at))

            if step >= cfg.steps:
                if store.read(f"eval/{step:04d}.json") is None:
                    await evaluate(step)
                await store.write("complete.json", {"step": step, "checkpoint": state})
                return state

            # A crash after checkpoint commit can interrupt that policy's eval.
            # Finish it before the restored trainer takes another update.
            if (
                step % cfg.eval_every == 0
                or step == store.read("lineage.json", {}).get("step")
            ) and store.read(f"eval/{step:04d}.json") is None:
                await evaluate(step)
            if getattr(cfg, "async_rollouts", False):

                async def produce(ticket, policy_step, sampler):
                    selected = random.Random(cfg.seed + ticket).sample(
                        training_problems, cfg.prompts_per_step
                    )
                    records = await gather_work(
                        *(group(p, cfg.group_size, None, sampler) for p in selected)
                    )
                    for record in records:
                        record.update(
                            sampling_ticket=ticket, behavior_policy_step_min=policy_step
                        )
                    return records

                buffer = RolloutBuffer(
                    produce,
                    policy=(step, sampling),
                    start_ticket=step,
                    workers=cfg.rollout_workers,
                    capacity=cfg.buffer_batches,
                    max_lag=cfg.max_policy_lag,
                )
                prefill_started = time.monotonic()
                await buffer.prefill(cfg.prefill_batches)
                await store.event(
                    "buffer_prefilled",
                    step=step,
                    ready=buffer.queue.qsize(),
                    seconds=time.monotonic() - prefill_started,
                )

            while step < cfg.steps:
                started = time.time()
                next_step = step + 1
                batch = None
                wait_started = time.monotonic()
                if buffer is not None:
                    batch = await buffer.get(step)
                    records = batch.records
                    for record in records:
                        await store.write(
                            f"rollouts/step-{next_step:04d}/{hashlib.sha256(record['problem_id'].encode()).hexdigest()[:16]}.json",
                            record,
                        )
                else:
                    selected = random.Random(cfg.seed + next_step).sample(
                        training_problems, cfg.prompts_per_step
                    )
                    records = await gather_work(
                        *(
                            group(p, cfg.group_size, f"step-{next_step:04d}")
                            for p in selected
                        )
                    )
                rollout_wait_seconds = time.monotonic() - wait_started
                items = []
                total_tokens = sum(len(r["tokens"]) for g in records for r in g["rows"])
                total_sequences = sum(len(g["rows"]) for g in records)
                for g in records:
                    adv = advantages(
                        [r["reward"] for r in g["rows"]], cfg.advantage_std_floor
                    )
                    for r, a in zip(g["rows"], adv, strict=True):
                        items.append(
                            datum(
                                g["prompt"],
                                r["tokens"],
                                r["logprobs"],
                                a,
                                total_tokens / (total_sequences * len(r["tokens"])),
                            )
                        )
                # Never retry a possibly applied update. Any mutation failure rebuilds
                # a trainer and restores the latest durable model+optimizer checkpoint.
                update_started = time.monotonic()
                fb = await training.forward_backward_async(
                    items,
                    "ppo",
                    loss_fn_config={
                        "clip_low_threshold": 0.8,
                        "clip_high_threshold": 1.2,
                    },
                )
                optim = await training.optim_step_async(
                    types.AdamParams(learning_rate=cfg.learning_rate)
                )
                fb_result = await fb.result_async()
                optim_result = await optim.result_async()
                update_seconds = time.monotonic() - update_started
                step = next_step
                metric = summarize(records, step)
                metric.update(
                    seconds=time.time() - started,
                    training=fb_result.metrics,
                    optimizer=optim_result.metrics,
                    model_id=training.model_id,
                )
                if batch is not None:
                    metric["pipeline"] = {
                        "policy_lag_upper_bound": step - 1 - batch.policy_step,
                        "behavior_policy_step_min": batch.policy_step,
                        "sampling_ticket": batch.ticket,
                        "rollout_batch_seconds": batch.seconds,
                        "rollout_wait_seconds": rollout_wait_seconds,
                        "update_seconds": update_seconds,
                        "ready_batches": buffer.queue.qsize(),
                        "inflight_batches": buffer.inflight,
                        "discarded_stale_batches": buffer.discarded,
                    }
                await store.write(f"metrics/{step:04d}.json", metric)
                log.info("STEP %s", json.dumps(metric))
                if step % cfg.checkpoint_every == 0 or step == cfg.steps:
                    saved = await (
                        await training.save_state_async(
                            f"step-{step:04d}-{uuid.uuid4().hex[:8]}"
                        )
                    ).result_async()
                    state = {"step": step, "path": saved.path}
                    await store.write("checkpoint.json", state)
                    await store.event("checkpoint_saved", **state)
                publish_started = time.monotonic()
                sampling = await training.save_weights_and_get_sampling_client_async()
                if buffer is not None:
                    buffer.publish(step, sampling)
                    metric["pipeline"]["publish_seconds"] = (
                        time.monotonic() - publish_started
                    )
                    await store.write(f"metrics/{step:04d}.json", metric)
                if step % cfg.eval_every == 0 or step == cfg.steps:
                    await evaluate(step)
            if step >= cfg.steps:
                await store.write("complete.json", {"step": step, "checkpoint": state})
                return state
        except Exception as exc:
            await store.event(
                "trainer_failure", step=step, error=f"{type(exc).__name__}: {exc}"
            )
            log.exception("Trainer failed; restoring durable checkpoint")
            await asyncio.sleep(min(60, 5 * (recovery + 1)))
        finally:
            if buffer is not None:
                await buffer.close()
            try:
                await release(training)
            except Exception:
                log.exception("Release failed")
            training = None
    raise RuntimeError(
        "Recovery budget exhausted; resume the same run after fixing infrastructure"
    )


def summarize(records, step):
    rows = [r for g in records for r in g["rows"]]
    passed = [r for r in rows if r["passed"]]
    return {
        "step": step,
        "reward": statistics.fmean(r["reward"] for r in rows),
        "pass_rate": len(passed) / len(rows),
        "mean_bytes": statistics.fmean(r["bytes"] for r in rows),
        "passing_bytes": statistics.fmean(r["bytes"] for r in passed)
        if passed
        else None,
        "samples": len(rows),
        "completion_tokens": statistics.fmean(len(r["tokens"]) for r in rows),
        "truncated": sum(r["truncated"] for r in rows),
        "informative_groups": sum(
            len({r["reward"] for r in g["rows"]}) > 1 for g in records
        ),
    }
