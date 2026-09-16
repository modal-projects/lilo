import asyncio
import dataclasses
import json
from types import SimpleNamespace

import pytest

from codegolf import train as module
from codegolf.reward import advantages, score
from codegolf.store import Store


class Future:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    async def result_async(self):
        if self.error:
            raise self.error
        return self.result


class Tokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return [10, 11]

    def decode(self, tokens, **kwargs):
        return "print(1)" if tokens[0] == 20 else "print(0)"


class Sampler:
    async def sample_async(self, *, num_samples, **kwargs):
        return await Future(
            SimpleNamespace(
                sequences=[
                    SimpleNamespace(
                        tokens=[20 + i % 2] + [30] * i,
                        logprobs=[-0.2] * (i + 1),
                        stop_reason="stop",
                    )
                    for i in range(num_samples)
                ]
            )
        ).result_async()


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("failure", ["optimizer", "publication"])
@pytest.mark.parametrize("estimator", ["grpo", "tailrl"])
def test_checkpoint_recovery(tmp_path, monkeypatch, failure, async_mode, estimator):
    created, restored, released = [], [], []
    identities = []
    fault = [True]

    class Trainer:
        def __init__(self):
            self.model_id = str(len(created))
            self.step = 0

        def get_tokenizer(self):
            return Tokenizer()

        async def save_weights_and_get_sampling_client_async(self):
            # Checkpoint is durable, but publishing for evaluation fails.
            if failure == "publication" and self.step == 1 and fault[0]:
                fault[0] = False
                return await Future(
                    error=RuntimeError("lost publication")
                ).result_async()
            return Sampler()

        async def forward_backward_async(self, items, loss, **kwargs):
            assert loss == "ppo"
            assert len(items) == cfg.group_size
            rewards = [
                score(
                    i % 2 == 0,
                    "print(1)" if i % 2 == 0 else "print(0)",
                    scale=cfg.reward_scale,
                    bonus=cfg.reward_bonus,
                    output_tokens=i + 1,
                    token_penalty=cfg.output_token_penalty,
                    token_scale=cfg.output_token_scale,
                )
                for i in range(cfg.group_size)
            ]
            expected = advantages(rewards, cfg.advantage_std_floor, estimator=estimator)
            total_weights = []
            for i, (item, a) in enumerate(zip(items, expected, strict=True)):
                inputs = item.loss_fn_inputs
                assert inputs["advantages"].data == pytest.approx([0] + [a] * (i + 1))
                assert inputs["logprobs"].data == pytest.approx([0] + [-0.2] * (i + 1))
                assert inputs["weights"].data[0] == 0
                total_weights.append(sum(inputs["weights"].data))
            assert total_weights == pytest.approx([total_weights[0]] * len(items))
            return Future(SimpleNamespace(metrics={}))

        async def optim_step_async(self, params):
            self.step += 1
            # The optimizer applied the update, but its response was lost.
            if failure == "optimizer" and self.step == 2 and fault[0]:
                fault[0] = False
                return Future(error=RuntimeError("lost optimizer response"))
            return Future(SimpleNamespace(metrics={}))

        async def save_state_async(self, name):
            return Future(SimpleNamespace(path=f"checkpoint/{self.step}"))

        async def load_state_with_optimizer_async(self, path):
            self.step = int(path.split("/")[-1])
            restored.append(self.step)
            return Future()

    async def create(*args, **kwargs):
        assert "rollout" not in kwargs
        identities.append(kwargs["user_metadata"])
        trainer = Trainer()
        created.append(trainer)
        return trainer

    async def release(trainer):
        released.append(trainer.model_id)

    async def judge(code, tests, app):
        return {
            "passed": code == "print(1)",
            "tests_passed": int(code == "print(1)"),
            "tests_total": 1,
        }

    async def sleep(_):
        pass

    monkeypatch.setattr(module, "create_full_training_client_async", create)
    monkeypatch.setattr(module, "release", release)
    monkeypatch.setattr(module, "judge", judge)
    monkeypatch.setattr(module.tinker, "ServiceClient", lambda **kwargs: object())
    monkeypatch.setattr(module.asyncio, "sleep", sleep)
    monkeypatch.setenv("TINKER_BASE_URL", "https://unused.invalid")
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")
    data = tmp_path / "data.json"
    data.write_text(
        json.dumps(
            {
                "problems": [
                    {
                        "id": str(i),
                        "statement": "Print 1",
                        "reference": "print(1)",
                        "tests": [{"input": "", "output": "1"}],
                    }
                    for i in range(3)
                ]
            }
        )
    )
    from codegolf.config import AsyncConfig

    config_type = AsyncConfig if async_mode else module.Config
    cfg = config_type(
        steps=3,
        prompts_per_step=1,
        group_size=4,
        eval_problems=1,
        eval_samples=4 if estimator == "tailrl" else 1,
        advantage_estimator=estimator,
        checkpoint_every=1,
        eval_every=1,
    )
    if async_mode:
        cfg = dataclasses.replace(
            cfg, rollout_workers=2, buffer_batches=2, prefill_batches=2
        )
    result = asyncio.run(module.train(tmp_path / "run", data, None, cfg))
    assert result == {"step": 3, "path": "checkpoint/3"}
    assert len(created) == 2
    assert len({identity["attempt_id"] for identity in identities}) == len(created)
    assert len({identity["run_id"] for identity in identities}) == 1
    assert restored == [1]
    assert created[-1].step == 3
    assert released == ["0", "1"]
    store = Store(tmp_path / "run")
    assert store.read("complete.json")["step"] == 3
    assert len(list((tmp_path / "run" / "metrics").glob("*.json"))) == 3
    assert [p.stem for p in sorted((tmp_path / "run" / "eval").glob("*.json"))] == [
        "0000",
        "0001",
        "0002",
        "0003",
    ]
    for at in range(4):
        metric = store.read(f"eval/{at:04d}.json")
        assert metric["samples"] == cfg.eval_samples
        assert metric["eval_samples"] == cfg.eval_samples
        assert metric["pass_at_k"]["1"] == metric["pass_rate"]
        assert metric["best_of_k"]["1"] == metric["reward"]
        assert metric["pass_at_k"][str(cfg.eval_samples)] == 1.0
        if estimator == "tailrl":
            assert metric["pass_at_k"]["2"] == pytest.approx(5 / 6)
    assert store.read("metrics/0003.json")["advantage_estimator"] == estimator
    # Extending a completed run must restore its optimizer, then take new steps.
    result = asyncio.run(
        module.train(tmp_path / "run", data, None, dataclasses.replace(cfg, steps=5))
    )
    assert result == {"step": 5, "path": "checkpoint/5"}
    assert restored == [1, 3]
    assert len(created) == 3 and created[-1].step == 5
    assert store.read("completions/0003.json")["step"] == 3
    assert store.read("complete.json")["step"] == 5
    if async_mode:
        for metric_path in (tmp_path / "run" / "metrics").glob("*.json"):
            metric = json.loads(metric_path.read_text())
            assert 0 <= metric["pipeline"]["policy_lag_upper_bound"] <= 4
            assert metric["pipeline"]["rollout_wait_seconds"] >= 0


def test_failed_rollout_drains_siblings_before_recovery():
    import pytest

    async def scenario():
        started = asyncio.Event()
        cleaned = asyncio.Event()

        async def sibling():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                # Model real async cleanup, such as terminating a judge sandbox.
                await asyncio.sleep(0)
                cleaned.set()

        async def failure():
            await started.wait()
            raise RuntimeError("sampler unavailable")

        with pytest.raises(ExceptionGroup) as error:
            await module.gather_work(sibling(), failure())
        assert any("sampler unavailable" in str(e) for e in error.value.exceptions)
        assert cleaned.is_set()

    asyncio.run(scenario())
