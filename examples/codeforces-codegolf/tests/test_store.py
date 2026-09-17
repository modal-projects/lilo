import asyncio
import copy

import pytest

from codegolf.store import Store


def test_resume_discards_only_uncheckpointed_metrics(tmp_path):
    async def exercise():
        store = Store(tmp_path)
        await store.write(
            "checkpoint.json", {"step": 5, "path": "tinker://model/checkpoint"}
        )
        for step in range(1, 8):
            await store.write(f"metrics/{step:04d}.json", {"step": step})
        await store.write("eval/0000.json", {"step": 0})
        await store.write("eval/0007.json", {"step": 7})
        assert (await store.resume())["step"] == 5
        assert len(list((tmp_path / "metrics").glob("*.json"))) == 5
        assert len(list((tmp_path / "rolled_back").glob("*.json"))) == 3
        assert len(list((tmp_path / "eval").glob("*.json"))) == 1
        assert (await store.resume())["step"] == 5

    asyncio.run(exercise())


def test_extension_repairs_old_completion_after_target_commit(tmp_path):
    async def exercise():
        store = Store(tmp_path)
        original = {
            "config": {"steps": 100, "learning_rate": 1e-6},
            "dataset_sha256": "same",
        }
        await store.prepare(original)
        checkpoint = {"step": 100, "path": "tinker://model/weights/100"}
        await store.write("checkpoint.json", checkpoint)
        await store.write("complete.json", {"step": 100, "checkpoint": checkpoint})
        extended = copy.deepcopy(original)
        extended["config"]["steps"] = 500
        await store.prepare(extended)
        assert store.read("spec_history/0100.json") == original
        assert store.read("checkpoint.json") == checkpoint
        assert store.read("complete.json") is None
        assert store.read("completions/0100.json")["step"] == 100
        # Simulate a crash after the new spec committed but before marker removal.
        await store.write("complete.json", {"step": 100, "checkpoint": checkpoint})
        await store.prepare(extended)
        assert store.read("complete.json") is None
        assert store.read("spec.json") == extended

    asyncio.run(exercise())


@pytest.mark.parametrize("change", ["smaller_target", "learning_rate", "dataset"])
def test_extension_rejects_other_input_changes(tmp_path, change):
    async def exercise():
        store = Store(tmp_path)
        original = {
            "config": {"steps": 100, "learning_rate": 1e-6},
            "dataset_sha256": "same",
        }
        await store.prepare(original)
        modified = copy.deepcopy(original)
        modified["config"]["steps"] = 500
        if change == "smaller_target":
            modified["config"]["steps"] = 99
        elif change == "learning_rate":
            modified["config"]["learning_rate"] = 1e-5
        else:
            modified["dataset_sha256"] = "different"
        with pytest.raises(ValueError, match="Resume configuration"):
            await store.prepare(modified)
        assert store.read("spec.json") == original

    asyncio.run(exercise())


def test_response_budget_increase_preserves_checkpoint_and_history(tmp_path):
    async def exercise():
        store = Store(tmp_path)
        old = {
            "config": {"steps": 500, "max_tokens": 1536, "learning_rate": 1e-6},
            "dataset_sha256": "same",
        }
        new = copy.deepcopy(old)
        new["config"]["max_tokens"] = 16384
        await store.prepare(old)
        with pytest.raises(ValueError):
            await store.prepare(new)
        cp = {"step": 50, "path": "tinker://model/weights/50"}
        await store.write("checkpoint.json", cp)
        await store.prepare(new)
        assert store.read("checkpoint.json") == cp
        assert store.read("spec.json") == new
        assert (
            len(list((tmp_path / "spec_history").glob("tokens-1536-to-16384-*.json")))
            == 1
        )
        await store.prepare(new)
        with pytest.raises(ValueError):
            await store.prepare(old)
        changed = copy.deepcopy(new)
        changed["config"].update(max_tokens=32768, learning_rate=1e-5)
        with pytest.raises(ValueError):
            await store.prepare(changed)

    asyncio.run(exercise())


def test_legacy_spec_resumes_with_explicit_defaults(tmp_path):
    async def exercise():
        store = Store(tmp_path)
        legacy = {"config": {"steps": 50}, "dataset_sha256": "same"}
        await store.prepare(legacy)
        explicit = copy.deepcopy(legacy)
        explicit["config"].update(advantage_estimator="grpo", eval_samples=1)
        await store.prepare(explicit)
        assert store.read("spec.json") == explicit
        extended = copy.deepcopy(explicit)
        extended["config"]["steps"] = 100
        await store.prepare(extended)
        assert store.read("spec_history/0050.json") == explicit

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "change", [{"advantage_estimator": "tailrl"}, {"eval_samples": 8}]
)
def test_resume_requires_fork_to_change_estimator_or_eval_budget(tmp_path, change):
    async def exercise():
        store = Store(tmp_path)
        legacy = {"config": {"steps": 50}, "dataset_sha256": "same"}
        await store.prepare(legacy)
        changed = copy.deepcopy(legacy)
        changed["config"].update(steps=100, **change)
        with pytest.raises(ValueError, match="Resume configuration"):
            await store.prepare(changed)
        assert store.read("spec.json") == legacy

    asyncio.run(exercise())


def test_rollout_records_share_one_commit_before_observation(tmp_path):
    async def exercise():
        values = {f"rollouts/step-0001/{i}.json": {"problem_id": i} for i in range(4)}
        commits = []
        observed = []

        async def commit():
            assert not observed
            assert {name: store.read(name) for name in values} == values
            commits.append(True)

        def observe(name, value):
            assert commits == [True]
            observed.append((name, value))

        store = Store(tmp_path, commit=commit, observer=observe)
        await store.write_many(values)
        assert commits == [True]
        assert observed == list(values.items())

    asyncio.run(exercise())


def test_failed_batch_commit_does_not_notify_observers(tmp_path):
    async def exercise():
        observed = []

        async def commit():
            raise RuntimeError("commit failed")

        store = Store(
            tmp_path, commit=commit, observer=lambda *args: observed.append(args)
        )
        with pytest.raises(RuntimeError, match="commit failed"):
            await store.write_many({"rollouts/step-0001/a.json": {"rows": []}})
        assert not observed

    asyncio.run(exercise())


def test_resume_commits_only_when_archiving_stale_results(tmp_path):
    async def scenario():
        commits = []

        async def commit():
            commits.append(True)

        store = Store(tmp_path, commit=commit)
        await store.resume()
        assert not commits
        await store.write_many(
            {
                "checkpoint.json": {"step": 1, "path": "checkpoint/1"},
                "metrics/0001.json": {"step": 1},
                "metrics/0002.json": {"step": 2},
                "eval/0002.json": {"step": 2},
            }
        )
        commits.clear()
        assert (await store.resume())["step"] == 1
        assert commits == [True]
        assert store.read("metrics/0001.json") == {"step": 1}
        assert store.read("metrics/0002.json") is None
        assert store.read("eval/0002.json") is None
        await store.resume()
        assert commits == [True]

    asyncio.run(scenario())
