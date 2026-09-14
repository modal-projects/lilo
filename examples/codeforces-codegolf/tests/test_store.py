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
