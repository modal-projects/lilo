import pytest

from codegolf.judge import output_digest, verdict

TESTS = [{"input": "1", "output": "1"}, {"input": "2", "output": "2"}]


def test_runtime_failure_can_stop_remaining_tests():
    result = verdict([{"ok": False, "digest": None}], TESTS)
    assert result == {
        "passed": False,
        "tests_passed": 0,
        "tests_run": 1,
        "tests_total": 2,
    }


def test_incomplete_success_cannot_receive_reward():
    with pytest.raises(RuntimeError):
        verdict([{"ok": True, "digest": output_digest("1")}], TESTS)


def test_every_test_must_match():
    assert verdict(
        [
            {"ok": True, "digest": output_digest(" 1\n")},
            {"ok": True, "digest": output_digest("2")},
        ],
        TESTS,
    )["passed"]
    assert not verdict(
        [
            {"ok": True, "digest": output_digest("1")},
            {"ok": True, "digest": output_digest("1")},
        ],
        TESTS,
    )["passed"]


def test_output_digest_preserves_token_comparison():
    assert output_digest("1\t2\n") == output_digest(" 1\u00a02 ")
    assert output_digest("1 23") != output_digest("12 3")
    assert output_digest("1") != output_digest("1.0")
    assert len(output_digest("x" * 1000000)) == 64


def test_stalled_output_and_cleanup_are_bounded():
    import asyncio
    from types import SimpleNamespace

    from codegolf.judge import collect_result

    async def exercise():
        cleaned = asyncio.Event()
        cancelled = asyncio.Event()

        async def read():
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def terminate():
            cleaned.set()
            await asyncio.Event().wait()

        sandbox = SimpleNamespace(
            object_id="stalled-test-sandbox",
            stdout=SimpleNamespace(read=SimpleNamespace(aio=read)),
            terminate=SimpleNamespace(aio=terminate),
        )
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                collect_result(sandbox, TESTS, timeout=0.01, cleanup_timeout=0.01),
                timeout=1,
            )
        assert cleaned.is_set() and cancelled.is_set()

    asyncio.run(exercise())


def test_cleanup_timeout_preserves_completed_verdict():
    import asyncio
    import json
    from types import SimpleNamespace

    from codegolf.judge import collect_result

    async def exercise():
        async def read():
            return json.dumps([{"ok": True, "digest": output_digest("1")}])

        async def wait():
            return None

        async def terminate():
            await asyncio.Event().wait()

        sandbox = SimpleNamespace(
            object_id="cleanup-test-sandbox",
            returncode=0,
            stdout=SimpleNamespace(read=SimpleNamespace(aio=read)),
            wait=SimpleNamespace(aio=wait),
            terminate=SimpleNamespace(aio=terminate),
        )
        result = await asyncio.wait_for(
            collect_result(sandbox, TESTS[:1], timeout=1, cleanup_timeout=0.01),
            timeout=2,
        )
        assert result["passed"]

    asyncio.run(exercise())


def test_large_payload_uses_stdin_and_preserves_verdict(tmp_path):
    import base64
    import json
    import subprocess
    import sys

    from codegolf.judge import RUNNER

    # Local test runs unprivileged; the deployed runner's privilege drop remains
    # unchanged and is separately exercised by the remote smoke test.
    runner = RUNNER.replace("os.setgid(65534);os.setuid(65534)", "pass")
    runner = runner.replace("/tmp/solution.py", str(tmp_path / "solution.py"))
    code = "#" + "padding" * 16000 + "\nprint(len(input()))"
    tests = [{"input": "a" * 90000 + "\n", "output": "90000\n"}]
    payload = base64.b64encode(
        json.dumps({"code": code, "inputs": [tests[0]["input"]]}).encode()
    )
    assert len(payload) > 65536
    result = subprocess.run(
        [sys.executable, "-c", runner],
        input=payload,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert verdict(json.loads(result.stdout), tests)["passed"]
