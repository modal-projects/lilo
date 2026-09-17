import subprocess
from types import SimpleNamespace

import pytest

from lilo.providers.modal.miles_revision import resolve_miles_commit


@pytest.fixture(autouse=True)
def fresh_resolution(monkeypatch):
    monkeypatch.delenv("LILO_MILES_COMMIT", raising=False)
    resolve_miles_commit.cache_clear()
    yield
    resolve_miles_commit.cache_clear()


def test_main_resolves_once_and_next_deployment_can_advance(monkeypatch):
    calls = []
    head = ["b" * 40]

    def run(argv, **kwargs):
        calls.append(argv)
        assert argv[-1] == "refs/heads/main"
        assert kwargs["check"] and kwargs["timeout"] == 30
        return SimpleNamespace(stdout=f"{head[0]}\trefs/heads/main\n")

    monkeypatch.setattr(subprocess, "run", run)
    assert resolve_miles_commit() == "b" * 40
    head[0] = "c" * 40
    assert resolve_miles_commit() == "b" * 40
    assert len(calls) == 1
    resolve_miles_commit.cache_clear()
    assert resolve_miles_commit() == "c" * 40


def test_exact_override_does_not_lookup_main(monkeypatch):
    monkeypatch.setenv("LILO_MILES_COMMIT", "d" * 40)
    def unexpected(*args, **kwargs):
        pytest.fail("Pinned reproduction should not query main")
    monkeypatch.setattr(subprocess, "run", unexpected)
    assert resolve_miles_commit() == "d" * 40
    resolve_miles_commit.cache_clear()
    monkeypatch.setenv("LILO_MILES_COMMIT", "main")
    with pytest.raises(ValueError, match="full lowercase Git commit"):
        resolve_miles_commit()


def test_lookup_failure_is_not_cached(monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(2, args[0])
    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        resolve_miles_commit()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=f'{"e" * 40}\trefs/heads/main\n'))
    assert resolve_miles_commit() == "e" * 40
