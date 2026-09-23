import json
import subprocess

import modal
import pytest

from lilo import deployment_cli as cli
from lilo.deployments import load, preset_path, DeploymentRecord
from lilo.providers.modal.yaml_apps import MANIFEST_ENV


class Registry(dict):
    def put(self, key, value, skip_if_exists=False):
        if skip_if_exists and key in self:
            return False
        self[key] = value
        return True


def deployment():
    return DeploymentRecord.create(
        load(preset_path("qwen35-9b-lora-16k")),
        revision="a" * 40,
        implementation="test",
    )


@pytest.fixture
def registry(monkeypatch):
    value = Registry()
    monkeypatch.setattr(modal.Dict, "from_name", lambda *args, **kwargs: value)

    def missing(*args, **kwargs):
        raise modal.exception.NotFoundError("not deployed")

    monkeypatch.setattr(modal.App, "lookup", missing)
    return value


def test_deploy_is_serialized_and_commits_only_after_success(registry, monkeypatch):
    row = deployment()
    seen = []

    def run(command, **kwargs):
        assert "apply_lock" in registry
        assert "pending" in registry
        assert "manifest" not in registry
        assert "lilo.providers.modal.app" in command
        seen.extend(json.loads(kwargs["env"][MANIFEST_ENV]))

    monkeypatch.setattr(subprocess, "run", run)
    cli.deploy([row])
    assert seen == registry["manifest"]
    assert "pending" not in registry and "apply_lock" not in registry
    registry["apply_lock"] = "other-operator"
    with pytest.raises(ValueError, match="An apply owns"):
        cli.deploy([row])
    assert registry["apply_lock"] == "other-operator"


def test_failed_apply_keeps_pending_generations_for_next_attempt(registry, monkeypatch):
    row = deployment()

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        cli.deploy([row])
    assert registry["pending"][0]["generation"] == row.generation
    assert "manifest" not in registry and "apply_lock" not in registry
    new_spec = row.spec.model_copy(deep=True)
    new_spec.trainer.scaling.max_instances = 2
    new = DeploymentRecord.create(new_spec, revision="a" * 40, implementation="test")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: None)
    cli.deploy([new])
    assert [(r["generation"], r["active"]) for r in registry["manifest"]] == [
        (new.generation, True),
        (row.generation, False),
    ]


def test_refuse_overwriting_legacy_frontend(registry, monkeypatch):
    monkeypatch.setattr(modal.App, "lookup", lambda *args, **kwargs: object())
    with pytest.raises(ValueError, match="already exists without a YAML registry"):
        cli.deploy([deployment()])
    assert "pending" not in registry


def test_validate_never_resolves_or_deploys(monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "compile_configs", lambda *args: pytest.fail("unexpected resolution")
    )
    monkeypatch.setattr(
        cli, "deploy", lambda *args: pytest.fail("unexpected deployment")
    )
    cli.main(["config", "validate", str(preset_path("qwen35-9b-lora-16k"))])
    assert "Validated 1 deployment" in capsys.readouterr().out


def test_deploy_rejects_python_mismatch_before_remote_changes(monkeypatch):
    monkeypatch.setattr(cli.sys, "version_info", (3, 11, 0))
    monkeypatch.setattr(
        modal.Dict,
        "from_name",
        lambda *a, **k: pytest.fail("must reject before touching Modal"),
    )
    with pytest.raises(ValueError, match="requires Python 3.12"):
        cli.deploy([deployment()])


@pytest.mark.parametrize("revision,lookups", [("main", 1), ("a" * 40, 0)])
def test_compile_pins_revision_at_external_boundary(
    tmp_path, monkeypatch, revision, lookups
):
    from types import SimpleNamespace
    from unittest.mock import Mock
    import huggingface_hub
    import yaml
    from lilo.providers.modal import miles_revision

    data = load(preset_path("qwen35-9b-lora-16k")).model_dump()
    data["model"]["revision"] = revision
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(data))
    lookup = Mock(return_value=SimpleNamespace(sha="a" * 40))
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", lookup)
    monkeypatch.setattr(miles_revision, "resolve_miles_commit", lambda: "b" * 40)
    monkeypatch.setattr(cli, "implementation_fingerprint", lambda _: "runtime")
    (row,) = cli.compile_configs([path])
    assert row.spec.model.revision == "a" * 40
    assert lookup.call_count == lookups
    if lookups:
        lookup.return_value.sha = None
        with pytest.raises(ValueError, match="did not return a commit"):
            cli.compile_configs([path])
