from copy import deepcopy
import json
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import modal
import pytest

from lilo import deployment_cli as cli
from lilo.deployments import load, config_path, DeploymentRecord
from lilo.providers.modal.deployment_records import MANIFEST_ENV, deployed_manifest


def deployment():
    return DeploymentRecord.create(
        load(config_path("qwen35-9b-lora-16k")), revision="a" * 40
    )


@pytest.fixture
def deployed(monkeypatch):
    state = SimpleNamespace(apps=set(), manifest=[], calls=[], fail=None)

    def read_manifest(frontend, environment):
        return state.manifest

    def lookup(name, **kwargs):
        if name not in state.apps:
            raise modal.exception.NotFoundError("not deployed")

    def run(command, *, env, check):
        role = env.get("LILO_WORKER_ROLE", "frontend")
        state.calls.append(role)
        if state.fail == role:
            raise subprocess.CalledProcessError(1, command)
        if role == "frontend":
            state.manifest = json.loads(env[MANIFEST_ENV])
        else:
            row = DeploymentRecord.model_validate_json(env["LILO_WORKER_DEPLOYMENT"])
            state.apps.add(
                row.trainer_app_name if role == "trainer" else row.inference_app_name
            )

    monkeypatch.setattr(cli, "deployed_manifest", read_manifest)
    monkeypatch.setattr(modal.App, "lookup", lookup)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(
        modal.Dict,
        "from_name",
        lambda *a, **k: pytest.fail("deployment must not use a Dict registry"),
    )
    return state


def test_deploy_reuses_modal_apps_and_retains_previous_config(deployed):
    row = deployment()
    cli.deploy([row])
    assert deployed.calls == ["trainer", "inference", "frontend"]
    deployed.calls.clear()
    cli.deploy([row])
    assert deployed.calls == ["frontend"]

    spec = deepcopy(row.spec)
    spec.inference_max_replicas = 6
    changed = DeploymentRecord.create(spec, revision="a" * 40)
    deployed.calls.clear()
    cli.deploy([changed])
    assert deployed.calls == ["inference", "frontend"]
    assert [(r["generation"], r["active"]) for r in deployed.manifest] == [
        (changed.generation, True),
        (row.generation, False),
    ]

    # Modal's actual state wins over a frontend record that mentions an old app.
    deployed.apps.remove(changed.inference_app_name)
    deployed.calls.clear()
    cli.deploy([changed])
    assert deployed.calls == ["inference", "frontend"]


@pytest.mark.parametrize("failure", ["inference", "frontend"])
def test_retry_discovers_completed_workers_without_pending_records(deployed, failure):
    row = deployment()
    deployed.fail = failure
    with pytest.raises(subprocess.CalledProcessError):
        cli.deploy([row])
    assert deployed.manifest == []
    assert row.trainer_app_name in deployed.apps
    deployed.calls.clear()
    deployed.fail = None
    cli.deploy([row])
    assert deployed.calls == (
        ["inference", "frontend"] if failure == "inference" else ["frontend"]
    )


def test_refresh_keeps_other_workers_and_is_remembered_by_frontend(deployed):
    miles = deployment()
    fft = DeploymentRecord.create(
        load(config_path("qwen35-4b-fft-64k")), revision="a" * 40
    )
    cli.deploy([miles, fft])
    deployed.calls.clear()
    cli.deploy([miles, fft], refresh_trainers=[miles.spec.name])
    assert deployed.calls == ["trainer", "frontend"]
    updated = deployed.manifest[0]
    assert updated["trainer_release"] != "initial"
    assert updated["inference_release"] == "initial"
    deployed.calls.clear()
    cli.deploy([miles, fft])
    assert deployed.calls == ["frontend"]
    assert deployed.manifest[0] == updated


def test_existing_frontend_without_deployment_metadata_can_be_updated(deployed):
    row = deployment()
    deployed.apps.add(row.platform["frontend"])
    cli.deploy([row])
    assert deployed.manifest[0]["generation"] == row.generation


def test_read_manifest_from_deployed_function(monkeypatch):
    function = SimpleNamespace(
        hydrate=Mock(), remote=Mock(return_value=[{"configuration": "saved"}])
    )
    lookup = Mock(return_value=function)
    monkeypatch.setattr(modal.Function, "from_name", lookup)
    assert deployed_manifest("my-app", "dev") == [{"configuration": "saved"}]
    lookup.assert_called_once_with(
        "my-app", "deployment_manifest", environment_name="dev"
    )
    function.hydrate.side_effect = modal.exception.NotFoundError("no function")
    assert deployed_manifest("my-app", "dev") == []
    function.hydrate.side_effect = None
    function.remote.side_effect = RuntimeError("frontend failed")
    with pytest.raises(RuntimeError, match="frontend failed"):
        deployed_manifest("my-app", "dev")


def test_validate_never_resolves_or_deploys(monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "compile_configs", lambda *args: pytest.fail("unexpected resolution")
    )
    monkeypatch.setattr(
        cli, "deploy", lambda *args: pytest.fail("unexpected deployment")
    )
    cli.main(["config", "validate", str(config_path("qwen35-9b-lora-16k"))])
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
    from lilo.providers.modal import miles_revision

    path = tmp_path / "model.py"
    path.write_text(
        "from lilo.configs.qwen35_9b_lora_16k import Config as Parent\n"
        f"class Config(Parent):\n    revision = {revision!r}\n"
        "config = Config()\n"
    )
    lookup = Mock(return_value=SimpleNamespace(sha="a" * 40))
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", lookup)
    monkeypatch.setattr(miles_revision, "resolve_miles_commit", lambda: "b" * 40)
    (row,) = cli.compile_configs([path])
    assert row.spec.revision == "a" * 40
    assert lookup.call_count == lookups
    if lookups:
        lookup.return_value.sha = None
        with pytest.raises(ValueError, match="did not return a commit"):
            cli.compile_configs([path])


def test_worker_source_mount_excludes_authoring_configs():
    from pathlib import Path
    from lilo.providers.modal.image_dependencies import ignore_config_source

    assert ignore_config_source(Path("configs/example.py"))
    assert ignore_config_source(Path("configs/__init__.py"))
    assert ignore_config_source(Path("data.json"))
    assert not ignore_config_source(Path("deployments.py"))
    assert not ignore_config_source(Path("backends/miles_config.py"))


def test_deploy_command_owns_platform_settings(monkeypatch):
    seen = {}

    def compile(paths, *, platform):
        seen["platform"] = platform
        return ["record"]

    def deploy(rows, **kwargs):
        seen["rows"] = rows
        seen.update(kwargs)

    monkeypatch.setattr(cli, "compile_configs", compile)
    monkeypatch.setattr(cli, "deploy", deploy)
    cli.main(
        [
            "deploy",
            "model.py",
            "--app",
            "my-lilo",
            "--env",
            "dev",
            "--region",
            "us-east",
            "--refresh-trainer",
            "my-model",
        ]
    )
    assert seen["platform"]["frontend"] == "my-lilo"
    assert seen["platform"]["modal"] == {"environment": "dev", "region": "us-east"}
    assert seen["refresh_trainers"] == ["my-model"]
    assert seen["rows"] == ["record"]


def test_builtin_config_resolves_revision_automatically(monkeypatch):
    from types import SimpleNamespace
    import huggingface_hub
    from lilo.providers.modal import miles_revision

    calls = []
    monkeypatch.setattr(miles_revision, "resolve_miles_commit", lambda: "b" * 40)
    monkeypatch.setattr(
        huggingface_hub.HfApi,
        "model_info",
        lambda self, model, *, revision: calls.append((model, revision))
        or SimpleNamespace(sha="a" * 40),
    )
    (row,) = cli.compile_configs([config_path("qwen35-9b-lora-16k")])
    assert calls == [("Qwen/Qwen3.5-9B-Base", "main")]
    assert row.spec.revision == "a" * 40
    assert load(config_path("qwen35-9b-lora-16k")).revision == "main"
