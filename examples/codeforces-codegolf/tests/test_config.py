import dataclasses
import json

import pytest

from codegolf import cli
from codegolf.config import Config, config_for
from fork_checkpoint import validate_config_change


def test_tailrl_variant_matches_async_v6_training_setup():
    baseline = dataclasses.asdict(config_for("async-v6", steps=30, eval_samples=8))
    tailrl = dataclasses.asdict(config_for("tailrl", steps=30))
    assert tailrl == {**baseline, "advantage_estimator": "tailrl"}
    assert config_for().advantage_estimator == "grpo"
    assert config_for().eval_samples == 1


def test_cli_config_selects_estimator_and_evaluation_budget(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        ["codegolf", "config", "--variant", "tailrl", "--eval-samples", "32"],
    )
    cli.main()
    config = json.loads(capsys.readouterr().out)
    assert config["advantage_estimator"] == "tailrl"
    assert config["eval_samples"] == 32


@pytest.mark.parametrize("value", [0, -1, 1.5])
def test_invalid_eval_samples(value):
    with pytest.raises(ValueError, match="eval_samples"):
        config_for("tailrl", eval_samples=value)


def test_invalid_estimator():
    with pytest.raises(ValueError, match="advantage estimator"):
        Config(advantage_estimator="typo")


def test_legacy_checkpoint_can_fork_into_tailrl():
    source = dataclasses.asdict(config_for("async-v6"))
    source.pop("advantage_estimator")
    source.pop("eval_samples")
    destination = dataclasses.asdict(config_for("tailrl", steps=1000))
    validate_config_change(source, destination)
    for field, value in [("model", "different"), ("group_size", 16), ("seed", 0)]:
        with pytest.raises(ValueError, match="Only reward/estimator"):
            validate_config_change(source, {**destination, field: value})


def test_cli_launch_forwards_evaluation_budget(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    launched = []

    def spawn(*args, **kwargs):
        launched.append((args, kwargs))
        return SimpleNamespace(object_id="test-call")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODAL_ENVIRONMENT", "test")
    monkeypatch.setattr(
        cli.modal.Function,
        "from_name",
        lambda *args: SimpleNamespace(spawn=spawn),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "codegolf",
            "launch",
            "--run",
            "tail",
            "--variant",
            "tailrl",
            "--eval-samples",
            "16",
        ],
    )
    cli.main()
    assert launched == [(("tail", 500, "tailrl"), {"eval_samples": 16})]
    assert json.loads(capsys.readouterr().out)["eval_samples"] == 16
