import pytest

from lilo.providers.modal.deployment import (
    TRAINER_MAX_CONTAINERS_ENV,
    trainer_deployment_env,
    trainer_max_containers,
)


def test_trainer_max_containers_is_configured_for_deployment(monkeypatch) -> None:
    monkeypatch.setenv(TRAINER_MAX_CONTAINERS_ENV, "3")

    assert trainer_max_containers() == 3
    assert trainer_deployment_env() == {TRAINER_MAX_CONTAINERS_ENV: "3"}


def test_trainer_max_containers_is_unlimited_when_unset(monkeypatch) -> None:
    monkeypatch.delenv(TRAINER_MAX_CONTAINERS_ENV, raising=False)

    assert trainer_max_containers() is None
    assert trainer_deployment_env() == {}


@pytest.mark.parametrize(
    "value",
    [
        "invalid",
        "0",
        "-1",
        "1.5",
    ],
)
def test_trainer_max_containers_rejects_invalid_config(monkeypatch, value) -> None:
    monkeypatch.setenv(TRAINER_MAX_CONTAINERS_ENV, value)

    with pytest.raises(ValueError, match=TRAINER_MAX_CONTAINERS_ENV):
        trainer_max_containers()
