from lilo.providers.modal.deployment import trainer_deployment_env


def test_capacity_is_not_forwarded_from_legacy_environment(monkeypatch):
    monkeypatch.setenv("LILO_TRAINER_MAX_CONTAINERS", "3")
    assert "LILO_TRAINER_MAX_CONTAINERS" not in trainer_deployment_env()
