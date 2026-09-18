import os

TRAINER_MAX_CONTAINERS_ENV = "LILO_TRAINER_MAX_CONTAINERS"
APP_NAME_ENV = "LILO_APP_NAME"


def trainer_max_containers() -> int | None:
    value = os.environ.get(TRAINER_MAX_CONTAINERS_ENV)
    if value is None:
        return None
    try:
        limit = int(value)
    except ValueError as exc:
        raise ValueError(
            f"{TRAINER_MAX_CONTAINERS_ENV} must be a positive integer"
        ) from exc
    if limit < 1:
        raise ValueError(f"{TRAINER_MAX_CONTAINERS_ENV} must be a positive integer")
    return limit


def trainer_deployment_env() -> dict[str, str]:
    env = {
        name: os.environ[name]
        for name in (TRAINER_MAX_CONTAINERS_ENV, APP_NAME_ENV)
        if name in os.environ
    }
    return env
