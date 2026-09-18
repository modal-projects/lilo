import os

TRAINER_MAX_CONTAINERS_ENV = "LILO_TRAINER_MAX_CONTAINERS"


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


FORWARDED_DEPLOYMENT_ENVS = (
    TRAINER_MAX_CONTAINERS_ENV,
    "LILO_TORCH_PROFILE_STEP",
    "LILO_TORCH_PROFILE_DIR",
    "LILO_TORCH_PROFILE_RANKS",
)


def trainer_deployment_env() -> dict[str, str]:
    return {
        name: value
        for name in FORWARDED_DEPLOYMENT_ENVS
        if (value := os.environ.get(name)) is not None
    }
