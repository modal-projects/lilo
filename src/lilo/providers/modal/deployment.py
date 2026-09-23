import os

APP_NAME_ENV = "LILO_APP_NAME"


FORWARDED_DEPLOYMENT_ENVS = (
    APP_NAME_ENV,
    "LILO_TORCH_PROFILE_STEP",
    "LILO_TORCH_PROFILE_DIR",
    "LILO_TORCH_PROFILE_RANKS",
    "LILO_REQUEST_TIMING",
)


def trainer_deployment_env() -> dict[str, str]:
    return {
        name: value
        for name in FORWARDED_DEPLOYMENT_ENVS
        if (value := os.environ.get(name)) is not None
    }
