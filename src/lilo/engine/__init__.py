from .api import (
    JSON_OPERATIONS,
    EngineApi,
    Executor,
    FutureState,
    FutureStatus,
    OperationKind,
)
from .http import HttpEngineClient, create_engine_app
from .server import Engine

__all__ = [
    "JSON_OPERATIONS",
    "DistributedExecutor",
    "EngineApi",
    "Engine",
    "Executor",
    "FutureState",
    "FutureStatus",
    "HttpEngineClient",
    "OperationKind",
    "create_engine_app",
]


def __getattr__(name: str):
    if name == "DistributedExecutor":
        from .spmd import DistributedExecutor

        return DistributedExecutor
    raise AttributeError(name)
