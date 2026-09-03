__all__ = [
    "CommandBackend",
    "ForwardBatch",
    "ForwardItem",
    "LossFn",
    "ModelSpec",
    "SamplerPublication",
]


def __getattr__(name: str):
    if name in __all__:
        from . import contract

        return getattr(contract, name)
    raise AttributeError(name)
