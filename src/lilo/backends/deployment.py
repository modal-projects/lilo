"""Dispatch backend configuration to its backend-owned integration.

These readers are CPU-only. Native libraries validate their options in workers.
"""

from importlib import import_module

TRAINERS = {
    "miles": "lilo.backends.miles_deployment",
    "megatron": "lilo.backends.megatron_deployment",
}
INFERENCE = {"sglang": "lilo.inference.sglang_deployment"}


def _reader(registry, backend):
    try:
        module = registry[backend]
    except KeyError:
        raise ValueError(f"unknown deployment backend: {backend}") from None
    return import_module(module).build_config


def backend_config(spec, asset_path="/assets/pending"):
    return _reader(TRAINERS, spec.trainer.backend)(spec, asset_path)


def serving_options(spec):
    return _reader(INFERENCE, spec.inference.backend)(spec)
