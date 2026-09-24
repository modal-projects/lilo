"""Flat Python recipes. Modal and the backend validate their own options."""

from copy import deepcopy


def _apply_overrides(values, overrides):
    for path, value in overrides.items():
        parts = path.split(".")
        target = values
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = deepcopy(value)


class BaseConfig:
    name = ""
    model = ""
    max_context_length = 16384
    parameterization = "lora"
    revision = "main"
    backend = "miles"

    trainer_gpu = "H100"
    trainer_gpus_per_node = 1
    trainer_nodes = 1
    trainer_cpu = 8
    trainer_memory_mib = 32768
    trainer_max_instances = 1
    trainer_max_clients_per_instance = 1
    trainer_timeout_s = 86400
    trainer_env = {}
    sampler_persistence_concurrency = 8

    inference_gpu = "H100"
    inference_gpus_per_node = 1
    inference_cpu = 8
    inference_memory_mib = 32768
    inference_min_replicas = 0
    inference_max_replicas = 8
    inference_target_concurrency = 16
    inference_scaledown_window_s = 300
    inference_startup_timeout_s = 1200
    inference_env = {}

    miles_cfg = {}
    megatron_cfg = {}
    sglang_cfg = {}
    session_idle_timeout_s = 300
    pool_idle_timeout_s = 300
    sweep_interval_s = 300

    def __init__(self, **kwargs):
        # Copy each ancestor's values before applying its overrides.
        values = {}
        for cls in reversed(type(self).__mro__):
            values.update(
                deepcopy(
                    {
                        key: value
                        for key, value in vars(cls).items()
                        if not key.startswith("_")
                        and key != "overrides"
                        and not callable(value)
                    }
                )
            )
            _apply_overrides(values, vars(cls).get("overrides", {}))
        overrides = kwargs.pop("overrides", {})
        values.update(deepcopy(kwargs))
        _apply_overrides(values, overrides)
        self.__dict__.update(values)
