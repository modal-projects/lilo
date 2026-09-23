"""CPU checks for native config forwarding into Megatron constructors."""

from dataclasses import asdict
from pydantic import TypeAdapter

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from runtime_stubs import backend_runtime_imports

from lilo.backends.deployment import backend_config
from lilo.backends.megatron_config import parse_backend_config
from lilo.deployments import BaseConfig, load, config_path

with backend_runtime_imports():
    from lilo.backends.megatron_runtime.common import modeling


def test_yaml_native_values_reach_megatron(monkeypatch):
    data = asdict(load(config_path("qwen35-4b-fft-64k")))
    data["trainer"]["config"]["optimizer"]["native_optimizer_setting"] = False
    data["trainer"]["config"]["distributed"] = {"native_ddp_setting": 123}
    data["trainer"]["config"]["provider"]["native_provider_setting"] = [1, 2]
    config, _ = parse_backend_config(
        backend_config(TypeAdapter(BaseConfig).validate_python(data))
    )
    # These stand in for an installed upstream version with extra fields. The
    # deployment reader must not need its own list of those fields.
    provider = SimpleNamespace(
        native_provider_setting=None,
        mtp_num_layers=0,
        recompute_granularity=None,
        recompute_method=None,
        recompute_num_layers=None,
        provide_distributed_model=Mock(return_value="model"),
    )
    bridge = SimpleNamespace(to_megatron_provider=lambda: provider)
    monkeypatch.setattr(
        modeling,
        "AutoBridge",
        SimpleNamespace(from_hf_pretrained=lambda *a, **k: bridge),
    )
    monkeypatch.setattr(modeling, "parameter_dtype", lambda c: "bf16")
    optimizer_constructor = Mock(side_effect=lambda **kwargs: kwargs)
    ddp_constructor = Mock(side_effect=lambda **kwargs: kwargs)
    monkeypatch.setattr(modeling, "MCoreOptimizerConfig", optimizer_constructor)
    monkeypatch.setattr(modeling, "DistributedDataParallelConfig", ddp_constructor)
    _, actual_provider, _ = modeling.model_provider(config)
    assert actual_provider.native_provider_setting == [1, 2]
    assert actual_provider.tensor_model_parallel_size == 2
    optimizer = modeling.optimizer_config(config, "bf16", distributed_optimizer=True)
    assert optimizer["native_optimizer_setting"] is False
    assert optimizer["lr"] == 0.0001
    assert (
        modeling.distributed_model(provider, config, distributed_optimizer=True)
        == "model"
    )
    assert ddp_constructor.call_args.kwargs["native_ddp_setting"] == 123
    assert ddp_constructor.call_args.kwargs["use_distributed_optimizer"] is True
    # An unsupported native field is the installed backend's error at startup.
    config.provider_overrides["unknown_field"] = True
    with pytest.raises(ValueError, match="unknown Megatron provider override"):
        modeling.model_provider(config)
