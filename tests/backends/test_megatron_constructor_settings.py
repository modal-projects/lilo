"""CPU checks for native config forwarding into Megatron constructors."""

from dataclasses import make_dataclass

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from runtime_stubs import backend_runtime_imports

from lilo.backends.deployment import backend_config
from lilo.backends.megatron_config import parse_backend_config
from lilo.deployments import load, config_path

with backend_runtime_imports():
    from lilo.backends.megatron_runtime.common import modeling


def test_config_overrides_reach_megatron(monkeypatch):
    spec = load(config_path("qwen35-4b-fft-64k"))
    spec.megatron_cfg["optimizer_overrides"] = {"native_optimizer_setting": False}
    spec.megatron_cfg["distributed_overrides"] = {"native_ddp_setting": 123}
    spec.megatron_cfg["provider_overrides"]["native_provider_setting"] = [1, 2]
    config, _ = parse_backend_config(backend_config(spec))
    # These stand in for an installed upstream version with extra fields. The
    # deployment reader must not need its own list of those fields.
    # Model providers in Megatron Bridge are dataclasses.
    fields = {
        **modeling.provider_settings(config, "bf16"),
        "provide_distributed_model": Mock(return_value="model"),
    }
    Provider = make_dataclass("Provider", [(name, object) for name in fields])
    provider = Provider(**fields)
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
    assert actual_provider is not provider
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
    with pytest.raises(TypeError, match="unknown_field"):
        modeling.model_provider(config)
