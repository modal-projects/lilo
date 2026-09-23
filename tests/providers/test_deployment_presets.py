from dataclasses import asdict
from pydantic import TypeAdapter
import pytest

from lilo.deployments import load, config_path, DeploymentRecord
from lilo.backends.deployment import backend_config, serving_options
from lilo.providers.modal.deployment_apps import (
    definition_from_spec,
    frontend_settings,
    manifest_from_env,
)


@pytest.mark.parametrize(
    "path",
    sorted(config_path("qwen35-9b-lora-16k").parent.glob("qwen*.py")),
    ids=lambda p: p.stem,
)
def test_all_packaged_recipes_validate_offline(path):
    spec = load(path)
    config = backend_config(spec)
    assert config[spec.trainer['backend']]["hf_checkpoint"] == "/assets/pending"
    serving_options(spec)


@pytest.mark.parametrize("preset", ["qwen35-35b-a3b-fft-64k", "qwen36-35b-a3b-fft-64k"])
def test_moe_recipes_preserve_trainer_expert_parallelism(preset):
    config = backend_config(load(config_path(preset)))["megatron"]
    assert config["tensor_model_parallel_size"] == 4
    assert config["context_parallel_size"] == 2
    assert config["expert_model_parallel_size"] == 8
    assert config["provider_overrides"]["moe_token_dispatcher_type"] == "alltoall"


def test_moe_rollout_preserves_attention_data_parallelism():
    spec = load(config_path("qwen35-35b-a3b-fft-64k"))
    options = serving_options(spec)
    assert options["tp_size"] == options["dp_size"] == options["ep_size"] == 4
    assert options["enable_dp_attention"] is True
    definition = definition_from_spec(
        DeploymentRecord.create(spec, revision="a" * 40, implementation="test"), register_trainer=False
    )
    assert definition.ROLLOUT_GPUS == 4
    assert definition.ROLLOUT_TENSOR_PARALLEL_SIZE == 1


@pytest.mark.parametrize("context,cp", [(16384, 1), (65536, 2), (131072, 4)])
def test_qwen38_context_parallel_token_budget(context, cp):
    spec = load(config_path(f"qwen38-27b-lora-{context // 1024}k"))
    config = backend_config(spec)["miles"]
    assert config["context_parallel_size"] == cp
    assert config["max_tokens_per_gpu"] == context // cp
    assert config["actor_num_gpus_per_node"] == 8


def test_single_client_recipe_keeps_shared_backend_capacity():
    shared = load(config_path("qwen35-9b-lora-16k"))
    single = load(config_path("qwen35-9b-lora-16k-single"))
    assert single.trainer["engine"]["max_clients_per_instance"] == 1
    assert single.trainer["resources"] == shared.trainer["resources"]
    assert backend_config(single) == backend_config(shared)


@pytest.mark.parametrize("value", [None, "", "[]", "{}"])
def test_missing_manifest_has_no_python_catalog_fallback(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("LILO_DEPLOYMENT_MANIFEST", raising=False)
    else:
        monkeypatch.setenv("LILO_DEPLOYMENT_MANIFEST", value)
    with pytest.raises(ValueError, match="manifest"):
        manifest_from_env()
    with pytest.raises(ValueError, match="manifest"):
        frontend_settings()


@pytest.mark.parametrize(
    "options",
    [
        {"dp_size": 3, "enable_dp_attention": True},
        {"dp_size": 2, "enable_dp_attention": False},
        {"dp_size": 2, "enable_dp_attention": "true"},
    ],
)
def test_invalid_attention_parallelism_is_rejected(options):
    from lilo.deployments import BaseConfig

    data = asdict(load(config_path("qwen35-35b-a3b-fft-64k")))
    data["inference"]["config"].update(options)
    from lilo.backends.deployment import serving_options

    spec = TypeAdapter(BaseConfig).validate_python(data)
    with pytest.raises(ValueError, match="sglang"):
        serving_options(spec)
