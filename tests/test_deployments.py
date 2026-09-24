from dataclasses import asdict, replace, FrozenInstanceError
from pydantic import TypeAdapter
import argparse
import asyncio
from types import SimpleNamespace

import httpx
import pytest

from lilo.deployments import (
    DeploymentRecord,
    Deployment,
    load,
    config_path,
    validate_frontend,
)
from lilo.deployment_cli import retain_generations
from lilo.control_plane.deployments import DeploymentRoutes
from lilo.backends.deployment import backend_config
from lilo.providers.modal.deployment_apps import definition_from_spec
from lilo.backends.miles_arguments import apply_config_overrides


def recipe(preset="qwen35-9b-lora-16k", **changes):
    data = asdict(load(config_path(preset)))
    for path, value in changes.items():
        keys = path.split("__")
        target = data
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    return TypeAdapter(Deployment).validate_python(data)


def resolved(spec=None, **changes):
    return DeploymentRecord.create(spec or recipe(**changes), revision="a" * 40)


def definition(value):
    return definition_from_spec(value, register_trainer=False)


def test_presets_context_topology_and_backend_options():
    spec = recipe()
    config = backend_config(spec)["miles"]
    assert (
        config["actor_num_gpus_per_node"],
        config["tensor_model_parallel_size"],
        config["max_lora_slots"],
    ) == (4, 4, 6)
    assert config["cli_options"]["recompute_num_layers"] == 1
    assert config["extra_args"] == ("--seq-length", "16384")
    large = recipe("qwen35-9b-lora-64k")
    assert large.max_context_length == 65536
    assert backend_config(large)["miles"]["actor_num_gpus_per_node"] == 8
    fft = backend_config(recipe("qwen35-4b-fft-64k"))["megatron"]
    assert (fft["tensor_model_parallel_size"], fft["context_parallel_size"]) == (2, 2)
    assert fft["provider_overrides"]["recompute_granularity"] == "full"


def test_no_model_catalog_required():
    spec = recipe(
        model="my-org/new-model",
        trainer__config__model_type="",
        trainer__config__cli_options={
            "num_layers": 12,
            "hidden_size": 768,
            "num_attention_heads": 12,
        },
    )
    assert definition(resolved(spec)).MODEL_NAME == "my-org/new-model"
    assert backend_config(spec)["miles"]["model_type"] == ""


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"trainer__config__cli_options": {"hf_checkpoint": "other"}}, "managed"),
        (
            {"trainer__config__cli_options": {"pipeline_model_parallel_size": 2}},
            "managed",
        ),
        ({"inference__config": {"model_path": "other"}}, "managed"),
        ({"inference__config": {"tp_size": 2}}, "replica GPU"),
        ({"trainer__max_clients_per_instance": 7}, "max_lora_slots"),
        (
            {
                "inference__config": {
                    "max_loaded_loras": 2,
                    "max_loras_per_batch": 8,
                }
            },
            "max_loaded_loras",
        ),
    ],
)
def test_invalid_integrations_fail_when_building_backend_settings(changes, match):
    from lilo.backends.deployment import serving_options

    spec = recipe(**changes)
    with pytest.raises(ValueError, match=match):
        backend_config(spec)
        serving_options(spec)


def test_generation_and_asset_paths_include_exact_base():
    a = resolved()
    assert a.generation == resolved(recipe(default=False)).generation
    assert a.generation != resolved(recipe(trainer__gpu="H200")).generation
    b = resolved(recipe(model="other/Qwen3.5-9B-Base"))
    assert a.asset_path != b.asset_path
    assert a.asset_path != DeploymentRecord.create(a.spec, revision="b" * 40).asset_path


def test_frontend_defaults_and_retained_generations():
    small = resolved()
    large = resolved(recipe("qwen35-9b-lora-64k"))
    routes = DeploymentRoutes([definition(small), definition(large)])
    assert routes.select(small.spec.model, "lora").DEFINITION_ID == small.definition_id
    switched = retain_generations(
        [small, large], [resolved(recipe("qwen35-9b-lora-64k", default=True))]
    )
    routes = DeploymentRoutes(map(definition, switched))
    assert routes.select(small.spec.model, "lora").DEFINITION_ID == large.definition_id
    # Saved model/checkpoint records continue using their original definition.
    assert (
        routes.select(small.definition_id, "lora").DEFINITION_ID == small.definition_id
    )
    assert routes.capabilities()[0]["max_context_length"] == 65536
    with pytest.raises(ValueError, match="multiple defaults"):
        validate_frontend([small.spec, recipe("qwen35-9b-lora-64k", default=True)])


def test_ambiguous_model_does_not_get_random_configuration():
    rows = [
        resolved(recipe(default=False)),
        resolved(recipe("qwen35-9b-lora-64k")),
    ]
    routes = DeploymentRoutes(map(definition, rows))
    with pytest.raises(ValueError, match="ambiguous.*16k.*64k"):
        routes.select(rows[0].spec.model, "lora")
    assert routes.capabilities() == []


def test_sampling_requires_default_across_training_modes():
    lora = resolved()
    fft = resolved(recipe("qwen35-4b-fft-64k", model=lora.spec.model))
    routes = DeploymentRoutes(map(definition, [lora, fft]))
    with pytest.raises(ValueError, match="sampling_default"):
        routes.sampling(lora.spec.model)
    fft.spec = replace(fft.spec, sampling_default=True)
    assert (
        DeploymentRoutes(map(definition, [lora, fft]))
        .sampling(lora.spec.model)
        .DEFINITION_ID
        == fft.definition_id
    )


def test_native_false_list_aliases_and_scalar_overrides():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-feature", action="store_true")
    parser.add_argument("--layers", nargs="+", type=int)
    parser.add_argument("--tp", "--tensor-parallel-size", dest="tp_size", type=int)
    parser.add_argument("--unchanged")
    argv = ["--use-feature", "--layers", "1", "2", "--tp=4", "--unchanged", "keep"]
    apply_config_overrides(
        parser, {"use_feature": False, "layers": [3], "tp_size": 8}, argv
    )
    parsed = parser.parse_args(argv)
    assert vars(parsed) == {
        "use_feature": False,
        "layers": [3],
        "tp_size": 8,
        "unchanged": "keep",
    }
    with pytest.raises(ValueError, match="unknown backend option"):
        apply_config_overrides(parser, {"typo": 1}, [])
    with pytest.raises(ValueError, match="boolean"):
        apply_config_overrides(parser, {"use_feature": "false"}, [])
    with pytest.raises(ValueError, match="list"):
        apply_config_overrides(parser, {"layers": "1,2"}, [])


def test_multiple_models_same_http_service_and_old_binding_survives_switch():
    from lilo.control_plane import ControlPlane, create_control_plane_app
    from lilo.providers.local import InMemoryKeyValueStore
    from lilo.control_plane.keys import model_key

    async def run():
        store = InMemoryKeyValueStore()
        plane = ControlPlane(store, SimpleNamespace())
        first = resolved()
        other = resolved(recipe(name="other", model="org/other-model"))
        app = create_control_plane_app(
            plane, list(map(definition, [first, other])), api_key="test"
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://one-frontend",
            headers={"x-api-key": "test"},
        ) as client:
            session = (await client.post("/api/v1/create_session", json={})).json()[
                "session_id"
            ]
            for seq, row in enumerate([first, other]):
                response = await client.post(
                    "/api/v1/create_model",
                    json={
                        "session_id": session,
                        "model_seq_id": seq,
                        "base_model": row.spec.model,
                        "lora_config": {"rank": 32},
                    },
                )
                assert response.status_code == 200, response.text
                record = await store.get(model_key(response.json()["model_id"]))
                assert record["engine_definition_id"] == row.definition_id
            configs = await client.get("/api/v1/lilo/deployments")
            assert len(configs.json()["deployments"]) == 2
            client.headers.clear()
            assert (await client.get("/api/v1/lilo/deployments")).status_code == 401

    asyncio.run(run())


def test_native_boolean_opposite_flags_and_optional_value():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bias", dest="bias", action="store_true")
    parser.add_argument("--no-bias", dest="bias", action="store_false")
    parser.add_argument("--optional", nargs="?")
    parser.add_argument("--keep", action="store_true")
    argv = ["--bias", "--optional", "--keep"]
    apply_config_overrides(parser, {"bias": False, "optional": "supplied"}, argv)
    assert vars(parser.parse_args(argv)) == {
        "bias": False,
        "optional": "supplied",
        "keep": True,
    }
    parser.add_argument("--custom", action="append")
    with pytest.raises(ValueError, match="unsupported argparse action"):
        apply_config_overrides(parser, {"custom": [1]}, [])


def test_native_type_callbacks_receive_text():
    def readable_int(value):
        return int(value.strip().removesuffix("k")) * (
            1000 if value.endswith("k") else 1
        )

    parser = argparse.ArgumentParser()
    parser.add_argument("--context-length", type=readable_int)
    parser.add_argument("--sizes", nargs="+", type=readable_int)
    argv = []
    apply_config_overrides(parser, {"context_length": 65536, "sizes": [32, "2k"]}, argv)
    args = parser.parse_args(argv)
    assert args.context_length == 65536
    assert args.sizes == [32, 2000]


def test_native_sections_survive_serialization_without_allowlist():
    import json
    from lilo.backends.megatron_config import parse_backend_config

    spec = recipe("qwen35-4b-fft-64k")
    data = asdict(spec)
    data["trainer"]["config"]["provider_overrides"]["future_provider_option"] = {
        "layers": [1, 4],
        "enabled": False,
    }
    data["trainer"]["config"]["optimizer_overrides"] = {
        "future_optimizer_option": 0.125
    }
    data["trainer"]["config"]["distributed_overrides"] = {"future_ddp_option": False}
    spec = TypeAdapter(Deployment).validate_python(data)
    settings = backend_config(spec, "/assets/pinned")
    config, _ = parse_backend_config(json.loads(json.dumps(settings)))
    assert config.hf_checkpoint == "/assets/pinned"
    assert config.seq_length == spec.max_context_length
    assert config.provider_overrides["future_provider_option"] == {
        "layers": [1, 4],
        "enabled": False,
    }
    assert config.optimizer_overrides == {"future_optimizer_option": 0.125}
    assert config.distributed_overrides == {"future_ddp_option": False}
    assert config.optimizer.lr == 0.0001
    assert asdict(spec) == data  # Building does not consume or mutate the config.


@pytest.mark.parametrize(
    "section,options,match",
    [
        ("provider_overrides", {"context_parallel_size": 4}, "managed"),
        ("optimizer_overrides", {"bf16": False}, "managed"),
        ("distributed_overrides", {"use_distributed_optimizer": False}, "managed"),
        ("provider_overrides", [], "mapping"),
        ("optimizer", {"optimizer": "sgd"}, "Adam"),
    ],
)
def test_megatron_cli_options_preserve_integration_contract(section, options, match):
    spec = recipe("qwen35-4b-fft-64k", **{f"trainer__config__{section}": options})
    with pytest.raises(ValueError, match=match):
        backend_config(spec)


def test_backend_dispatch_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        recipe(trainer__backend="missing")


def test_new_miles_and_sglang_options_need_no_deployment_schema_change():
    from lilo.backends.deployment import serving_options

    spec = recipe(
        trainer__config__cli_options__future_miles_option=[1, 2],
        inference__config__future_sglang_option=False,
    )
    assert backend_config(spec)["miles"]["cli_options"]["future_miles_option"] == [
        1,
        2,
    ]
    assert serving_options(spec)["future_sglang_option"] is False


def test_fft_capacity_is_checked_by_backend_setup():
    with pytest.raises(ValueError, match="FFT trainers admit one client"):
        recipe("qwen35-4b-fft-64k", trainer__max_clients_per_instance=2)


def test_reserved_environment_is_checked_by_modal_setup():
    from lilo.providers.modal.deployment_apps import deployment_env

    with pytest.raises(ValueError, match="managed"):
        recipe(trainer__env={"LILO_BACKEND_CONFIG": "oops"})
    assert deployment_env({"MY_SETTING": "value"}) == {"MY_SETTING": "value"}


def test_record_creation_copies_without_reparsing():
    import hashlib
    import json

    spec = recipe()
    original = asdict(spec)
    row = DeploymentRecord.create(spec, revision="a" * 40)
    assert asdict(spec) == original
    assert row.spec.revision == "a" * 40
    # The record hash covers settings and the pinned backend dependency, not source.
    expected = original | {"revision": "a" * 40}
    expected.pop("default")
    expected.pop("sampling_default")
    assert (
        row.generation
        == hashlib.sha256(
            json.dumps(
                {
                    "config": expected,
                    "miles_commit": None,
                    "platform": row.platform,
                    "trainer_release": "initial",
                    "inference_release": "initial",
                    "trainer_settings": row.trainer_settings,
                    "inference_settings": row.inference_settings,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
    )
    row.spec.trainer.config["max_lora_rank"] = 64
    assert spec.trainer.config["max_lora_rank"] == 32
    saved = row.model_dump_json()
    assert DeploymentRecord.model_validate_json(saved).model_dump(
        mode="json"
    ) == row.model_dump(mode="json")


def test_python_config_composition(tmp_path):
    path = tmp_path / "model.py"
    path.write_text(
        "from lilo.configs.qwen35_9b_lora_16k import Config as Parent\n"
        "class Config(Parent):\n"
        "    name = 'custom'\n"
        "    trainer = {**Parent.trainer, 'memory_mib': 123456}\n"
        "config = Config()\n"
    )
    custom = load(path)
    original = recipe()
    assert custom.name == "custom"
    assert custom.trainer.memory_mib == 123456
    assert custom.trainer.config == original.trainer.config
    assert original.trainer.memory_mib == 65536
    with pytest.raises(FrozenInstanceError):
        custom.trainer.max_instances = 9


def test_worker_record_contains_resolved_settings(monkeypatch):
    record = resolved()
    assert record.trainer_settings["miles"]["actor_num_gpus_per_node"] == 4
    assert record.inference_settings["max_lora_rank"] == 32
    assert DeploymentRecord.model_validate_json(record.model_dump_json()).model_dump(
        mode="json"
    ) == record.model_dump(mode="json")


@pytest.mark.parametrize("source", ["Config = {}", "class Config: pass", "value = 1"])
def test_config_file_must_export_config_subclass(tmp_path, source):
    path = tmp_path / "model.py"
    path.write_text(source)
    with pytest.raises(ValueError, match="BaseConfig instance"):
        load(path)


def test_config_import_error_preserves_traceback_and_restores_path(tmp_path):
    import sys

    path = tmp_path / "model.py"
    path.write_text("raise RuntimeError('bad user config')")
    before = list(sys.path)
    with pytest.raises(RuntimeError, match="bad user config"):
        load(path)
    assert sys.path == before


def test_no_yaml_config_ingestion(tmp_path):
    with pytest.raises(ValueError, match="Python .py"):
        load(tmp_path / "old.yaml")


@pytest.mark.parametrize(
    "section,key",
    [
        ("trainer", "memroy_mib"),
        ("trainer", "max_instnaces"),
        ("trainer", "min_instances"),
        ("inference", "timeout_s"),
        ("inference", "nodes"),
    ],
)
def test_orchestration_typos_and_unused_fields_are_rejected(section, key):
    base = recipe()
    component = {
        "trainer": base.trainer,
        "inference": base.inference,
    }[section]
    with pytest.raises(ValueError, match=key):
        replace(component, **{key: 9})


def test_worker_hashes_cover_only_their_settings():
    base = resolved()
    inference = resolved(recipe(inference__config__max_running_requests=24))
    assert inference.trainer_hash == base.trainer_hash
    assert inference.inference_hash != base.inference_hash

    trainer = resolved(recipe(trainer__config__max_tokens_per_gpu=8192))
    assert trainer.trainer_hash != base.trainer_hash
    assert trainer.inference_hash == base.inference_hash

    adapter = resolved(recipe(trainer__config__max_lora_rank=64))
    assert adapter.trainer_hash != base.trainer_hash
    assert adapter.inference_hash != base.inference_hash

    routing = resolved(recipe(default=False))
    assert routing.trainer_hash == base.trainer_hash
    assert routing.inference_hash == base.inference_hash
    assert routing.generation == base.generation

    upgraded = DeploymentRecord.create(
        base.spec, revision="a" * 40, inference_release="2"
    )
    assert upgraded.trainer_hash == base.trainer_hash
    assert upgraded.inference_hash != base.inference_hash
    assert "implementation" not in upgraded.model_dump()


def test_examples_only_contain_model_infrastructure():
    from pathlib import Path

    for path in Path(config_path("qwen35-9b-lora-16k")).parent.glob("qwen*.py"):
        config = load(path)
        assert not hasattr(config, "deployment")
        assert config.revision == "main"
        assert "runtime_version" not in asdict(config.trainer)
        assert "runtime_version" not in asdict(config.inference)


@pytest.mark.parametrize("value", ["invalid", 7])
def test_backend_parser_validates_configured_types_and_choices(value):
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, choices=[1, 2], required=True)
    argv = ["--count", "1"]
    apply_config_overrides(parser, {"count": value}, argv)
    with pytest.raises(SystemExit):
        parser.parse_args(argv)


@pytest.mark.parametrize(
    "section,field",
    [
        ("optimizer_overrides", "lr"),
        ("optimizer_overrides", "adam_eps"),
        ("provider_overrides", "calculate_per_token_loss"),
        ("provider_overrides", "attention_backend"),
        ("distributed_overrides", "overlap_grad_reduce"),
    ],
)
def test_managed_backend_values_fail_before_record_creation(section, field):
    base = recipe("qwen35-4b-fft-64k")
    config = {**base.trainer.config, section: {field: 1}}
    candidate = replace(base, trainer=replace(base.trainer, config=config))
    with pytest.raises(ValueError, match=field):
        DeploymentRecord.create(candidate, revision="a" * 40)


def test_multinode_ownership_and_topology():
    config = load(config_path("qwen38-27b-lora-256k"))
    row = DeploymentRecord.create(config, revision="a" * 40)
    miles = row.trainer_settings["miles"]
    assert miles["actor_num_nodes"] == 2
    assert miles["actor_num_gpus_per_node"] == 8
    assert miles["tensor_model_parallel_size"] == 2
    assert miles["context_parallel_size"] == 8
    invalid = replace(
        config,
        trainer=replace(
            config.trainer,
            config={**config.trainer.config, "actor_num_nodes": 3},
        ),
    )
    with pytest.raises(ValueError, match="actor_num_nodes"):
        DeploymentRecord.create(invalid, revision="a" * 40)


def test_config_inheritance_and_constructor_overrides_copy_nested_options():
    from lilo.configs.qwen35_9b_lora_16k import Config as Parent

    class Child(Parent):
        name = "child"
        max_context_length = 8192
        trainer = {
            **Parent.trainer,
            "config": {**Parent.trainer["config"], "max_tokens_per_gpu": 8192},
        }

    first, second = Child(), Child(name="second")
    first.trainer.config["target_modules"].append("extra")
    first.trainer.config["cli_options"]["recompute_num_layers"] = 2
    assert second.name == "second"
    assert second.max_context_length == 8192
    assert second.trainer.gpu == "H100"
    assert backend_config(second)["miles"]["max_tokens_per_gpu"] == 8192
    assert Parent().max_context_length == 16384
    for config in (second, Parent()):
        assert "extra" not in config.trainer.config["target_modules"]
        assert config.trainer.config["cli_options"]["recompute_num_layers"] == 1


@pytest.mark.parametrize(
    "field,value",
    [("max_contex_length", 8192), ("max_context_length", "8192"), ("default", "false")],
)
def test_recipe_class_fields_and_constructor_overrides_are_validated(field, value):
    from lilo.configs.qwen35_9b_lora_16k import Config as Parent

    cls = type("Invalid", (Parent,), {field: value})
    with pytest.raises(ValueError, match=field):
        cls()
    with pytest.raises(ValueError, match=field):
        Parent(**{field: value})


def test_recipe_section_replacement_uses_defaults_without_implicit_merge():
    from lilo.configs.qwen35_9b_lora_16k import Config as Parent

    class Child(Parent):
        inference = {"gpu": "H100", "config": {"max_running_requests": 4}}

    config = Child()
    assert config.inference.gpu == "H100"
    assert config.inference.max_replicas == 8
    assert config.inference.config == {"max_running_requests": 4}
