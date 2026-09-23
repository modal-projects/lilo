from dataclasses import asdict
from pydantic import TypeAdapter
import argparse
import asyncio
from types import SimpleNamespace

import httpx
import pytest

from lilo.deployments import (
    DeploymentRecord,
    BaseConfig,
    load,
    config_path,
    validate_frontend,
)
from lilo.deployment_cli import retain_generations
from lilo.control_plane.deployments import DeploymentRoutes
from lilo.backends.deployment import backend_config
from lilo.providers.modal.deployment_apps import definition_from_spec
from lilo.native_options import apply_defaults


def recipe(preset="qwen35-9b-lora-16k", **changes):
    data = asdict(load(config_path(preset)))
    for path, value in changes.items():
        keys = path.split("__")
        target = data
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    return TypeAdapter(BaseConfig).validate_python(data)


def resolved(spec=None, **changes):
    return DeploymentRecord.create(
        spec or recipe(**changes), revision="a" * 40, implementation="test-runtime"
    )


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
    assert config["native_options"]["recompute_num_layers"] == 1
    assert config["extra_args"] == ("--seq-length", "16384")
    large = recipe("qwen35-9b-lora-64k")
    assert large.model.max_context_length == 65536
    assert backend_config(large)["miles"]["actor_num_gpus_per_node"] == 8
    fft = backend_config(recipe("qwen35-4b-fft-64k"))["megatron"]
    assert (fft["tensor_model_parallel_size"], fft["context_parallel_size"]) == (2, 2)
    assert fft["provider_overrides"]["recompute_granularity"] == "full"


def test_no_model_catalog_required():
    spec = recipe(
        model__id="my-org/new-model",
        trainer__config__model_args=None,
        trainer__config__options={
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
        ({"trainer__config__options": {"hf_checkpoint": "other"}}, "managed"),
        ({"trainer__config__options": {"pipeline_model_parallel_size": 2}}, "managed"),
        ({"inference__config": {"model_path": "other"}}, "managed"),
        ({"inference__config": {"tp_size": 2}}, "replica GPU"),
        ({"trainer__engine__max_clients_per_instance": 7}, "multi_lora_n_adapters"),
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
    assert a.generation == resolved(recipe(routing__default=False)).generation
    assert a.generation != resolved(recipe(trainer__resources__gpu="H200:4")).generation
    b = resolved(recipe(model__id="other/Qwen3.5-9B-Base"))
    assert a.asset_path != b.asset_path
    assert (
        a.asset_path
        != DeploymentRecord.create(
            a.spec, revision="b" * 40, implementation="test-runtime"
        ).asset_path
    )


def test_frontend_defaults_and_retained_generations():
    small = resolved()
    large = resolved(recipe("qwen35-9b-lora-64k"))
    routes = DeploymentRoutes([definition(small), definition(large)])
    assert (
        routes.select(small.spec.model.id, "lora").DEFINITION_ID == small.definition_id
    )
    switched = retain_generations(
        [small, large], [resolved(recipe("qwen35-9b-lora-64k", routing__default=True))]
    )
    routes = DeploymentRoutes(map(definition, switched))
    assert (
        routes.select(small.spec.model.id, "lora").DEFINITION_ID == large.definition_id
    )
    # Saved model/checkpoint records continue using their original definition.
    assert (
        routes.select(small.definition_id, "lora").DEFINITION_ID == small.definition_id
    )
    assert routes.capabilities()[0]["max_context_length"] == 65536
    with pytest.raises(ValueError, match="different Lilo/runtime"):
        retain_generations(
            [small.model_copy(update={"implementation": "old"})], [large]
        )
    with pytest.raises(ValueError, match="multiple defaults"):
        validate_frontend(
            [small.spec, recipe("qwen35-9b-lora-64k", routing__default=True)]
        )


def test_ambiguous_model_does_not_get_random_configuration():
    rows = [
        resolved(recipe(routing__default=False)),
        resolved(recipe("qwen35-9b-lora-64k")),
    ]
    routes = DeploymentRoutes(map(definition, rows))
    with pytest.raises(ValueError, match="ambiguous.*16k.*64k"):
        routes.select(rows[0].spec.model.id, "lora")
    assert routes.capabilities() == []


def test_sampling_requires_default_across_training_modes():
    lora = resolved()
    fft = resolved(recipe("qwen35-4b-fft-64k", model__id=lora.spec.model.id))
    routes = DeploymentRoutes(map(definition, [lora, fft]))
    with pytest.raises(ValueError, match="sampling_default"):
        routes.sampling(lora.spec.model.id)
    fft.spec.routing.sampling_default = True
    assert (
        DeploymentRoutes(map(definition, [lora, fft]))
        .sampling(lora.spec.model.id)
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
    apply_defaults(parser, {"use_feature": False, "layers": [3], "tp_size": 8}, argv)
    parsed = parser.parse_args(argv)
    assert vars(parsed) == {
        "use_feature": False,
        "layers": [3],
        "tp_size": 8,
        "unchanged": "keep",
    }
    with pytest.raises(ValueError, match="unknown backend option"):
        apply_defaults(parser, {"typo": 1}, [])
    with pytest.raises(ValueError, match="boolean"):
        apply_defaults(parser, {"use_feature": "false"}, [])
    with pytest.raises(ValueError, match="list"):
        apply_defaults(parser, {"layers": "1,2"}, [])


def test_multiple_models_same_http_service_and_old_binding_survives_switch():
    from lilo.control_plane import ControlPlane, create_control_plane_app
    from lilo.providers.local import InMemoryKeyValueStore
    from lilo.control_plane.keys import model_key

    async def run():
        store = InMemoryKeyValueStore()
        plane = ControlPlane(store, SimpleNamespace())
        first = resolved()
        other = resolved(recipe(name="other", model__id="org/other-model"))
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
                        "base_model": row.spec.model.id,
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
    apply_defaults(parser, {"bias": False, "optional": "supplied"}, argv)
    assert vars(parser.parse_args(argv)) == {
        "bias": False,
        "optional": "supplied",
        "keep": True,
    }
    parser.add_argument("--custom", action="append")
    with pytest.raises(ValueError, match="unsupported argparse action"):
        apply_defaults(parser, {"custom": [1]}, [])


def test_native_type_callbacks_receive_text():
    def readable_int(value):
        return int(value.strip().removesuffix("k")) * (
            1000 if value.endswith("k") else 1
        )

    parser = argparse.ArgumentParser()
    parser.add_argument("--context-length", type=readable_int)
    parser.add_argument("--sizes", nargs="+", type=readable_int)
    apply_defaults(parser, {"context_length": 65536, "sizes": [32, "2k"]}, [])
    args = parser.parse_args([])
    assert args.context_length == 65536
    assert args.sizes == [32, 2000]


def test_native_sections_survive_serialization_without_allowlist():
    import json
    from lilo.backends.megatron_config import parse_backend_config

    spec = recipe("qwen35-4b-fft-64k")
    data = asdict(spec)
    data["trainer"]["config"]["provider"]["future_provider_option"] = {
        "layers": [1, 4],
        "enabled": False,
    }
    data["trainer"]["config"]["optimizer"]["future_optimizer_option"] = 0.125
    data["trainer"]["config"]["distributed"] = {"future_ddp_option": False}
    spec = TypeAdapter(BaseConfig).validate_python(data)
    settings = backend_config(spec, "/assets/pinned")
    config, _ = parse_backend_config(json.loads(json.dumps(settings)))
    assert config.hf_checkpoint == "/assets/pinned"
    assert config.seq_length == spec.model.max_context_length
    assert config.provider_overrides["future_provider_option"] == {
        "layers": [1, 4],
        "enabled": False,
    }
    assert config.optimizer_overrides == {"future_optimizer_option": 0.125}
    assert config.distributed_overrides == {"future_ddp_option": False}
    assert config.optimizer.lr == 0.0001
    assert asdict(spec) == data  # Building does not consume or mutate YAML.


@pytest.mark.parametrize(
    "section,options,match",
    [
        ("provider", {"context_parallel_size": 4}, "managed"),
        ("optimizer", {"bf16": False}, "managed"),
        ("distributed", {"use_distributed_optimizer": False}, "managed"),
        ("runtime", {"optimizer_overrides": {}}, "managed"),
        ("runtime", {"misspelled_loop_option": 1}, "runtime options"),
        ("provider", [], "mapping"),
        ("optimizer", {"optimizer": "sgd"}, "Adam"),
    ],
)
def test_megatron_native_options_preserve_integration_contract(section, options, match):
    spec = recipe("qwen35-4b-fft-64k", **{f"trainer__config__{section}": options})
    with pytest.raises(ValueError, match=match):
        backend_config(spec)


def test_backend_dispatch_rejects_unknown_backend():
    spec = recipe(trainer__backend="missing")
    with pytest.raises(ValueError, match="unknown deployment backend"):
        backend_config(spec)


def test_new_miles_and_sglang_options_need_no_deployment_schema_change():
    from lilo.backends.deployment import serving_options

    spec = recipe(
        trainer__config__options__future_miles_option=[1, 2],
        inference__config__future_sglang_option=False,
    )
    assert backend_config(spec)["miles"]["native_options"]["future_miles_option"] == [
        1,
        2,
    ]
    assert serving_options(spec)["future_sglang_option"] is False


def test_fft_capacity_is_checked_by_backend_setup():
    spec = recipe("qwen35-4b-fft-64k", trainer__engine__max_clients_per_instance=2)
    with pytest.raises(ValueError, match="FFT trainers admit one client"):
        backend_config(spec)


def test_reserved_environment_is_checked_by_modal_setup():
    from lilo.providers.modal.deployment_apps import deployment_env

    spec = recipe(trainer__env={"LILO_BACKEND_CONFIG": "oops"})
    with pytest.raises(ValueError, match="managed"):
        deployment_env(spec.trainer.env)
    assert deployment_env({"MY_SETTING": "value"}) == {"MY_SETTING": "value"}


def test_record_creation_copies_without_reparsing():
    import hashlib
    import json

    spec = recipe()
    original = asdict(spec)
    row = DeploymentRecord.create(spec, revision="a" * 40, implementation="test")
    assert asdict(spec) == original
    assert row.spec.model.revision == "a" * 40
    # Keep the existing manifest fields and hash format stable.
    expected = original | {"model": original["model"] | {"revision": "a" * 40}}
    expected.pop("routing")
    assert (
        row.generation
        == hashlib.sha256(
            json.dumps(["test", expected], sort_keys=True).encode()
        ).hexdigest()
    )
    row.spec.trainer.config["options"]["lora_rank"] = 64
    assert spec.trainer.config["options"]["lora_rank"] == 32
    saved = row.model_dump_json()
    assert DeploymentRecord.model_validate_json(saved) == row


def test_python_config_inheritance_and_independent_defaults(tmp_path):
    from dataclasses import is_dataclass

    path = tmp_path / "model.py"
    path.write_text(
        "from dataclasses import dataclass\n"
        "from lilo.configs.qwen35_9b_lora_64k import Config as ParentConfig\n"
        "@dataclass(kw_only=True)\n"
        "class Config(ParentConfig):\n"
        "    name: str = 'custom'\n"
        "    def __post_init__(self):\n"
        "        super().__post_init__()\n"
        "        self.trainer.config['options']['new_backend_option'] = False\n"
    )
    first, second = load(path), load(path)
    assert is_dataclass(first)
    assert first.name == "custom"
    assert first.model.max_context_length == 65536
    assert first.trainer.config["options"]["new_backend_option"] is False
    first.trainer.config["options"]["target_modules"].append("extra")
    assert "extra" not in second.trainer.config["options"]["target_modules"]
    assert (
        "extra"
        not in load(config_path("qwen35-9b-lora-16k")).trainer.config["options"][
            "target_modules"
        ]
    )


def test_loading_python_config_does_not_call_backend_readers(monkeypatch):
    import lilo.backends.deployment as backends

    monkeypatch.setattr(
        backends, "backend_config", lambda *a: pytest.fail("backend read")
    )
    monkeypatch.setattr(
        backends, "serving_options", lambda *a: pytest.fail("serving read")
    )
    spec = load(config_path("qwen35-9b-lora-16k"))
    original_revision = spec.model.revision
    record = DeploymentRecord.create(spec, revision="a" * 40, implementation="test")
    assert record.spec.model.revision == "a" * 40
    assert spec.model.revision == original_revision


@pytest.mark.parametrize("source", ["Config = {}", "class Config: pass", "value = 1"])
def test_config_file_must_export_config_subclass(tmp_path, source):
    path = tmp_path / "model.py"
    path.write_text(source)
    with pytest.raises(ValueError, match="Config subclass"):
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
