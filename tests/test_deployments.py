import argparse
import asyncio
from types import SimpleNamespace

import httpx
import pytest
import yaml

from lilo.deployments import (
    DeploymentSpec,
    load,
    preset_path,
    resolve,
    validate_frontend,
)
from lilo.deployment_cli import retain_generations
from lilo.control_plane.deployments import DeploymentRoutes
from lilo.providers.modal.recipe import backend_config
from lilo.providers.modal.yaml_apps import definition_from_spec
from lilo.native_options import apply_defaults


def recipe(preset="qwen35-9b-lora-16k", **changes):
    data = load(preset_path(preset)).model_dump()
    for path, value in changes.items():
        keys = path.split("__")
        target = data
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    return DeploymentSpec.model_validate(data)


def resolved(spec=None, **changes):
    return resolve(
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
        trainer__miles__model_args=None,
        trainer__miles__options={
            "num_layers": 12,
            "hidden_size": 768,
            "num_attention_heads": 12,
        },
    )
    assert definition(resolved(spec)).MODEL_NAME == "my-org/new-model"
    assert backend_config(spec)["miles"]["model_type"] == ""


def test_extends_false_and_lists_replace(tmp_path):
    path = tmp_path / "child.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "extends": "builtin:qwen35-9b-lora-16k",
                "routing": {"default": False},
                "trainer": {"miles": {"options": {"target_modules": ["linear_qkv"]}}},
            }
        )
    )
    spec = load(path)
    assert spec.routing.default is False
    assert spec.trainer.miles.options["target_modules"] == ["linear_qkv"]
    assert spec.trainer.miles.options["lora_rank"] == 32


def test_duplicate_keys_and_cycles(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("name: first\nname: second\n")
    with pytest.raises(ValueError, match="duplicate"):
        load(path)
    path.write_text("extends: bad.yaml\n")
    with pytest.raises(ValueError, match="cyclic"):
        load(path)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"trainer__miles__options": {"hf_checkpoint": "other"}}, "managed"),
        ({"trainer__miles__options": {"pipeline_model_parallel_size": 2}}, "managed"),
        ({"inference__sglang__options": {"model_path": "other"}}, "managed"),
        ({"inference__sglang__options": {"tp_size": 2}}, "replica GPU"),
        ({"trainer__engine__max_clients_per_instance": 7}, "multi_lora_n_adapters"),
        ({"trainer__env": {"LILO_BACKEND_CONFIG": "oops"}}, "managed"),
        (
            {
                "inference__sglang__options": {
                    "max_loaded_loras": 2,
                    "max_loras_per_batch": 8,
                }
            },
            "max_loaded_loras",
        ),
    ],
)
def test_invalid_integrations_fail_locally(changes, match):
    with pytest.raises(ValueError, match=match):
        recipe(**changes)


def test_generation_and_asset_paths_include_exact_base():
    a = resolved()
    assert a.generation == resolved(recipe(routing__default=False)).generation
    assert a.generation != resolved(recipe(trainer__resources__gpu="H200:4")).generation
    b = resolved(recipe(model__id="other/Qwen3.5-9B-Base"))
    assert a.asset_path != b.asset_path
    assert (
        a.asset_path
        != resolve(a.spec, revision="b" * 40, implementation="test-runtime").asset_path
    )
    with pytest.raises(ValueError, match="exact commit"):
        resolve(a.spec, revision="main", implementation="test-runtime")


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
