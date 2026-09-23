from dataclasses import replace
import asyncio
import json
from types import SimpleNamespace

import modal
import pytest

from lilo.deployments import load, config_path, DeploymentRecord
from lilo.providers.modal import deployment_apps, deployment_records
from lilo.providers.modal.fft_pool import FFTPoolSpec
from lilo.providers.modal.lora_pool import LoraPoolSpec


def deployment(preset="qwen35-9b-lora-16k"):
    return DeploymentRecord.create(load(config_path(preset)), revision="a" * 40)


class App:
    def __init__(self, name):
        self.name = name
        self.functions = {}
        self.servers = {}

    def function(self, **settings):
        def decorate(fn):
            self.functions[settings["name"]] = (settings, fn)
            return fn

        return decorate

    def server(self, **settings):
        def decorate(cls):
            self.servers[settings["name"]] = (settings, cls)
            return cls

        return decorate


@pytest.fixture
def builders(monkeypatch):
    monkeypatch.setattr(modal, "App", App)
    monkeypatch.setattr(modal, "enter", lambda: lambda fn: fn)
    monkeypatch.setattr(modal, "exit", lambda: lambda fn: fn)


@pytest.mark.parametrize(
    "preset,backend,clients,nproc",
    [
        ("qwen35-9b-lora-16k", "miles_lora", 6, 1),
        ("qwen35-4b-fft-64k", "megatron_fft", 1, 4),
    ],
)
def test_trainer_declaration_and_executor_configuration(
    builders, monkeypatch, preset, backend, clients, nproc
):
    row = deployment(preset)
    row.platform["storage"]["checkpoints"] = "test-custom-checkpoints"
    image = object()
    app, trainer = deployment_apps.build_trainer_app(row, image=image)
    declaration, _ = app.functions["trainer"]
    assert declaration["gpu"] == "H100:4"
    assert declaration["region"] == "us-west"
    assert declaration["max_containers"] is None
    assert declaration["single_use_containers"] is True
    assert declaration["image"] is image
    calls = []
    reloaded = []

    monkeypatch.setattr(deployment_apps, "shared_kv", lambda: "store")
    monkeypatch.setattr(
        deployment_apps,
        "volumes_for",
        lambda spec: {"/assets": SimpleNamespace(reload=lambda: reloaded.append(True))},
    )
    monkeypatch.setattr(
        deployment_apps,
        "run_engine_with_backend",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    trainer("instance-a", row.model_dump_json())
    args, kwargs = calls[0]
    assert args == ("store", f"lilo.backends.{backend}:build_executor")
    assert kwargs["max_models"] == clients
    assert kwargs["nproc"] == nproc
    assert kwargs["backend_env"]["LILO_CHECKPOINT_VOLUME"] == "test-custom-checkpoints"
    assert kwargs["backend_env"]["LILO_BASE_MODEL_REVISION"] == "a" * 40
    config = json.loads(kwargs["backend_env"]["LILO_BACKEND_CONFIG"])
    assert config[row.spec.trainer.backend]["hf_checkpoint"] == row.asset_path
    assert config["checkpoint_dir"] == "/checkpoints"
    assert reloaded == [True]


@pytest.mark.parametrize("kind", ["lora", "base", "latest", "pinned"])
def test_pool_starts_native_server_and_correct_sidecar(builders, monkeypatch, kind):
    row = deployment("qwen35-9b-lora-16k" if kind == "lora" else "qwen35-4b-fft-64k")
    pool = (
        LoraPoolSpec(row.definition_id)
        if kind == "lora"
        else (
            FFTPoolSpec.base(row.definition_id)
            if kind == "base"
            else FFTPoolSpec(row.definition_id, "job", kind == "latest", 7)
        )
    )
    app, server = deployment_apps.build_rollout_app(row, pool, image="test-image")
    settings, _ = app.servers["Server"]
    assert app.name == pool.app_name
    assert settings["gpu"] == row.spec.inference.compute.modal_gpu
    assert settings["min_containers"] == 0
    assert settings["target_concurrency"] == 16
    assert settings["compute_region"] == "us-west"
    import subprocess

    calls, commands, stops = [], [], []
    process = object()
    monkeypatch.setattr(
        subprocess, "Popen", lambda argv, **kw: (commands.append(argv) or process)
    )
    monkeypatch.setattr(deployment_apps, "wait_http", lambda *args: None)
    monkeypatch.setattr(deployment_apps, "supervise_children", lambda *args: None)
    monkeypatch.setattr(
        deployment_apps,
        "start_lora_sidecar",
        lambda **kw: (calls.append(("lora", kw)) or process),
    )
    monkeypatch.setattr(
        deployment_apps,
        "start_fft_sidecar",
        lambda **kw: (calls.append(("fft", kw)) or process),
    )
    monkeypatch.setattr(deployment_apps, "terminate", stops.append)
    replica = server()
    replica.start()
    assert commands[0][2] == "lilo.inference.sglang"
    assert commands[0][3] == row.asset_path
    native = json.loads(commands[0][4])
    assert native["context_length"] == row.spec.model.max_context_length
    if kind == "lora":
        assert native["enable_lora"] is True
        assert native["max_lora_rank"] == 32
        assert native["lora_target_modules"] == [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ]
    else:
        assert native["enable_cpu_weight_cache"] is True
        assert calls[0][1]["pinned_version"] == (
            None if kind == "latest" else 0 if kind == "base" else 7
        )
    replica.stop()
    assert stops == [process, process]


def test_pool_subprocess_receives_recorded_generation(monkeypatch):
    row = deployment()
    monkeypatch.setenv(deployment_records.MANIFEST_ENV, json.dumps([row.model_dump()]))
    env = deployment_records.pool_environment(row.definition_id)
    assert (
        json.loads(env[deployment_records.POOL_CONFIG_ENV])["generation"]
        == row.generation
    )
    with pytest.raises(ValueError, match="missing recorded"):
        deployment_records.pool_environment("yaml_missing_123")
    with pytest.raises(ValueError, match="missing recorded"):
        deployment_records.pool_environment("unconfigured-python-definition")


def test_startup_failure_is_visible_and_blocks_new_spawns(monkeypatch):
    import importlib
    from lilo.providers.local import InMemoryKeyValueStore

    app = importlib.import_module("lilo.providers.modal.app")
    store = InMemoryKeyValueStore()
    row = deployment()
    monkeypatch.setattr(app, "shared_kv", lambda: store)

    async def run():
        await store.put(
            f"deployment_failure:{row.definition_id}",
            {"error": "backend exited with code 1", "instance_id": "failed-instance"},
        )
        with pytest.raises(ValueError, match="Trainer startup failed.*failed-instance"):
            await app._spawn_engine(row.definition_id, "new-instance")
        assert await app.deployment_error("legacy") is None

    asyncio.run(run())


def test_real_modal_app_constructs_from_manifest_without_legacy_catalog(monkeypatch):
    import os
    import subprocess
    import sys

    row = deployment()
    env = {
        **os.environ,
        deployment_records.MANIFEST_ENV: json.dumps([row.model_dump()]),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib, sys, modal
from lilo.providers.modal import deployment_apps, deployment_records
deployment_apps.image_for = lambda backend: modal.Image.debian_slim()
app = importlib.import_module('lilo.providers.modal.app')
assert len(app.DEFINITIONS) == 1
assert app.APP_NAME == 'lilo-yaml'
assert app.DEFINITIONS[0].ENGINE_FUNCTION is not None
assert not any(name.startswith('lilo.providers.modal.definitions.') for name in sys.modules)
print('constructed')
""",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "constructed" in result.stdout


def test_admission_changes_preserve_serialized_trainer(builders):
    from modal._serialization import serialize

    first = deployment()
    old_bytes = serialize(deployment_apps.build_trainer_app(first, image="test")[1])
    changed = first.model_copy(deep=True)
    changed.active = False
    changed.spec = replace(
        changed.spec,
        routing=replace(changed.spec.routing, default=False, sampling_default=True),
    )
    new_bytes = serialize(deployment_apps.build_trainer_app(changed, image="test")[1])
    assert new_bytes == old_bytes
    assert first.active is True and first.spec.routing.default is True
    changed.spec = replace(
        changed.spec,
        trainer=replace(
            changed.spec.trainer,
            compute=replace(changed.spec.trainer.compute, gpu="H200"),
        ),
    )
    assert (
        serialize(deployment_apps.build_trainer_app(changed, image="test")[1])
        != old_bytes
    )


@pytest.mark.parametrize("kind", ["lora", "full"])
def test_pool_launch_uses_only_generic_yaml_app(monkeypatch, kind):
    from lilo.providers.modal import fft_pool, lora_pool

    row = deployment("qwen35-9b-lora-16k" if kind == "lora" else "qwen35-4b-fft-64k")
    monkeypatch.setenv(deployment_records.MANIFEST_ENV, json.dumps([row.model_dump()]))
    module = lora_pool if kind == "lora" else fft_pool
    spec = (
        LoraPoolSpec(row.definition_id)
        if kind == "lora"
        else FFTPoolSpec(row.definition_id, "model", True, 0)
    )
    calls = []

    class Pool:
        def __init__(self, *args):
            self.lookups = 0

        def gateway_url(self):
            self.lookups += 1
            if self.lookups == 1:
                raise modal.exception.NotFoundError("not deployed")
            return "https://pool"

    monkeypatch.setattr(module, "ModalFlashPool", Pool)
    monkeypatch.setattr(module.shutil, "which", lambda _: "/bin/modal")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )
    assert module.deploy_pool(spec, record=row) == "https://pool"
    command, kwargs = calls[0]
    assert (
        command[command.index("-m") + 1] == "lilo.providers.modal.deployment_pool_app"
    )
    assert (
        json.loads(kwargs["env"][deployment_records.POOL_CONFIG_ENV])["generation"]
        == row.generation
    )


def test_frontend_uses_deployed_trainer_without_building_it(monkeypatch):
    row = deployment()
    calls = []
    monkeypatch.setattr(
        deployment_apps,
        "build_trainer_app",
        lambda *a, **k: pytest.fail("frontend must not rebuild trainer"),
    )
    monkeypatch.setattr(
        modal.Function,
        "from_name",
        lambda *a, **k: calls.append((a, k)) or "remote-trainer",
    )
    definition = deployment_apps.definition_from_spec(row)
    assert definition.ENGINE_FUNCTION == "remote-trainer"
    assert calls[0][0] == (row.trainer_app_name, "trainer")


@pytest.mark.parametrize("kind", ["lora", "full"])
def test_missing_pool_uses_saved_provisioner(monkeypatch, kind):
    from lilo.providers.modal import fft_pool, lora_pool

    row = deployment("qwen35-9b-lora-16k" if kind == "lora" else "qwen35-4b-fft-64k")
    monkeypatch.setenv(deployment_records.MANIFEST_ENV, json.dumps([row.model_dump()]))
    module = lora_pool if kind == "lora" else fft_pool
    spec = (
        LoraPoolSpec(row.definition_id)
        if kind == "lora"
        else FFTPoolSpec.base(row.definition_id)
    )

    class MissingPool:
        def __init__(self, *args):
            pass

        def gateway_url(self):
            raise modal.exception.NotFoundError("not deployed")

    monkeypatch.setattr(module, "ModalFlashPool", MissingPool)
    calls = []
    monkeypatch.setattr(
        modal.Function,
        "from_name",
        lambda name, function, **kw: (
            calls.append((name, function))
            or SimpleNamespace(remote=lambda record, pool: "https://saved-runtime")
        ),
    )
    monkeypatch.setattr(
        module.subprocess, "run", lambda *a, **k: pytest.fail("frontend rebuilt pool")
    )
    assert module.deploy_pool(spec) == "https://saved-runtime"
    assert calls == [(row.inference_app_name, "provision")]


def test_provisioner_rejects_wrong_settings_and_uses_saved_record(
    builders, monkeypatch
):
    row = deployment()
    app, provision = deployment_apps.build_inference_app(row, image="test")
    assert app.name == row.inference_app_name
    calls = []
    monkeypatch.setattr(
        deployment_apps,
        "deploy_lora",
        lambda pool, *, record: calls.append((pool, record)) or "https://pool",
    )
    pool = LoraPoolSpec(row.definition_id)
    assert provision(row.model_dump_json(), pool.as_dict()) == "https://pool"
    assert calls[0][1].inference_hash == row.inference_hash
    changed = row.model_copy(deep=True)
    changed.inference_release = "2"
    with pytest.raises(ValueError, match="inference settings"):
        provision(changed.model_dump_json(), pool.as_dict())


@pytest.mark.parametrize("role", ["trainer", "inference"])
def test_real_worker_entrypoint_constructs_offline(monkeypatch, role):
    import os
    import subprocess
    import sys

    row = deployment()
    env = {
        **os.environ,
        "LILO_WORKER_DEPLOYMENT": row.model_dump_json(),
        "LILO_WORKER_ROLE": role,
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import modal
from lilo.providers.modal import deployment_apps, deployment_records
deployment_apps.image_for = lambda backend: modal.Image.debian_slim()
import lilo.providers.modal.deployment_worker_app as worker
assert worker.app.name.startswith("lilo-")
print(worker.app.name)
""",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert f"lilo-{role}-" in result.stdout


def test_spawn_passes_job_configuration_to_saved_trainer(monkeypatch):
    import importlib

    app = importlib.import_module("lilo.providers.modal.app")
    row = deployment()
    calls = []

    async def spawn(instance_id, config_json):
        calls.append((instance_id, config_json))
        return SimpleNamespace(object_id="call-id")

    async def no_error(definition_id):
        return None

    definition = SimpleNamespace(
        RESOLVED=row,
        ENGINE_FUNCTION=SimpleNamespace(spawn=SimpleNamespace(aio=spawn)),
    )
    monkeypatch.setattr(app, "deployment_error", no_error)
    monkeypatch.setattr(app, "module_for", lambda _: definition)
    assert asyncio.run(app._spawn_engine(row.definition_id, "instance")) == "call-id"
    assert calls == [("instance", row.model_dump_json())]


def test_declared_compute_settings_reach_modal(builders):
    base = deployment().spec
    spec = replace(
        base,
        trainer=replace(
            base.trainer,
            timeout_s=90,
            compute=replace(base.trainer.compute, cpu=12, memory_mib=123456),
        ),
        inference=replace(
            base.inference,
            startup_timeout_s=90,
            min_replicas=1,
            max_replicas=3,
            compute=replace(base.inference.compute, cpu=6, memory_mib=45000),
        ),
    )
    row = DeploymentRecord.create(spec, revision="a" * 40)
    trainer_app, _ = deployment_apps.build_trainer_app(row, image="test")
    trainer, _ = trainer_app.functions["trainer"]
    assert (trainer["cpu"], trainer["memory"], trainer["timeout"]) == (12, 123456, 90)
    pool_app, _ = deployment_apps.build_rollout_app(
        row, LoraPoolSpec(row.definition_id), image="test"
    )
    server, _ = pool_app.servers["Server"]
    assert (server["cpu"], server["memory"], server["startup_timeout"]) == (
        6,
        45000,
        90,
    )
    assert (server["min_containers"], server["max_containers"]) == (1, 3)


def test_multinode_trainer_uses_cluster_launcher(builders, monkeypatch):
    row = deployment("qwen38-27b-lora-256k")
    clusters = []

    def clustered(nodes, *, rdma):
        clusters.append((nodes, rdma))

        def decorate(fn):
            return fn

        return decorate

    monkeypatch.setattr(modal.experimental, "clustered", clustered)
    app, _ = deployment_apps.build_trainer_app(row, image="test")
    settings, _ = app.functions["trainer"]
    assert clusters == [(2, True)]
    assert settings["gpu"] == "H200:8"
    assert settings["experimental_options"] == {"efa_enabled": True}
    calls = []
    monkeypatch.setattr(deployment_apps, "shared_kv", lambda: "store")
    monkeypatch.setattr(
        deployment_apps,
        "volumes_for",
        lambda _: {"/assets": SimpleNamespace(reload=lambda: None)},
    )
    monkeypatch.setattr(
        deployment_apps,
        "start_trainer_cluster",
        lambda nodes, **kwargs: "10.0.0.1:6379",
    )
    monkeypatch.setattr(
        deployment_apps, "run_engine_with_backend", lambda *a, **kw: calls.append(kw)
    )
    deployment_apps.run_trainer(row, "instance")
    assert calls[0]["backend_env"]["LILO_RAY_ADDRESS"] == "10.0.0.1:6379"
    assert (
        json.loads(calls[0]["backend_env"]["LILO_BACKEND_CONFIG"])["miles"][
            "actor_num_nodes"
        ]
        == 2
    )
    monkeypatch.setattr(deployment_apps, "start_trainer_cluster", lambda *a, **k: None)
    deployment_apps.run_trainer(row, "worker")
    assert len(calls) == 1


def test_launchers_do_not_reparse_backend_config(builders, monkeypatch):
    import lilo.backends.deployment as backend

    row = deployment()

    def unexpected(*args, **kwargs):
        pytest.fail("launcher must use the saved resolved settings")

    monkeypatch.setattr(backend, "backend_config", unexpected)
    monkeypatch.setattr(backend, "serving_options", unexpected)
    deployment_apps.build_trainer_app(row, image="test")
    deployment_apps.build_rollout_app(
        row, LoraPoolSpec(row.definition_id), image="test"
    )
    deployment_apps.definition_from_spec(row, register_trainer=False)
