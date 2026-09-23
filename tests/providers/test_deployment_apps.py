import asyncio
import json
from types import SimpleNamespace

import modal
import pytest

from lilo.deployments import load, config_path, DeploymentRecord
from lilo.providers.modal import deployment_apps
from lilo.providers.modal.fft_pool import FFTPoolSpec
from lilo.providers.modal.lora_pool import LoraPoolSpec


def deployment(preset="qwen35-9b-lora-16k"):
    return DeploymentRecord.create(
        load(config_path(preset)), revision="a" * 40, implementation="runtime"
    )


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
    row.spec.deployment['storage']['checkpoints'] = "test-custom-checkpoints"
    image = object()
    app, trainer = deployment_apps.build_trainer_app(row, image=image)
    declaration, _ = app.functions[row.definition_id]
    assert declaration["gpu"] == "H100:4"
    assert declaration["region"] == "us-west"
    assert declaration["max_containers"] == 1
    assert declaration["single_use_containers"] is True
    assert declaration["image"] is image
    calls = []
    reloaded = []
    from lilo.providers.modal import serve, kv

    monkeypatch.setattr(kv, "shared_kv", lambda: "store")
    monkeypatch.setattr(
        deployment_apps,
        "volumes_for",
        lambda spec: {"/assets": SimpleNamespace(reload=lambda: reloaded.append(True))},
    )
    monkeypatch.setattr(
        serve,
        "run_engine_with_backend",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    trainer("instance-a")
    args, kwargs = calls[0]
    assert args == ("store", f"lilo.backends.{backend}:build_executor")
    assert kwargs["max_models"] == clients
    assert kwargs["nproc"] == nproc
    assert kwargs["backend_env"]["LILO_CHECKPOINT_VOLUME"] == "test-custom-checkpoints"
    assert kwargs["backend_env"]["LILO_BASE_MODEL_REVISION"] == "a" * 40
    config = json.loads(kwargs["backend_env"]["LILO_BACKEND_CONFIG"])
    assert config[row.spec.trainer['backend']]["hf_checkpoint"] == row.asset_path
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
    assert settings["gpu"] == row.spec.inference['resources']['gpu']
    assert settings["min_containers"] == 0
    assert settings["target_concurrency"] == 16
    assert settings["compute_region"] == "us-west"
    from lilo.inference import serving
    import subprocess

    calls, commands, stops = [], [], []
    process = object()
    monkeypatch.setattr(
        subprocess, "Popen", lambda argv, **kw: (commands.append(argv) or process)
    )
    monkeypatch.setattr(serving, "wait_http", lambda *args: None)
    monkeypatch.setattr(serving, "supervise_children", lambda *args: None)
    monkeypatch.setattr(
        serving,
        "start_lora_sidecar",
        lambda **kw: (calls.append(("lora", kw)) or process),
    )
    monkeypatch.setattr(
        serving,
        "start_fft_sidecar",
        lambda **kw: (calls.append(("fft", kw)) or process),
    )
    monkeypatch.setattr(serving, "terminate", stops.append)
    replica = server()
    replica.start()
    assert commands[0][2] == "lilo.inference.native_sglang"
    assert commands[0][3] == row.asset_path
    native = json.loads(commands[0][4])
    assert native["context_length"] == row.spec.model['max_context_length']
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
    monkeypatch.setenv(deployment_apps.MANIFEST_ENV, json.dumps([row.model_dump()]))
    env = deployment_apps.pool_environment(row.definition_id)
    assert json.loads(env[deployment_apps.POOL_CONFIG_ENV])["generation"] == row.generation
    with pytest.raises(ValueError, match="missing recorded"):
        deployment_apps.pool_environment("yaml_missing_123")
    with pytest.raises(ValueError, match="missing recorded"):
        deployment_apps.pool_environment("unconfigured-python-definition")


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
    env = {**os.environ, deployment_apps.MANIFEST_ENV: json.dumps([row.model_dump()])}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib, sys, modal
from lilo.providers.modal import deployment_apps
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
    changed.spec.routing['default'] = not first.spec.routing['default']
    changed.spec.routing['sampling_default'] = True
    new_bytes = serialize(deployment_apps.build_trainer_app(changed, image="test")[1])
    assert new_bytes == old_bytes
    assert first.active is True and first.spec.routing['default'] is True
    changed.spec.trainer['resources']['gpu'] = "H200:4"
    assert serialize(deployment_apps.build_trainer_app(changed, image="test")[1]) != old_bytes


@pytest.mark.parametrize("kind", ["lora", "full"])
def test_pool_launch_uses_only_generic_yaml_app(monkeypatch, kind):
    from lilo.providers.modal import fft_pool, lora_pool

    row = deployment("qwen35-9b-lora-16k" if kind == "lora" else "qwen35-4b-fft-64k")
    monkeypatch.setenv(deployment_apps.MANIFEST_ENV, json.dumps([row.model_dump()]))
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
    assert module.deploy_pool(spec) == "https://pool"
    command, kwargs = calls[0]
    assert command[command.index("-m") + 1] == "lilo.providers.modal.deployment_pool_app"
    assert (
        json.loads(kwargs["env"][deployment_apps.POOL_CONFIG_ENV])["generation"]
        == row.generation
    )
