"""The SGLang worker receives its resolved constructor options without argparse."""

import json
import runpy
import sys
from types import ModuleType

from lilo.deployments import DeploymentRecord, config_path, load


def test_sglang_constructor_receives_recorded_settings(monkeypatch):
    row = DeploymentRecord.create(
        load(config_path("qwen35-9b-lora-16k")), revision="a" * 40
    )
    calls = []

    class ServerArgs:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.log_level = "info"

    modules = {
        "sglang": {"__path__": []},
        "sglang.srt": {"__path__": []},
        "sglang.launch_server": {"run_server": lambda args: calls.append("run")},
        "sglang.srt.plugins": {"load_plugins": lambda: calls.append("plugins")},
        "sglang.srt.server_args": {"ServerArgs": ServerArgs},
        "sglang.srt.utils": {"kill_process_tree": lambda *a, **k: calls.append("stop")},
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        sys, "argv", ["sglang", row.asset_path, json.dumps(row.inference_settings)]
    )
    runpy.run_module("lilo.inference.sglang", run_name="__main__")
    assert calls == [
        "plugins",
        {
            "model_path": row.asset_path,
            "host": "127.0.0.1",
            "port": 8001,
            **row.inference_settings,
        },
        "run",
        "stop",
    ]
