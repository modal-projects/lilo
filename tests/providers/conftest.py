"""Shared provider tests construct the app from an explicit offline manifest."""

import json
import os

from lilo.deployments import load, config_path, DeploymentRecord

os.environ.setdefault(
    "LILO_DEPLOYMENT_MANIFEST",
    json.dumps(
        [
            DeploymentRecord.create(
                load(config_path(name)), revision="a" * 40, implementation="tests"
            ).model_dump(mode="json")
            for name in ("qwen35-9b-fft-64k", "qwen35-9b-lora-16k")
        ]
    ),
)
