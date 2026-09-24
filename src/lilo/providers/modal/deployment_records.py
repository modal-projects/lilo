"""Saved records used to route worker and pool requests."""

import json
import os

import modal

from lilo.deployments import DeploymentRecord

MANIFEST_ENV = "LILO_DEPLOYMENT_MANIFEST"
POOL_CONFIG_ENV = "LILO_POOL_DEPLOYMENT"


def manifest_from_env():
    data = os.environ.get(MANIFEST_ENV)
    if not data:
        raise ValueError(
            "Missing deployment manifest. Use lilo deploy with your Python config files."
        )
    rows = json.loads(data)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Deployment manifest must be a nonempty list")
    return [DeploymentRecord.model_validate(row) for row in rows]


def deployed_manifest(frontend, environment=None):
    """Read the configuration carried by the currently deployed frontend."""
    function = modal.Function.from_name(
        frontend, "deployment_manifest", environment_name=environment
    )
    try:
        function.hydrate()
    except modal.exception.NotFoundError:
        return []
    return function.remote()


def pool_deployment(definition_id):
    for row in manifest_from_env():
        if row.definition_id == definition_id:
            return row
    return None


def provision_pool(record, pool):
    """Ask the saved inference app to create a pool using its original code."""
    provision = modal.Function.from_name(
        record.inference_app_name,
        "provision",
        environment_name=record.platform["modal"]["environment"],
    )
    return provision.remote(record.model_dump_json(), pool.as_dict())
