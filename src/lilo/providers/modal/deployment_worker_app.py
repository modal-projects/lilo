"""Deploy one trainer or inference provisioner; the frontend references its name."""

import os

from lilo.deployments import DeploymentRecord

from .deployment_apps import build_inference_app, build_trainer_app

record = DeploymentRecord.model_validate_json(os.environ["LILO_WORKER_DEPLOYMENT"])
if record.miles_commit:
    os.environ["LILO_MILES_COMMIT"] = record.miles_commit
if os.environ["LILO_WORKER_ROLE"] == "trainer":
    app, trainer = build_trainer_app(record)
elif os.environ["LILO_WORKER_ROLE"] == "inference":
    app, provision = build_inference_app(record)
else:
    raise ValueError("unknown worker role")
