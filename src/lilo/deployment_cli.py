"""Operator CLI for Python deployments. Only `deploy` provisions Modal resources."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import uuid
from copy import deepcopy
from pathlib import Path

import modal
from huggingface_hub import HfApi

from lilo.backends.deployment import resolve_backend_settings
from lilo.deployments import (
    LIFECYCLE_FIELDS,
    PLATFORM_DEFAULTS,
    DeploymentRecord,
    config_path,
    load,
    validate_frontend,
)
from lilo.providers.modal.deployment_records import MANIFEST_ENV
from lilo.providers.modal.miles_revision import resolve_miles_commit


def compile_configs(paths, *, platform=None):
    specs = [load(path) for path in paths]
    validate_frontend(specs)

    miles_commit = (
        resolve_miles_commit()
        if any(spec.backend == "miles" for spec in specs)
        else None
    )
    records = []
    for spec in specs:
        revision = spec.revision
        if not re.fullmatch(r"[0-9a-fA-F]{40,64}", revision):
            revision = HfApi().model_info(spec.model, revision=revision).sha
            if not revision or not re.fullmatch(r"[0-9a-fA-F]{40,64}", revision):
                raise ValueError(
                    f"Hugging Face did not return a commit for {spec.model}"
                )
        records.append(
            DeploymentRecord.create(
                spec,
                platform=platform,
                revision=revision,
                miles_commit=miles_commit if spec.backend == "miles" else None,
            )
        )
    return records


def retain_generations(previous, desired):
    """The command supplies the complete active set; old generations remain usable."""
    validate_frontend([row.spec for row in desired])
    expected = desired[0]
    for row in previous:
        if row.platform != expected.platform or any(
            getattr(row.spec, field) != getattr(expected.spec, field)
            for field in LIFECYCLE_FIELDS
        ):
            raise ValueError(
                "Cannot change shared storage, secrets, region, or lifecycle while retaining generations; use a separate frontend."
            )
    ids = {row.definition_id for row in desired}
    return [
        *desired,
        *(
            row.model_copy(update={"active": False})
            for row in previous
            if row.definition_id not in ids
        ),
    ]


def select_worker_releases(desired, previous, refresh_trainers, refresh_inference):
    """Keep deployed code unless the operator explicitly requests a worker update."""
    names = {row.spec.name for row in desired}
    unknown = (set(refresh_trainers) | set(refresh_inference)) - names
    if unknown:
        raise ValueError(f"unknown configs to refresh: {sorted(unknown)}")
    active = {row.spec.name: row for row in previous if row.active}
    result = []
    for row in desired:
        old = active.get(row.spec.name, row)
        trainer = (
            uuid.uuid4().hex
            if row.spec.name in refresh_trainers
            else old.trainer_release
        )
        inference = (
            uuid.uuid4().hex
            if row.spec.name in refresh_inference
            else old.inference_release
        )
        result.append(row.with_releases(trainer, inference))
    return result


def worker_apps_ready(row, deployed):
    return row.trainer_app_name in deployed and row.inference_app_name in deployed


def deploy(desired, *, refresh_trainers=(), refresh_inference=()):
    """Serialize operator applies and retain interrupted attempts for safe recovery."""

    if sys.version_info[:2] != (3, 12):
        raise ValueError(
            "Python deployment requires Python 3.12 to match the serialized GPU runtime images"
        )
    settings = desired[0].platform
    registry = modal.Dict.from_name(
        f"{settings['frontend']}-yaml-deployments",
        create_if_missing=True,
        environment_name=settings["modal"]["environment"],
    )
    owner = uuid.uuid4().hex
    if not registry.put("apply_lock", owner, skip_if_exists=True):
        raise ValueError(
            f"An apply owns {settings['frontend']}. If it was interrupted, confirm it has stopped before running lilo deployment unlock --frontend {settings['frontend']}."
        )
    try:
        rows = registry.get("manifest", [])
        if not rows and not registry.get("pending", []):
            try:
                modal.App.lookup(
                    settings["frontend"],
                    environment_name=settings["modal"]["environment"],
                )
            except modal.exception.NotFoundError:
                pass
            else:
                raise ValueError(
                    "The frontend already exists without a deployment registry. Choose a new frontend name; an app with no deployment registry cannot be safely updated."
                )
        # Only complete worker pairs could have been exposed by a pending frontend.
        deployed = set(registry.get("worker_apps", []))
        pending = [
            row
            for row in registry.get("pending", [])
            if worker_apps_ready(DeploymentRecord.model_validate(row), deployed)
        ]
        rows = {r["generation"]: r for r in [*rows, *pending]}
        previous = [DeploymentRecord.model_validate(row) for row in rows.values()]
        desired = select_worker_releases(
            desired, previous, refresh_trainers, refresh_inference
        )
        manifest = retain_generations(previous, desired)
        data = [row.model_dump(mode="json") for row in manifest]
        env = {
            **os.environ,
            MANIFEST_ENV: json.dumps(data),
            "LILO_APP_NAME": settings["frontend"],
        }
        command = [
            sys.executable,
            "-m",
            "modal",
            "deploy",
            "-m",
            "lilo.providers.modal.app",
        ]
        if settings["modal"]["environment"]:
            command += ["--env", settings["modal"]["environment"]]
        registry.put("pending", data)
        # Deploy each worker app once. Retained apps keep their original code.
        deployed = set(registry.get("worker_apps", []))
        for row in manifest:
            for role, app_name in (
                ("trainer", row.trainer_app_name),
                ("inference", row.inference_app_name),
            ):
                if app_name in deployed:
                    continue
                if not row.active:
                    raise ValueError(
                        f"Retained worker {app_name} is missing; restore its original deployment."
                    )
                # Recover a crash after Modal succeeded but before the registry write.
                try:
                    modal.App.lookup(
                        app_name, environment_name=settings["modal"]["environment"]
                    )
                except modal.exception.NotFoundError:
                    pass
                else:
                    deployed.add(app_name)
                    registry.put("worker_apps", sorted(deployed))
                    continue
                worker_env = {
                    **os.environ,
                    "LILO_WORKER_DEPLOYMENT": row.model_dump_json(),
                    "LILO_WORKER_ROLE": role,
                }
                if row.miles_commit:
                    worker_env["LILO_MILES_COMMIT"] = row.miles_commit
                worker_command = [
                    sys.executable,
                    "-m",
                    "modal",
                    "deploy",
                    "-m",
                    "lilo.providers.modal.deployment_worker_app",
                ]
                if settings["modal"]["environment"]:
                    worker_command += ["--env", settings["modal"]["environment"]]
                subprocess.run(worker_command, check=True, env=worker_env)
                deployed.add(app_name)
                registry.put("worker_apps", sorted(deployed))
        subprocess.run(command, check=True, env=env)
        registry.put("manifest", data)
        registry.pop("pending", None)
    finally:
        if registry.get("apply_lock") == owner:
            registry.pop("apply_lock", None)


def parser():
    result = argparse.ArgumentParser(prog="lilo")
    commands = result.add_subparsers(dest="command", required=True)
    config = commands.add_parser("config").add_subparsers(dest="action", required=True)
    init = config.add_parser("init")
    init.add_argument("--preset", required=True)
    for name in ("validate", "resolve"):
        cmd = config.add_parser(name)
        cmd.add_argument("files", nargs="+")
        if name == "resolve":
            cmd.add_argument("--output")
    apply = commands.add_parser(
        "deploy",
        help="Deploy the complete active Python config set behind one frontend",
    )
    apply.add_argument("files", nargs="+")
    apply.add_argument("--app", default=PLATFORM_DEFAULTS["frontend"])
    apply.add_argument("--env")
    apply.add_argument("--region", default=PLATFORM_DEFAULTS["modal"]["region"])
    apply.add_argument(
        "--refresh-trainer", action="append", default=[], metavar="CONFIG_NAME"
    )
    apply.add_argument(
        "--refresh-inference", action="append", default=[], metavar="CONFIG_NAME"
    )
    management = commands.add_parser("deployment").add_subparsers(
        dest="action", required=True
    )
    for name in ("unlock", "retry"):
        cmd = management.add_parser(name)
        cmd.add_argument("--frontend", required=True)
        cmd.add_argument("--env")
        if name == "retry":
            cmd.add_argument("definition_id")
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        if args.command == "config":
            if args.action == "init":
                module = config_path(args.preset).stem
                if not config_path(args.preset).is_file():
                    raise ValueError(f"unknown example config: {args.preset}")
                print(
                    f'from lilo.configs.{module} import Config as Parent\n\n\nclass Config(Parent):\n    name = "my-model"\n\n\nconfig = Config()'
                )
            elif args.action == "validate":
                specs = [load(path) for path in args.files]
                validate_frontend(specs)
                for spec in specs:
                    resolve_backend_settings(spec, "/assets/pending")
                print(
                    f"Validated {len(specs)} deployment(s). Backend integration settings are checked when preparing trainers and pools; native options are checked at worker startup."
                )
            else:
                output = (
                    json.dumps(
                        [
                            row.model_dump(mode="json")
                            for row in compile_configs(args.files)
                        ],
                        indent=2,
                    )
                    + "\n"
                )
                if args.output:
                    Path(args.output).write_text(output)
                else:
                    print(output, end="")
        elif args.command == "deploy":
            platform = deepcopy(PLATFORM_DEFAULTS)
            platform["frontend"] = args.app
            platform["modal"].update(environment=args.env, region=args.region)
            deploy(
                compile_configs(args.files, platform=platform),
                refresh_trainers=args.refresh_trainer,
                refresh_inference=args.refresh_inference,
            )
        else:
            if args.action == "unlock":
                registry = modal.Dict.from_name(
                    f"{args.frontend}-yaml-deployments", environment_name=args.env
                )
                registry.pop("apply_lock", None)
            else:
                modal.Function.from_name(
                    args.frontend, "clear_deployment_failure", environment_name=args.env
                ).remote(args.definition_id)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        cli.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
