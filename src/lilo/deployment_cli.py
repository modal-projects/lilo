"""Operator CLI for YAML deployments. Only `deploy` provisions Modal resources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

import yaml

from lilo.deployments import (
    ResolvedDeployment,
    load,
    preset_path,
    resolve,
    validate_frontend,
)


def implementation_fingerprint(miles_commit: str | None) -> str:
    """Hash shipped source and dependency declarations, independent of Git checkout."""
    from importlib.metadata import requires

    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    digest.update(json.dumps([miles_commit, sorted(requires("lilo") or [])]).encode())
    return digest.hexdigest()


def compile_configs(paths):
    specs = [load(path) for path in paths]
    validate_frontend(specs)
    from lilo.providers.modal.miles_revision import resolve_miles_commit

    miles_commit = resolve_miles_commit()
    implementation = implementation_fingerprint(miles_commit)
    resolved = []
    for spec in specs:
        revision = spec.model.revision
        if not re.fullmatch(r"[0-9a-f]{40,64}", revision):
            from huggingface_hub import HfApi

            revision = HfApi().model_info(spec.model.id, revision=revision).sha
        resolved.append(
            resolve(
                spec,
                revision=revision,
                implementation=implementation,
                miles_commit=miles_commit,
            )
        )
    return resolved


def retain_generations(previous, desired):
    """The command supplies the complete active set; old generations remain usable."""
    validate_frontend([row.spec for row in desired])
    expected = desired[0]
    for row in previous:
        if row.implementation != expected.implementation:
            raise ValueError(
                "This draft cannot rebuild retained generations with different Lilo/runtime code. Use a separate frontend for a code upgrade; YAML-only changes can retain existing generations."
            )
        if (
            row.spec.deployment != expected.spec.deployment
            or row.spec.lifecycle != expected.spec.lifecycle
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


def deploy(desired):
    """Serialize operator applies and retain interrupted attempts for safe recovery."""
    import modal
    from lilo.providers.modal.yaml_apps import MANIFEST_ENV

    if sys.version_info[:2] != (3, 12):
        raise ValueError(
            "YAML deployment requires Python 3.12 to match the serialized GPU runtime images"
        )
    settings = desired[0].spec.deployment
    registry = modal.Dict.from_name(
        f"{settings.frontend}-yaml-deployments",
        create_if_missing=True,
        environment_name=settings.modal.environment,
    )
    owner = uuid.uuid4().hex
    if not registry.put("apply_lock", owner, skip_if_exists=True):
        raise ValueError(
            f"An apply owns {settings.frontend}. If it was interrupted, confirm it has stopped before running lilo deployment unlock --frontend {settings.frontend}."
        )
    try:
        rows = registry.get("manifest", [])
        if not rows and not registry.get("pending", []):
            try:
                modal.App.lookup(
                    settings.frontend, environment_name=settings.modal.environment
                )
            except modal.exception.NotFoundError:
                pass
            else:
                raise ValueError(
                    "The frontend already exists without a YAML registry. Choose a new frontend name; an app with no deployment registry cannot be safely updated."
                )
        # A killed deploy may already have updated Modal. Keep its functions on retry.
        rows = {r["generation"]: r for r in [*rows, *registry.get("pending", [])]}
        manifest = retain_generations(
            [ResolvedDeployment.model_validate(row) for row in rows.values()], desired
        )
        data = [row.model_dump(mode="json") for row in manifest]
        env = {
            **os.environ,
            MANIFEST_ENV: json.dumps(data),
            "LILO_APP_NAME": settings.frontend,
        }
        if desired[0].miles_commit:
            env["LILO_MILES_COMMIT"] = desired[0].miles_commit
        command = [
            sys.executable,
            "-m",
            "modal",
            "deploy",
            "-m",
            "lilo.providers.modal.app",
        ]
        if settings.modal.environment:
            command += ["--env", settings.modal.environment]
        registry.put("pending", data)
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
        "deploy", help="Deploy the complete active YAML set behind one frontend"
    )
    apply.add_argument("files", nargs="+")
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
                print(
                    yaml.safe_dump(
                        load(preset_path(args.preset)).model_dump(mode="json"),
                        sort_keys=False,
                    ),
                    end="",
                )
            elif args.action == "validate":
                specs = [load(path) for path in args.files]
                validate_frontend(specs)
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
            deploy(compile_configs(args.files))
        else:
            import modal

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
