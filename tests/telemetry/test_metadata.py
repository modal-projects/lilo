import asyncio

from lilo.control_plane import ControlPlane
from lilo.control_plane.keys import sampler_artifact_key
from lilo.providers.local import InMemoryKeyValueStore, LocalEnginePlatform
from lilo.telemetry.metadata import common_tags, experiment_tags
from tests.control_plane.test_sampler_exports import (
    BASE_MODEL,
    DEFINITION,
    VersionedExecutor,
    export_request,
)


def test_only_explicit_bounded_experiment_labels_are_exported():
    assert experiment_tags(
        {"run_id": "run", "attempt_id": "attempt", "secret": "PRIVATE"}
    ) == {"lilo.run_id": "run", "lilo.run_attempt_id": "attempt"}
    assert experiment_tags({"run_id": "x" * 257, "attempt_id": 42}) == {}
    assert common_tags(
        [
            {"lilo.run_id": "run", "lilo.run_attempt_id": "a"},
            {"lilo.run_id": "run", "lilo.run_attempt_id": "b"},
        ]
    ) == {"lilo.run_id": "run"}
    assert common_tags([{"lilo.run_id": "a"}, {"lilo.run_id": "b"}]) == {}


def test_sampler_artifact_and_session_preserve_experiment_labels():
    async def run():
        plane = ControlPlane(
            InMemoryKeyValueStore(),
            LocalEnginePlatform(DEFINITION, VersionedExecutor),
            clock=lambda: 100.0,
        )
        session = await plane.create_session()
        creation = await plane.create_model(
            session_id=session.session_id,
            model_seq_id=0,
            definition_id=DEFINITION,
            spec={
                "base_model": BASE_MODEL,
                "user_metadata": {
                    "run_id": "run",
                    "attempt_id": "attempt",
                    "private": "SECRET",
                },
            },
        )
        await plane.retrieve(creation.request_id)
        rid = await plane.submit_sampler_export(export_request(creation.model.model_id))
        result = await plane.retrieve(rid, timeout=1.0)
        artifact = await plane.get_sampler_artifact(result.result["path"])
        assert artifact.telemetry_tags == {
            "lilo.run_id": "run",
            "lilo.run_attempt_id": "attempt",
        }
        sampling = await plane.create_sampling_session(
            session_id=session.session_id,
            sampling_session_seq_id=0,
            model_path=artifact.model_path,
        )
        assert sampling.telemetry_tags == artifact.telemetry_tags
        # Adding labels must not change the identity of older publications.
        versioned_key = sampler_artifact_key(
            plane._latest_sampler_model_path(creation.model.model_id, 7)
        )
        old = await plane.kv.get(versioned_key)
        old.pop("telemetry_tags")
        await plane.kv.put(versioned_key, old)
        rid = await plane.submit_sampler_export(
            export_request(creation.model.model_id, seq_id=2, path="second")
        )
        assert (await plane.retrieve(rid, timeout=1.0)).result is not None

    asyncio.run(run())
