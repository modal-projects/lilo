"""CPU checks for the E2E harness's failure gates and TITO training payloads."""

import importlib.util
import pytest
import threading
from pathlib import Path
from types import SimpleNamespace

from lilo.backends.miles_runtime.data import _datum_row

spec = importlib.util.spec_from_file_location(
    "miles_multi_lora_e2e",
    Path(__file__).parents[2] / "scripts" / "e2e_miles_multi_lora.py",
)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


def placement(model_id, instance="engine-1", boot="boot-1"):
    return dict(
        model_id=model_id,
        engine_instance_id=instance,
        engine_boot_id=boot,
        engine_definition_id="miles-test",
    )


def test_two_clients_must_share_instance_and_boot():
    records = [placement("a"), placement("b")]
    assert harness.validate_placements(records, ["a", "b"], "miles-test") == (
        "engine-1",
        "boot-1",
    )
    for other in (placement("b", instance="engine-2"), placement("b", boot="boot-2")):
        with pytest.raises(RuntimeError, match="different trainer"):
            harness.validate_placements([records[0], other], ["a", "b"], "miles-test")
    with pytest.raises(RuntimeError, match="missing placement"):
        harness.validate_placements([records[0], None], ["a", "b"], "miles-test")
    with pytest.raises(RuntimeError, match="wrong trainer"):
        harness.validate_placements(records, ["a", "b"], "other")


def test_tito_rl_payload_satisfies_actual_miles_shift_contract():
    datum = harness.make_datum([11, 12, 13], [14, 15], logprobs=[-0.5, -0.7], advantage=-0.5)
    row = _datum_row(datum, "importance_sampling", 0)
    assert row["tokens"] == [11, 12, 13, 14, 15]
    assert row["advantages"] == [0, 0, -0.5, -0.5]
    assert row["sampling_logprobs"] == pytest.approx([0, 0, -0.5, -0.7])
    with pytest.raises(RuntimeError, match="length mismatch"):
        harness.make_datum([11], [12, 13], logprobs=[-0.5], advantage=1)
    with pytest.raises(RuntimeError, match="nonfinite"):
        harness.make_datum([11], [12], logprobs=[float("nan")], advantage=1)


def test_parallel_dispatch_reaches_both_clients_before_waiting():
    barrier = threading.Barrier(2)

    def client(value):
        barrier.wait(timeout=2)
        return value

    assert harness.parallel([lambda: client("a"), lambda: client("b")]) == ["a", "b"]


@pytest.mark.parametrize("answer", [r"work \\boxed{1,234}", "work\n#### 1234", "1234.0"])
def test_numeric_grading(answer):
    assert harness.numeric_answer(answer) == 1234


@pytest.mark.parametrize("answer", ["NaN", "Infinity", "maybe 1234", r"\\boxed{\\frac{1}{2}}"])
def test_grading_does_not_accept_invalid_or_unparsed_answers(answer):
    assert harness.numeric_answer(answer) is None


def test_nonfinite_and_mismatched_logprobs_cannot_pass_parity():
    with pytest.raises(RuntimeError, match="nonfinite"):
        harness.max_error([float("nan")], [1])
    with pytest.raises(RuntimeError, match="shape"):
        harness.max_error([1], [1, 2])


def test_rollout_parity_excludes_prompt_and_keeps_zero_logprob_response():
    datum = harness.make_datum([11, 12, 13], [14, 15], logprobs=[-0.5, 0.0], advantage=0)
    forward = SimpleNamespace(
        loss_fn_outputs=[{"logprobs": SimpleNamespace(data=[-100.0, -100.0, -0.7, 0.1])}],
        metrics={"loss:mean": 0.0},
    )
    optimizer = SimpleNamespace(metrics={"grad_norm:mean": 0.0})
    training = SimpleNamespace(
        model_id="a",
        forward_backward=lambda *_: SimpleNamespace(result=lambda **_: forward),
        optim_step=lambda *_: SimpleNamespace(result=lambda **_: optimizer),
    )
    result = harness.train_step(training, [datum], "importance_sampling", 1e-4, response_lengths=[2])
    parity = result["rollout_trainer_parity"]
    assert parity["tokens"] == 2
    assert parity["mean_signed"] == pytest.approx(-0.05)
    assert parity["mean_abs"] == pytest.approx(0.15)
    assert parity["max_abs"] == pytest.approx(0.2)


def test_publication_uses_tinker_export_path_and_checks_ownership():
    calls = []

    class Training:
        model_id = "adapter-a"

        def save_weights_for_sampler(self, name):
            calls.append(("export", name))
            return SimpleNamespace(
                result=lambda timeout: SimpleNamespace(path="tinker://adapter-a/sampler_weights/step-0")
            )

        def create_sampling_client(self, path):
            calls.append(("sampling_client", path))
            return "sampler"

    class Probe:
        def artifact(self, path, model_id):
            calls.append(("verify", path, model_id))
            return {"path": path, "model_id": model_id, "publish_version": 1}

    sampler, artifact = harness.publish(Training(), Probe(), "step-0")
    assert sampler == "sampler"
    assert artifact["model_id"] == "adapter-a"
    assert [call[0] for call in calls] == ["export", "verify", "sampling_client"]
