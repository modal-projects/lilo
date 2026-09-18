from functools import partial
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def loss_scaling(monkeypatch):
    modules = {}
    for name in (
        "megatron",
        "megatron.core",
        "megatron.core.pipeline_parallel",
        "megatron.core.pipeline_parallel.schedules",
        "miles",
        "miles.backends",
        "miles.backends.megatron_utils",
        "miles.backends.megatron_utils.model",
        "miles.utils",
        "miles.utils.multi_lora",
    ):
        module = modules[name] = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(modules[parent], child, module)

    calls = []

    def miles_loss(
        args,
        batch,
        num_microbatches,
        logits,
        apply_megatron_loss_scaling=False,
        num_rollouts=None,
    ):
        calls.append(("miles", num_microbatches, num_rollouts))
        return logits * num_microbatches if apply_megatron_loss_scaling else logits

    def megatron_loss(
        model,
        output_tensor,
        loss_func,
        config,
        vp_stage,
        collect_non_loss_data,
        num_microbatches,
        forward_data_store,
        cp_group_size=None,
        is_last_stage=None,
    ):
        calls.append(("megatron", num_microbatches, cp_group_size, is_last_stage))
        if collect_non_loss_data or loss_func is None:
            return output_tensor
        return loss_func(output_tensor) / num_microbatches

    schedules = modules["megatron.core.pipeline_parallel.schedules"]
    schedules.forward_step_calc_loss = megatron_loss
    modules["miles.backends.megatron_utils.model"].loss_function = miles_loss
    modules["miles.utils.multi_lora"].is_multi_lora_enabled = lambda args: args.lora
    path = Path(__file__).parents[2] / "src/lilo/backends/miles_runtime/loss_scaling.py"
    spec = importlib.util.spec_from_file_location("test_loss_scaling_impl", path)
    implementation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(implementation)
    implementation.configure_deterministic_loss_scaling()
    return implementation, schedules, miles_loss, calls


def invoke(
    schedules,
    miles_loss,
    count,
    value=2.0,
    *,
    enabled=True,
    lora=True,
    per_token=False,
    loss_fn="importance_sampling",
    collect=False,
    config=None,
):
    args = SimpleNamespace(
        lilo_deterministic_attention=enabled,
        calculate_per_token_loss=per_token,
        lora=lora,
    )
    batch = {"loss_fn": loss_fn}
    callback = partial(
        miles_loss,
        args,
        batch,
        count,
        apply_megatron_loss_scaling=True,
        num_rollouts=256,
    )
    result = schedules.forward_step_calc_loss(
        None,
        value,
        callback,
        config or SimpleNamespace(),
        None,
        collect,
        count,
        [],
        cp_group_size=1,
        is_last_stage=True,
    )
    assert callback.args[2] == count  # Do not mutate the scheduler's callback.
    return result


def test_only_scalar_normalizers_change_and_install_is_idempotent(loss_scaling):
    implementation, schedules, miles_loss, calls = loss_scaling
    installed = schedules.forward_step_calc_loss
    implementation.configure_deterministic_loss_scaling()
    assert schedules.forward_step_calc_loss is installed
    for count in (55, 173):
        assert invoke(schedules, miles_loss, count) == 2.0
        assert calls[-2:] == [("megatron", 1, 1, True), ("miles", 1, 256)]


@pytest.mark.parametrize(
    "options",
    [
        {"enabled": False},
        {"lora": False},
        {"per_token": True},
        {"loss_fn": "policy_loss"},
        {"collect": True},
    ],
)
def test_other_paths_keep_original_normalizers(loss_scaling, options):
    _, schedules, miles_loss, calls = loss_scaling
    invoke(schedules, miles_loss, 55, **options)
    assert calls[0] == ("megatron", 55, 1, True)
    if not options.get("collect"):
        assert calls[1] == ("miles", 55, 256)


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_moe_experts", 8),
        ("mtp_num_layers", 1),
        ("experimental_attention_variant", "dsa"),
        ("experimental_attention_variant_loss_scale_func", object()),
    ],
)
def test_auxiliary_losses_are_rejected_before_scaling(loss_scaling, field, value):
    _, schedules, miles_loss, calls = loss_scaling
    with pytest.raises(ValueError, match="no auxiliary losses"):
        invoke(schedules, miles_loss, 55, config=SimpleNamespace(**{field: value}))
    assert calls == []


def test_unrelated_loss_callback_is_unchanged(loss_scaling):
    _, schedules, _, calls = loss_scaling
    callback = partial(lambda multiplier, value: multiplier * value, 55)
    assert (
        schedules.forward_step_calc_loss(
            None,
            2.0,
            callback,
            SimpleNamespace(),
            None,
            False,
            55,
            [],
        )
        == 2.0
    )
    assert calls == [("megatron", 55, None, None)]


def test_autograd_parity_for_55_and_173_microbatches(loss_scaling):
    torch = pytest.importorskip("torch")
    _, schedules, miles_loss, _ = loss_scaling
    old_gradients, new_gradients = [], []
    for count in (55, 173):
        old = torch.tensor(1.0, requires_grad=True)
        ((old * count) / count).backward()
        old_gradients.append(old.grad.item())
        new = torch.tensor(1.0, requires_grad=True)
        invoke(schedules, miles_loss, count, value=new).backward()
        new_gradients.append(new.grad.item())
    assert old_gradients == [0.9999999403953552, 1.0]
    assert new_gradients == [1.0, 1.0]


def test_accumulation_and_adam_match_across_shared_work_units(loss_scaling):
    torch = pytest.importorskip("torch")
    _, schedules, miles_loss, _ = loss_scaling
    parameters, gradients = [], []
    for count in (55, 173):
        parameter = torch.tensor([0.1, -0.3], requires_grad=True)
        optimizer = torch.optim.Adam([parameter], lr=1e-5)
        for step in range(2):
            optimizer.zero_grad()
            # Both schedules accumulate this client's same 55 microbatches;
            # other clients contribute the extra work in the shared schedule.
            for index in range(55):
                loss = (parameter.exp() * torch.tensor([0.375, -0.125])).sum()
                invoke(schedules, miles_loss, count, value=loss).backward()
            gradients.append(parameter.grad.clone())
            optimizer.step()
        parameters.append(parameter.detach().clone())
    assert torch.equal(gradients[0], gradients[2])
    assert torch.equal(gradients[1], gradients[3])
    assert torch.equal(parameters[0], parameters[1])
