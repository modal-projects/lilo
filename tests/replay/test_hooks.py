"""Check our adapter hooks at the Miles boundary without loading Megatron."""

from types import SimpleNamespace

import pytest
import torch

from lilo.backends.miles_runtime.replay import install_replay_hooks


@pytest.fixture
def miles():
    manager = SimpleNamespace(
        enabled=True,
        stage="fallthrough",
        replays=[[]],
        data_key="rollout_routed_experts",
        if_sp_region=True,
        replay_indices_are_token_positions=False,
        register_replay_list_func=object(),
    )
    manager.clear_all = lambda: manager.replays[0].clear()
    events = []

    def fill(**kwargs):
        assert manager.enabled
        assert manager.stage == "replay_backward"
        assert not manager.replays[0]
        manager.replays[0].append(kwargs["rollout_data"].pop(manager.data_key))
        events.append("fill")

    lora = SimpleNamespace()
    lora.get_data_iterator = lambda *args: ([object()], [1])

    def run(args, batch_id, model, data, **kwargs):
        lora.get_data_iterator(args, model, data)
        events.append((manager.enabled, len(manager.replays[0])))
        if kwargs.get("fail"):
            raise RuntimeError("forward failed")
        return "result"

    lora.run_forward_backward = run
    model = SimpleNamespace()
    model.get_batch = lambda iterator, keys: keys
    logit = SimpleNamespace()
    logit.build_local_sampling_mask = lambda *args, **kwargs: None
    loss = SimpleNamespace()
    loss._target_logprobs = lambda *args: "ordinary"
    loss.get_log_probs_and_entropy = lambda logits, **kwargs: {"log_probs": kwargs}
    dependencies = dict(
        megatron_model=model,
        lora_model=lora,
        logit_processors=logit,
        tinker_losses=loss,
        fill_replay_data=fill,
        manager=manager,
    )
    install_replay_hooks(**dependencies)
    return SimpleNamespace(
        manager=manager,
        events=events,
        lora=lora,
        model=model,
        loss=loss,
        dependencies=dependencies,
    )


@pytest.mark.parametrize("fail", [False, True])
def test_router_queues_cleared_and_normal_request_does_not_replay(miles, fail):
    data = {"rollout_routed_experts": "routes"}
    if fail:
        with pytest.raises(RuntimeError, match="forward failed"):
            miles.lora.run_forward_backward(None, 0, [], data, fail=True)
    else:
        assert miles.lora.run_forward_backward(None, 0, [], data) == "result"
    assert miles.events == ["fill", (True, 1)]
    assert miles.manager.replays == [[]]
    assert miles.manager.enabled is True
    assert miles.manager.stage == "fallthrough"
    assert miles.lora.run_forward_backward(None, 1, [], {}) == "result"
    assert miles.events[-1] == (False, 0)
    assert miles.manager.enabled is True
    assert miles.manager.replays == [[]]


def test_hooks_idempotent_and_replay_requires_registered_routers(miles):
    original = miles.lora.run_forward_backward
    install_replay_hooks(**miles.dependencies)
    assert miles.lora.run_forward_backward is original
    miles.manager.enabled = False
    with pytest.raises(ValueError, match="MoE trainer"):
        original(None, 0, [], {"rollout_routed_experts": "routes"})
    assert miles.events == []


def test_mask_fields_forwarded_and_temperature_is_request_local(miles):
    batch = {
        "rollout_sampling_mask_ids": [[3, 5, 4]],
        "rollout_sampling_mask_offsets": [[0, 0, 2, 3]],
        "loss_fn_config": {"sampling_temperature": 0.7},
        "unconcat_tokens": [torch.tensor([1, 2, 3, 4])],
        "target_tokens": [[2, 3, 4]],
        "total_lengths": [4],
        "response_lengths": [3],
    }
    keys = ["tokens"]
    actual = miles.model.get_batch(SimpleNamespace(rollout_data=batch), keys)
    assert actual == [
        "tokens",
        "rollout_sampling_mask_ids",
        "rollout_sampling_mask_offsets",
    ]
    assert keys == ["tokens"]
    args = SimpleNamespace(vocab_size=6, rollout_temperature=1.0)
    result = miles.loss._target_logprobs(args, batch, None)
    assert args.rollout_temperature == 1.0
    assert result["args"].rollout_temperature == 0.7
    assert result["unconcat_tokens"][0].tolist() == [1, 2, 3, 4]
    mask = result["rollout_sampling_mask"][0]
    assert mask._select_masks([2, 0, 1])[0].tolist() == [4, 3, 5]
    assert miles.loss._target_logprobs(args, {}, None) == "ordinary"
    with pytest.raises(ValueError, match="outside the model vocabulary"):
        miles.loss._target_logprobs(SimpleNamespace(vocab_size=5), batch, None)
