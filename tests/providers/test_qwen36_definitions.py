from lilo.providers.modal.app import (
    DEFINITIONS,
    module_for,
    parameterization_for,
)
from lilo.providers.modal.definitions import (
    qwen3_6_27b_full_64k,
    qwen3_6_35b_a3b_full_64k,
)


def test_qwen36_full_definitions_are_registered() -> None:
    for definition in (
        qwen3_6_27b_full_64k,
        qwen3_6_35b_a3b_full_64k,
    ):
        assert definition in DEFINITIONS
        assert module_for(definition.DEFINITION_ID) is definition
        assert parameterization_for(definition.DEFINITION_ID) == "full"


def test_qwen36_35b_reuses_qwen35_moe_topology() -> None:
    assert qwen3_6_35b_a3b_full_64k.TENSOR_MODEL_PARALLEL_SIZE == 4
    assert qwen3_6_35b_a3b_full_64k.CONTEXT_PARALLEL_SIZE == 2
    assert qwen3_6_35b_a3b_full_64k.EXPERT_MODEL_PARALLEL_SIZE == 8
    assert qwen3_6_35b_a3b_full_64k.ROLLOUT_EXPERT_PARALLEL_SIZE == 4


def test_qwen36_27b_uses_dense_tp4_topology() -> None:
    assert qwen3_6_27b_full_64k.TENSOR_MODEL_PARALLEL_SIZE == 4
    assert qwen3_6_27b_full_64k.CONTEXT_PARALLEL_SIZE == 2
    assert qwen3_6_27b_full_64k.ROLLOUT_TENSOR_PARALLEL_SIZE == 4
    assert qwen3_6_27b_full_64k.ROLLOUT_EXPERT_PARALLEL_SIZE == 1
