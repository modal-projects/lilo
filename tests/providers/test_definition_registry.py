from lilo.providers.modal.app import (
    DEFINITIONS,
    module_for,
    parameterization_for,
)


def test_definition_registry_resolves_every_definition() -> None:
    for definition in DEFINITIONS:
        assert module_for(definition.DEFINITION_ID) is definition
        assert parameterization_for(definition.DEFINITION_ID) == (
            definition.PARAMETERIZATION
        )
        assert definition.ENGINE_FUNCTION is not None


def test_public_model_parameterizations_are_unique() -> None:
    visible = [
        (definition.MODEL_NAME, definition.PARAMETERIZATION)
        for definition in DEFINITIONS
        if definition.CATALOG_VISIBLE
    ]
    assert len(visible) == len(set(visible))
