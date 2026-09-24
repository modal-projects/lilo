"""Select recipes in deployment order, or explicitly by definition ID."""


class DeploymentRoutes:
    def __init__(self, definitions):
        self.definitions = tuple(definitions)

    def select(self, model, mode=None):
        candidates = [
            d for d in self.definitions if mode is None or d.PARAMETERIZATION == mode
        ]
        for field in ("DEFINITION_ID", "MODEL_NAME"):
            for definition in candidates:
                if getattr(definition, field) == model:
                    return definition
        return None

    def capabilities(self):
        selected = {}
        for definition in self.definitions:
            selected.setdefault(
                (definition.MODEL_NAME, definition.PARAMETERIZATION), definition
            )
        contexts = {}
        for (model, _), definition in selected.items():
            contexts[model] = min(
                contexts.get(model, definition.MAX_CONTEXT_LENGTH),
                definition.MAX_CONTEXT_LENGTH,
            )
        return [
            {"model_name": model, "max_context_length": context}
            for model, context in contexts.items()
        ]
