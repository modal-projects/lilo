"""Model-name routing for shared YAML deployments and scoped engines."""


class DeploymentRoutes:
    def __init__(self, definitions):
        self.all = tuple(definitions)
        self.visible = tuple(d for d in self.all if d.CATALOG_VISIBLE)
        for model, mode in {(d.MODEL_NAME, d.PARAMETERIZATION) for d in self.visible}:
            defaults = [
                d
                for d in self.visible
                if d.MODEL_NAME == model
                and d.PARAMETERIZATION == mode
                and getattr(d, "ROUTING_DEFAULT", False)
            ]
            if len(defaults) > 1:
                raise ValueError(f"multiple defaults for {model} ({mode})")

    def select(self, model, mode):
        explicit = [
            d
            for d in self.all
            if d.DEFINITION_ID == model and d.PARAMETERIZATION == mode
        ]
        if explicit:
            return explicit[0]
        matches = [
            d
            for d in self.visible
            if d.MODEL_NAME == model and d.PARAMETERIZATION == mode
        ]
        if len(matches) <= 1:
            return next(iter(matches), None)
        defaults = [d for d in matches if getattr(d, "ROUTING_DEFAULT", False)]
        if len(defaults) == 1:
            return defaults[0]
        choices = ", ".join(
            f"{getattr(d, 'DEPLOYMENT_NAME', d.DEFINITION_ID)} ({d.MAX_CONTEXT_LENGTH} tokens)"
            for d in matches
        )
        raise ValueError(
            f"ambiguous {mode} deployment for {model}; configure routing.default: {choices}"
        )

    def sampling(self, model):
        requested = [d for d in self.all if d.DEFINITION_ID == model]
        if len(requested) == 1:
            return requested[0]
        matches = [d for d in self.visible if d.MODEL_NAME == model]
        explicit = [d for d in matches if getattr(d, "SAMPLING_DEFAULT", False)]
        if len(explicit) == 1:
            return explicit[0]
        if len(explicit) > 1:
            raise ValueError(f"multiple sampling defaults for {model}")
        selected = [
            d
            for mode in {d.PARAMETERIZATION for d in matches}
            if (d := self.select(model, mode)) is not None
        ]
        if len(selected) > 1:
            raise ValueError(
                f"ambiguous sampling deployment for {model}; configure routing.sampling_default"
            )
        return next(iter(selected), None)

    def capabilities(self):
        result = []
        for model in dict.fromkeys(d.MODEL_NAME for d in self.visible):
            try:
                selected = [
                    self.select(model, mode)
                    for mode in {
                        d.PARAMETERIZATION
                        for d in self.visible
                        if d.MODEL_NAME == model
                    }
                ]
            except ValueError:
                continue
            result.append(
                {
                    "model_name": model,
                    "max_context_length": min(d.MAX_CONTEXT_LENGTH for d in selected),
                }
            )
        return result
