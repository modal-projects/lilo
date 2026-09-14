"""Retry-safe model preparation for scoped deployments."""
from lilo.control_plane import ControlPlane


class ScopedControlPlane(ControlPlane):
    def __init__(self, *args, prepare_model, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepare_scoped_model = prepare_model

    async def create_model(self, **kwargs):
        creation = await super().create_model(**kwargs)
        # Creation is durable before infrastructure preparation. A retried API
        # request must finish preparation, not silently skip a failed first try.
        await self.prepare_scoped_model(creation.model)
        return creation
