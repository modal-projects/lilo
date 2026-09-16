"""Single live model assignment. Call only from the serialized manager."""

from lilo.control_plane.keys import placement_key, trainer_demand_key
from lilo.control_plane.records import PlacementRecord


async def model_lost(kv, engines, model_id):
    # The control plane removes demand when it reports ModelLost, including
    # after deleting a stale placement. Absence of placement alone is not loss:
    # a newly created model may still be waiting for its first acceptance.
    if await kv.get(trainer_demand_key(model_id)) is None:
        return True
    value = await kv.get(placement_key(model_id))
    if value is None:
        return False
    placement = PlacementRecord.model_validate(value)
    instance = await engines.get_instance(placement.engine_instance_id)
    return (
        instance is None
        or instance.terminal
        or placement.engine_boot_id not in ("", instance.boot_id)
    )


async def claim_model(registry, kv, engines, definition_id, model_id):
    if await registry.get.aio("retired:" + model_id):
        raise ValueError("training model was replaced; create a new training client")
    current = await registry.get.aio("slot:0")
    active = await engines.active_instances(definition_id)
    lost = current is not None and await model_lost(kv, engines, current)
    if current == model_id and lost:
        raise ValueError("trainer was lost; create a new training client")
    if current != model_id:
        if current and active and not lost:
            raise ValueError("a training model is already active in this deployment")
        if current:
            # Fence unplaced operations as well as the old, dead placement.
            await kv.delete("trainer_demand:" + current)
            await registry.put.aio("retired:" + current, True)
        route = (await registry.get.aio("routes"))[1]
        await registry.put.aio("model:" + model_id, route)
        # This single assignment is the sampling admission fence and store selector.
        await registry.put.aio("slot:0", model_id)
    if not active:
        await engines.spawn_instance(definition_id)
    return await registry.get.aio("model:" + model_id)
