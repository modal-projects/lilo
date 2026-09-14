"""Single live model assignment. Call only from the serialized manager."""


async def claim_model(registry, kv, engines, definition_id, model_id):
    if await registry.get.aio('retired:' + model_id):
        raise ValueError('training model was replaced; create a new training client')
    current = await registry.get.aio('slot:0')
    active = await engines.active_instances(definition_id)
    if current == model_id and not active and await kv.get('placement:' + model_id):
        raise ValueError('trainer was lost; create a new training client')
    if current != model_id:
        if current and active:
            raise ValueError('a training model is already active in this deployment')
        if current:
            # Fence unplaced operations as well as the old, dead placement.
            await kv.delete('trainer_demand:' + current)
            await registry.put.aio('retired:' + current, True)
        route = (await registry.get.aio('routes'))[1]
        await registry.put.aio('model:' + model_id, route)
        # This single assignment is the sampling admission fence and store selector.
        await registry.put.aio('slot:0', model_id)
    if not active:
        await engines.spawn_instance(definition_id)
    return await registry.get.aio('model:' + model_id)
