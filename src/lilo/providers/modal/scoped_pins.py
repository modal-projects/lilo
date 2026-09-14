"""Idle pinned app reclamation. All mutations run in the serialized manager."""
PIN_IDLE_SECONDS = 600
# execute_sample has a 3600s Modal timeout. A lease outlives that invocation,
# including cancellation delivery, so a crashed worker cannot pin an app forever.
PIN_LEASE_SECONDS = 3900


async def touch_pin(registry, key, *, now, lease=None, release=False):
    records = await registry.get.aio('pin_records') or {}
    record = records[key]
    record['last_used'] = now
    leases = record.setdefault('leases', {})
    if lease is not None:
        if release:
            leases.pop(lease, None)
        else:
            leases[lease] = now + PIN_LEASE_SECONDS
    await registry.put.aio('pin_records', records)


async def reap_pins(registry, stop, *, now):
    records = await registry.get.aio('pin_records') or {}
    stopped = []
    for key, record in list(records.items()):
        if now - record['last_used'] < PIN_IDLE_SECONDS:
            continue
        if any(expiry > now for expiry in record.get('leases', {}).values()):
            continue
        # Stop before forgetting the route/ownership. A failed stop is retryable.
        await stop(record['app_name'])
        await registry.put.aio(key, None)
        del records[key]
        await registry.put.aio('pin_records', records)
        children = await registry.get.aio('children') or []
        await registry.put.aio('children', [c for c in children if c != record['app_name']])
        stopped.append(record['app_name'])
    return stopped
