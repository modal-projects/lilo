"""Pinned demand and routes. Mutations run only in the serialized manager."""

PIN_IDLE_SECONDS = 600
# A lease outlives execute_sample's 3600s timeout, including cancellation delivery.
PIN_LEASE_SECONDS = 3900


def wanted(record, now):
    return now - record["last_used"] < PIN_IDLE_SECONDS or any(
        expiry > now for expiry in record.get("leases", {}).values()
    )


async def touch_pin(registry, key, *, now, lease=None, release=False):
    records = await registry.get.aio("pin_records") or {}
    record = records.get(key)
    if record is None:
        if release:
            return  # Releasing an expired request must not resurrect demand.
        record = records[key] = {"last_used": now, "leases": {}}
    record["last_used"] = now
    leases = record["leases"] = {
        k: expiry for k, expiry in record.get("leases", {}).items() if expiry > now
    }
    if lease is not None:
        if release:
            leases.pop(lease, None)
        else:
            leases[lease] = now + PIN_LEASE_SECONDS
    await registry.put.aio("pin_records", records)


async def pin_demand(registry, *, now, routes=None):
    """Forget idle demand, including versions that never finished provisioning."""
    records = await registry.get.aio("pin_records") or {}
    live = {}
    for key, record in records.items():
        if wanted(record, now):
            record["leases"] = {
                k: v for k, v in record.get("leases", {}).items() if v > now
            }
            live[key] = record
        else:
            await registry.pop.aio(key, None)
    await registry.put.aio("pin_records", live)
    # Repair a route lost during cleanup/RPC races without opening a duplicate app.
    for key, route in (routes or {}).items():
        if key in live and not await registry.get.aio(key):
            await registry.put.aio(key, route)
    return live


async def publish_pin(registry, key, route, *, now):
    records = await registry.get.aio("pin_records") or {}
    if key not in records or not wanted(records[key], now):
        return False
    await registry.put.aio(key, route)
    return True


async def forget_pin_route(registry, key, route):
    # A finishing old app must not remove a replacement's route or fresh demand.
    if await registry.get.aio(key) == route:
        await registry.pop.aio(key, None)
