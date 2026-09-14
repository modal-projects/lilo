import asyncio
from types import SimpleNamespace

import pytest

from lilo.providers.modal.scoped_pins import reap_pins, touch_pin
class Registry:
    def __init__(self, values):
        self.values = values
        self.get = SimpleNamespace(aio=self.read)
        self.put = SimpleNamespace(aio=self.write)
    async def read(self, key): return self.values.get(key)
    async def write(self, key, value): self.values[key] = value


def test_idle_reaping_preserves_active_leases_and_releases_route():
    async def check():
        registry = Registry({'children': ['idle-app', 'busy-app'], 'idle': {'url': 'idle'},
            'pin_records': {
                'idle': {'app_name': 'idle-app', 'last_used': 0, 'leases': {}},
                'busy': {'app_name': 'busy-app', 'last_used': 0, 'leases': {'request': 4000}},
            }})
        stopped = []
        async def stop(name): stopped.append(name)
        await reap_pins(registry, stop, now=1000)
        assert stopped == ['idle-app']
        assert registry.values['idle'] is None
        assert registry.values['children'] == ['busy-app']
        await touch_pin(registry, 'busy', now=1000, lease='request', release=True)
        await reap_pins(registry, stop, now=1500)
        assert stopped == ['idle-app']
        await reap_pins(registry, stop, now=1601)
        assert stopped == ['idle-app', 'busy-app']
    asyncio.run(check())


def test_failed_stop_keeps_ownership_and_route_for_retry():
    async def check():
        registry = Registry({'children': ['child'], 'pin': {'url': 'valid'},
            'pin_records': {'pin': {'app_name': 'child', 'last_used': 0}}})
        async def stop(name): raise RuntimeError('provider unavailable')
        with pytest.raises(RuntimeError): await reap_pins(registry, stop, now=1000)
        assert registry.values['children'] == ['child']
        assert registry.values['pin'] == {'url': 'valid'}
    asyncio.run(check())


def test_abandoned_worker_lease_expires_and_next_acquire_renews():
    async def check():
        registry = Registry({'children': ['child'],
            'pin_records': {'pin': {'app_name': 'child', 'last_used': 0}}})
        stopped = []
        async def stop(name): stopped.append(name)
        await touch_pin(registry, 'pin', now=0, lease='lost-worker')
        await reap_pins(registry, stop, now=3800)
        assert not stopped
        await touch_pin(registry, 'pin', now=3800, lease='new-worker')
        await reap_pins(registry, stop, now=4500)
        assert not stopped
        await reap_pins(registry, stop, now=7800)
        assert stopped == ['child']
    asyncio.run(check())
