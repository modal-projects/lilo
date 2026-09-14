"""Safe HTTP transport timestamps: never retain headers, bodies, or callback info."""

import time


class HTTPTrace:
    def __init__(self, attrs):
        self.events = attrs.setdefault("http_trace", [])

    async def record(self, name, info):
        if len(self.events) < 64:
            self.events.append({"event": name, "t": time.time()})
