"""Isolate sample-worker HTTP overhead through an actual Modal Flash gateway.

The gateway's CPU handler returns a fixed one-token response immediately. This
measures transport/connection reuse only, not GPU TTFT or adapter scheduling.
"""

import modal
from pathlib import Path

from lilo.providers.modal.app import image

app = modal.App("lilo-sampling-connection-benchmark")
server_source = """
from fastapi import FastAPI, Request
app = FastAPI()
@app.post("/generate")
async def generate(request: Request):
    body = await request.json()
    return {"meta_info": {
        "finish_reason": "stop", "output_token_logprobs": [[-0.1, 5]],
        "weight_version_start": 7, "weight_version_end": 7, "cached_tokens": 0,
    }}
"""


@app.server(
    image=image,
    port=8000,
    target_concurrency=32,
    min_containers=1,
    max_containers=1,
    routing_region="us-west",
    startup_timeout=120,
)
class Server:
    @modal.enter()
    def start(self):
        import subprocess
        from lilo.inference.serving import wait_http

        Path("/tmp/connection_server.py").write_text(server_source)
        self.proc = subprocess.Popen(
            [
                "python",
                "-m",
                "uvicorn",
                "connection_server:app",
                "--app-dir",
                "/tmp",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
            ],
            start_new_session=True,
        )
        wait_http("http://127.0.0.1:8000/openapi.json", self.proc, 90)

    @modal.exit()
    def stop(self):
        from lilo.inference.serving import terminate

        terminate(self.proc)


@app.function(image=image, timeout=600, secrets=[modal.Secret.from_name("lilo-proxy")])
async def benchmark(gateway: str):
    import time
    from lilo.inference.http_client import sampling_client, close_sampling_client
    from lilo.inference.sampling import sample_task
    from lilo.providers.modal.fft_pool import proxy_auth_headers

    rows = []
    try:
        for label, reuse in (
            ("warmup", True),
            ("new-client-a", False),
            ("pooled-a", True),
            ("pooled-b", True),
            ("new-client-b", False),
        ):
            for index in range(40 if label != "warmup" else 3):
                events = []
                task = {
                    "request_id": f"{label}-{index}",
                    "sampling_session_id": "connection-bench",
                    "model_id": "model-a",
                    "publish_version": 7,
                    "latest": False,
                    "payload": {
                        "prompt": {
                            "chunks": [{"type": "encoded_text", "tokens": [1] * 1024}]
                        },
                        "num_samples": 8,
                        "sampling_params": {"max_tokens": 1},
                    },
                }
                start = time.monotonic()
                await sample_task(
                    task,
                    gateway,
                    client=sampling_client() if reuse else None,
                    headers=proxy_auth_headers(),
                    on_event=events.append,
                )
                elapsed = time.monotonic() - start
                attempts = [
                    e["attrs"] for e in events if e.get("name") == "sample_attempt"
                ]
                rows.append(
                    {
                        "phase": label,
                        "elapsed_s": elapsed,
                        "connections": sum(
                            t["event"] == "connection.connect_tcp.started"
                            for a in attempts
                            for t in a.get("http_trace", [])
                        ),
                        "attempts": len(attempts),
                    }
                )
            print(f"Completed {label}", flush=True)
    finally:
        await close_sampling_client()
    return rows


@app.local_entrypoint()
def main():
    import json

    result = benchmark.remote(Server.get_url())
    path = Path("scripts/results/lora-admission/connections.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(f"Saved {path}")
