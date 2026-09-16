"""Exercise the installed Tinker SDK, including its HTTP retry classification."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import tinker
from tinker import types


def test_retired_sampler_410_is_terminal_without_content_length():
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            self.rfile.read(int(self.headers.get('Content-Length', 0)))
            path = self.path.rsplit('/', 1)[-1]
            status = 200
            if path == 'create_session': body = {'session_id': 'test-session'}
            elif path == 'create_sampling_session': body = {'sampling_session_id': 'old-sampler'}
            elif path == 'asample':
                requests.append(path)
                status = 410
                body = {'error': 'gone', 'message': 'sampling model was replaced'}
            else: body = {'status': 'accepted'}
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            # Match gateways that omit Content-Length. Tinker specially retries
            # some 400 responses with that header missing; Gone must be terminal.
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        service = tinker.ServiceClient(base_url=f'http://127.0.0.1:{server.server_port}', api_key='tml-test')
        sampler = service.create_sampling_client(base_model='test')
        with pytest.raises(tinker.APIStatusError) as error:
            sampler.sample(prompt=types.ModelInput.from_ints([1]), num_samples=1,
                           sampling_params=types.SamplingParams(max_tokens=1)).result(timeout=10)
        assert error.value.status_code == 410
        assert requests == ['asample']
    finally:
        if 'service' in locals() and service._session_holder:
            service._session_holder.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
