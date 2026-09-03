from types import SimpleNamespace

from stitch.types import VersionRef

from lilo.providers.modal import fft_pool


def test_latest_pool_wakes_through_flash_gateway(monkeypatch) -> None:
    calls = []
    clients = []
    monkeypatch.setenv("MODAL_PROXY_TOKEN_ID", "wk-a")
    monkeypatch.setenv("MODAL_PROXY_TOKEN_SECRET", "ws-a")

    class Response:
        def raise_for_status(self) -> None:
            return None

    class Client:
        def __init__(self, **kwargs):
            clients.append(kwargs["headers"])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, url, *, headers):
            calls.append((url, headers))
            return Response()

    monkeypatch.setattr(fft_pool.httpx, "Client", Client)
    monkeypatch.setattr(
        fft_pool,
        "list_flash_containers",
        lambda *_: [
            SimpleNamespace(host="replica-a.modal.host", port=443),
            {"host": "https://replica-b.modal.host/", "port": 8443},
        ],
    )
    monkeypatch.setattr(
        fft_pool.FFTLatestPool,
        "gateway_url",
        lambda _: "https://rollout.modal.direct",
    )

    pool = fft_pool.FFTLatestPool("definition", "model")
    pool.wake(["replica-a", "replica-b"], VersionRef("model", 3))

    assert {
        headers["modal-flash-upstream"]: url for url, headers in calls
    } == {
        "replica-a.modal.host:443": "https://rollout.modal.direct/wake",
        "replica-b.modal.host:8443": "https://rollout.modal.direct/wake",
    }
    assert clients == [{"Modal-Key": "wk-a", "Modal-Secret": "ws-a"}]


def test_pool_spec_round_trips_sizing_through_dict_and_env() -> None:
    spec = fft_pool.FFTPoolSpec(
        "definition", "model", True, 0, min_containers=2, max_containers=8
    )
    assert fft_pool.FFTPoolSpec.from_dict({**spec.as_dict(), "touched_at": 1.0}) == spec
    assert fft_pool.FFTPoolSpec.from_dict(
        {
            "definition_id": "definition",
            "model_id": "model",
            "latest": False,
            "version": 3,
        }
    ) == fft_pool.FFTPoolSpec("definition", "model", False, 3)
    env = spec.env()
    assert env["LILO_FFT_POOL_APP_NAME"] == spec.app_name
    assert env["LILO_FFT_POOL_LATEST"] == "1"
    assert env["LILO_FFT_POOL_VERSION"] == "0"
    assert env["LILO_FFT_POOL_MIN_CONTAINERS"] == "2"
    assert env["LILO_FFT_POOL_MAX_CONTAINERS"] == "8"
    assert "LILO_FFT_POOL_SCALEDOWN_WINDOW" not in env
