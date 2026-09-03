import contextlib
from types import SimpleNamespace

import pytest
import tinker
from tinker import types

from tests.support import SingleEnginePlatform, TinkerStubExecutor, serve
from lilo.control_plane import ControlPlane, create_control_plane_app
from lilo.engine import EngineServer
from lilo.engine.backend import HttpExecutor, create_backend_app
from lilo.engine.http import HttpEngineClient, create_engine_app
from lilo.providers.local import InMemoryKeyValueStore

BASE_MODEL = "Qwen/Qwen3-8B"
DEFINITION = "qwen3_8b"
DEFINITIONS = (
    SimpleNamespace(
        DEFINITION_ID=DEFINITION,
        MODEL_NAME=BASE_MODEL,
        PARAMETERIZATION="lora",
        CATALOG_VISIBLE=True,
    ),
)
API_KEY = "tml-test"
ENGINE_TOKEN = "engine-token"


@pytest.fixture(scope="module")
def base_url():
    with contextlib.ExitStack() as stack:
        backend_url = stack.enter_context(
            serve(create_backend_app(TinkerStubExecutor()))
        )
        engine_server = EngineServer(HttpExecutor(backend_url))
        engine_url = stack.enter_context(
            serve(create_engine_app(engine_server, token=ENGINE_TOKEN))
        )
        platform = SingleEnginePlatform(
            DEFINITION,
            HttpEngineClient(engine_url, token=ENGINE_TOKEN),
        )
        plane = ControlPlane(InMemoryKeyValueStore(), platform)
        app = create_control_plane_app(
            plane,
            DEFINITIONS,
            api_key=API_KEY,
            retrieve_window=5.0,
        )
        yield stack.enter_context(serve(app))


def test_full_stack_training_loop(base_url: str) -> None:
    service = tinker.ServiceClient(base_url=base_url, api_key=API_KEY)

    training = service.create_lora_training_client(base_model=BASE_MODEL, rank=32)

    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={"target_tokens": [2, 3, 4], "weights": [1.0, 1.0, 1.0]},
    )
    forward_future = training.forward_backward([datum], "cross_entropy")
    optim_future = training.optim_step(types.AdamParams(learning_rate=1e-4))

    assert forward_future.result(timeout=30).metrics["loss:sum"] == 1.25
    assert optim_future.result(timeout=30).metrics == {"lr": 1e-4}
