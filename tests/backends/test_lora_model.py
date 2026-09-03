from contextlib import nullcontext
from types import SimpleNamespace

from runtime_stubs import backend_runtime_imports

with backend_runtime_imports():
    from lilo.backends.megatron_runtime.lora import model as lora_model

_copy_embeddings_to_output = lora_model._copy_embeddings_to_output


class Weight:
    def __init__(self, value):
        self.value = value

    def copy_(self, source):
        self.value = source.value


def test_copy_embeddings_supports_nested_language_model(monkeypatch) -> None:
    monkeypatch.setattr(lora_model, "torch", SimpleNamespace(no_grad=nullcontext))
    monkeypatch.setattr(lora_model, "unwrap_model", lambda model: model)
    embedding = SimpleNamespace(weight=Weight(7))
    output = SimpleNamespace(weight=Weight(0))
    chunk = SimpleNamespace(
        language_model=SimpleNamespace(
            embedding=SimpleNamespace(word_embeddings=embedding),
            output_layer=SimpleNamespace(to_wrap=output),
        )
    )

    _copy_embeddings_to_output([chunk])
    assert output.weight.value == 7
