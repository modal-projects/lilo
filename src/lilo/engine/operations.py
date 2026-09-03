from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias

from pydantic import BaseModel, ConfigDict, Field
from tinker import AdamParams, LoraConfig
from tinker.lib._pydantic_conv import to_pydantic_input
from tinker.types._pydantic_types.forward_backward_input import (
    ForwardBackwardInput as ForwardBackwardInputModel,
)
from tinker.types.forward_backward_input import ForwardBackwardInput

from lilo.backends.contract import ModelSpec

from .api import OperationKind


class OperationPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    extensions: dict[str, Any] = Field(default_factory=dict)


class LoadWeightsPayload(OperationPayloadModel):
    uri: str
    restore_optimizer: bool = False


class SaveWeightsPayload(OperationPayloadModel):
    destination: str = "latest"
    include_optimizer: bool = True
    overwrite: bool = False


class SaveWeightsForSamplerPayload(OperationPayloadModel):
    publish_version: int = Field(gt=0)


class SkipPayload(OperationPayloadModel):
    error: str


OperationPayload: TypeAlias = (
    ForwardBackwardInput
    | AdamParams
    | LoadWeightsPayload
    | SaveWeightsPayload
    | SaveWeightsForSamplerPayload
    | SkipPayload
)

_PAYLOAD_TYPES = {
    OperationKind.FORWARD: ForwardBackwardInput,
    OperationKind.FORWARD_BACKWARD: ForwardBackwardInput,
    OperationKind.OPTIM_STEP: AdamParams,
    OperationKind.LOAD_WEIGHTS: LoadWeightsPayload,
    OperationKind.SAVE_WEIGHTS: SaveWeightsPayload,
    OperationKind.SAVE_WEIGHTS_FOR_SAMPLER: SaveWeightsForSamplerPayload,
}
_LORA_FIELDS = {
    "rank",
    "seed",
    "train_unembed",
    "train_mlp",
    "train_attn",
}


def parse_operation_payload(
    kind: OperationKind,
    value: object,
) -> OperationPayload:
    expected = _PAYLOAD_TYPES[kind]
    if isinstance(value, expected):
        return value
    if not isinstance(value, Mapping):
        raise ValueError(f"{kind.value} payload must be an object")

    raw = dict(value)
    if kind in {OperationKind.FORWARD, OperationKind.FORWARD_BACKWARD}:
        parsed = ForwardBackwardInputModel.model_validate(
            {
                "data": raw.get("data", ()),
                "loss_fn": raw.get("loss_fn"),
                "loss_fn_config": raw.get("loss_fn_config"),
            }
        )
        return ForwardBackwardInput(
            data=list(parsed.data),
            loss_fn=parsed.loss_fn,
            loss_fn_config=parsed.loss_fn_config,
        )
    if kind == OperationKind.OPTIM_STEP:
        return AdamParams.model_validate(raw.get("adam_params") or {})
    if kind == OperationKind.LOAD_WEIGHTS:
        known = {
            "path",
            "uri",
            "optimizer",
            "restore_optimizer",
            "extensions",
        }
        restore_optimizer = bool(
            raw.get("restore_optimizer", raw.get("optimizer", False))
        )
        return LoadWeightsPayload(
            uri=raw.get("path") or raw.get("uri"),
            restore_optimizer=restore_optimizer,
            extensions=_extensions(raw, known),
        )
    if kind == OperationKind.SAVE_WEIGHTS:
        known = {
            "checkpoint_id",
            "name",
            "path",
            "destination",
            "include_optimizer",
            "overwrite",
            "extensions",
        }
        destination = next(
            (
                raw[key]
                for key in ("path", "destination", "checkpoint_id", "name")
                if key in raw and raw[key] is not None
            ),
            "latest",
        )
        if (
            not isinstance(destination, str)
            or not destination
            or destination != destination.strip()
            or destination in {".", ".."}
            or "/" in destination
            or "\\" in destination
            or "\0" in destination
        ):
            raise ValueError("checkpoint name must be a single path component")
        return SaveWeightsPayload(
            destination=destination,
            include_optimizer=bool(raw.get("include_optimizer", True)),
            overwrite=bool(raw.get("overwrite", False)),
            extensions=_extensions(raw, known),
        )
    if kind == OperationKind.SAVE_WEIGHTS_FOR_SAMPLER:
        known = {"publish_version", "extensions"}
        return SaveWeightsForSamplerPayload(
            publish_version=raw.get("publish_version"),
            extensions=_extensions(raw, known),
        )
    raise ValueError(f"unsupported operation: {kind.value}")


def serialize_operation_payload(payload: OperationPayload) -> dict[str, Any]:
    if isinstance(payload, ForwardBackwardInput):
        return to_pydantic_input(payload).model_dump(
            mode="json",
            exclude_defaults=True,
        )
    if isinstance(payload, AdamParams):
        return {"adam_params": payload.model_dump(mode="json", exclude_defaults=True)}
    return payload.model_dump(mode="json", exclude_defaults=True)


def parse_model_spec(value: object) -> ModelSpec:
    if isinstance(value, ModelSpec):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("model spec must be an object")

    raw_parameterization = value.get("parameterization")
    parameterization = (
        raw_parameterization.get("type")
        if isinstance(raw_parameterization, Mapping)
        else raw_parameterization
    )
    raw_lora = value.get("lora_config")
    if parameterization is None and raw_lora is not None:
        parameterization = "lora"
    if parameterization not in {"lora", "full"}:
        raise ValueError("parameterization must be lora or full")

    base_model = value.get("base_model")
    if not isinstance(base_model, str) or not base_model:
        raise ValueError("base_model is required")

    if isinstance(raw_lora, LoraConfig):
        lora = raw_lora
    elif isinstance(raw_lora, Mapping):
        lora = LoraConfig.model_validate(
            {key: raw_lora[key] for key in _LORA_FIELDS if key in raw_lora}
        )
    elif raw_lora is None:
        lora = None
    else:
        raise ValueError("lora_config must be an object")
    return ModelSpec(
        base_model=base_model,
        parameterization=parameterization,
        lora_config=lora,
    )


def _extensions(raw: dict[str, Any], known: set[str]) -> dict[str, Any]:
    extensions = raw.get("extensions") or {}
    if not isinstance(extensions, Mapping):
        raise ValueError("extensions must be an object")
    return {
        **dict(extensions),
        **{key: value for key, value in raw.items() if key not in known},
    }
