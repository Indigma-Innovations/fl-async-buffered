"""Public API schemas."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class LinearLayerSpec(BaseModel):
    type: Literal["linear"] = "linear"
    in_features: int = Field(gt=0)
    out_features: int = Field(gt=0)
    bias: bool = True


class ReLULayerSpec(BaseModel):
    type: Literal["relu"] = "relu"


LayerSpec = Annotated[LinearLayerSpec | ReLULayerSpec, Field(discriminator="type")]


class ArchitectureSpec(BaseModel):
    framework: Literal["pytorch"] = "pytorch"
    type: Literal["sequential"] = "sequential"
    layers: list[LayerSpec]


class ModelSnapshot(BaseModel):
    version: int
    architecture: ArchitectureSpec
    state_dict: dict[str, Any]


class UpdateSubmission(BaseModel):
    client_id: str = Field(min_length=1, max_length=128)
    base_version: int = Field(ge=0)
    num_samples: int = Field(gt=0)
    state_dict: dict[str, Any]
    metrics: dict[str, float] = Field(default_factory=dict)


class UpdateReceipt(BaseModel):
    accepted: bool = True
    aggregated: bool
    model_version: int
    buffered: int
    buffer_target: int


class ServerStatus(BaseModel):
    model_version: int
    buffered: int
    buffer_target: int
    updates_received: int
    aggregations: int
