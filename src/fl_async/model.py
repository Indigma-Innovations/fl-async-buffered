"""PyTorch model specification and construction helpers."""

import torch
from torch import nn

from .schemas import ArchitectureSpec, LinearLayerSpec, ReLULayerSpec


def mlp_spec(input_size: int, hidden_size: int, output_size: int) -> ArchitectureSpec:
    """Return the server-owned specification for the example classifier."""
    return ArchitectureSpec(
        layers=[
            LinearLayerSpec(in_features=input_size, out_features=hidden_size),
            ReLULayerSpec(),
            LinearLayerSpec(in_features=hidden_size, out_features=output_size),
        ]
    )


def build_model(spec: ArchitectureSpec) -> nn.Sequential:
    """Construct a PyTorch module solely from a server-provided specification."""
    layers: list[nn.Module] = []
    for layer in spec.layers:
        if isinstance(layer, LinearLayerSpec):
            layers.append(nn.Linear(layer.in_features, layer.out_features, bias=layer.bias))
        elif isinstance(layer, ReLULayerSpec):
            layers.append(nn.ReLU())
        else:  # pragma: no cover - Pydantic prevents unknown layer types
            raise ValueError(f"unsupported layer: {layer}")  # noqa: TRY004
    return nn.Sequential(*layers)


def state_dict_to_json(model: nn.Module) -> dict[str, list]:
    """Convert a module state dictionary to JSON-compatible arrays."""
    return {name: tensor.detach().cpu().tolist() for name, tensor in model.state_dict().items()}


def load_json_state(model: nn.Module, raw: dict[str, object]) -> None:
    """Validate and load a JSON state dictionary into a module."""
    expected = model.state_dict()
    if set(raw) != set(expected):
        raise ValueError(f"state_dict keys must be {sorted(expected)}")
    converted: dict[str, torch.Tensor] = {}
    for name, reference in expected.items():
        try:
            tensor = torch.as_tensor(raw[name], dtype=reference.dtype)
        except (TypeError, ValueError) as error:
            raise ValueError(f"parameter {name} is not a valid tensor") from error
        if tensor.shape != reference.shape:
            raise ValueError(
                f"parameter {name} has shape {tuple(tensor.shape)}; expected {tuple(reference.shape)}"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError(f"parameter {name} contains non-finite values")
        converted[name] = tensor
    model.load_state_dict(converted, strict=True)
