"""Runtime configuration for the local server."""


import torch
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _detect_device() -> str:
    """Return 'cuda' if a working GPU is available, else 'cpu'."""
    if torch.cuda.is_available():
        try:
            torch.cuda.init()
            return "cuda"
        except RuntimeError:
            return "cpu"
    return "cpu"


class Settings(BaseSettings):
    """Configuration loaded from ``FL_`` environment variables."""

    model_config = SettingsConfigDict(env_prefix="FL_", env_file=".env")

    buffer_target: int = Field(default=3, ge=1)
    input_size: int = Field(default=20, ge=1)
    hidden_size: int = Field(default=16, ge=1)
    output_size: int = Field(default=2, ge=2)
    server_learning_rate: float = Field(default=1.0, gt=0, le=1)
    staleness_decay: float = Field(default=1.0, ge=0)
    seed: int = 42

    # None (default) = auto-detect at runtime. Set FL_DEVICE=cpu/cuda to force it.
    device: str | None = Field(default=None)

    @field_validator("device", mode="after")
    @classmethod
    def resolve_device(cls, v: str | None) -> str:
        if v is not None:
            v = v.lower()
            if v not in {"cpu", "cuda"}:
                raise ValueError(f"device must be 'cpu' or 'cuda', got {v!r}")
            return v
        return _detect_device()
