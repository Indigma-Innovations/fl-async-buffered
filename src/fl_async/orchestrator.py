"""In-memory asynchronous FedBuff orchestration."""

import asyncio
from dataclasses import dataclass

import torch

from .config import Settings
from .model import build_model, load_json_state, mlp_spec, state_dict_to_json
from .schemas import ArchitectureSpec, ModelSnapshot, ServerStatus, UpdateReceipt, UpdateSubmission


@dataclass(slots=True)
class BufferedUpdate:
    state_dict: dict[str, torch.Tensor]
    base_version: int
    num_samples: int


class FedAvgOrchestrator:
    """Collect updates and atomically apply sample/staleness-weighted FedAvg.

    The service deliberately keeps state in memory: this mini distribution is a
    single-process local reference implementation, not a production control plane.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.architecture: ArchitectureSpec = mlp_spec(
            settings.input_size, settings.hidden_size, settings.output_size
        )
        self.device = torch.device(settings.device)
        fork_devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(settings.seed)
            self._model = build_model(self.architecture).to(self.device)
        self._version = 0
        self._buffer: list[BufferedUpdate] = []
        self._updates_received = 0
        self._aggregations = 0
        self._lock = asyncio.Lock()

    def _parse_state_dict(self, raw: dict[str, object]) -> dict[str, torch.Tensor]:
        candidate = build_model(self.architecture).to(self.device)
        load_json_state(candidate, raw)
        return {name: tensor.detach().clone() for name, tensor in candidate.state_dict().items()}

    async def snapshot(self) -> ModelSnapshot:
        async with self._lock:
            return ModelSnapshot(
                version=self._version,
                architecture=self.architecture,
                state_dict=state_dict_to_json(self._model),
            )

    async def status(self) -> ServerStatus:
        async with self._lock:
            return ServerStatus(
                model_version=self._version,
                buffered=len(self._buffer),
                buffer_target=self.settings.buffer_target,
                updates_received=self._updates_received,
                aggregations=self._aggregations,
            )

    async def submit(self, submission: UpdateSubmission) -> UpdateReceipt:
        state_dict = self._parse_state_dict(submission.state_dict)
        async with self._lock:
            if submission.base_version > self._version:
                raise ValueError("base_version is newer than the server model")
            self._buffer.append(
                BufferedUpdate(state_dict, submission.base_version, submission.num_samples)
            )
            self._updates_received += 1
            aggregated = len(self._buffer) >= self.settings.buffer_target
            if aggregated:
                self._aggregate()
            return UpdateReceipt(
                aggregated=aggregated,
                model_version=self._version,
                buffered=len(self._buffer),
                buffer_target=self.settings.buffer_target,
            )

    def _aggregate(self) -> None:
        coefficients = torch.tensor([
            update.num_samples / (1 + self.settings.staleness_decay * (self._version - update.base_version))
            for update in self._buffer
        ], dtype=torch.float64)
        coefficients /= coefficients.sum()
        rate = self.settings.server_learning_rate
        current = self._model.state_dict()
        averaged = {
            name: sum(
                coefficient * update.state_dict[name]
                for coefficient, update in zip(coefficients, self._buffer, strict=True)
            ).to(dtype=tensor.dtype)
            for name, tensor in current.items()
        }
        self._model.load_state_dict({
            name: (1 - rate) * tensor + rate * averaged[name]
            for name, tensor in current.items()
        })
        self._buffer.clear()
        self._version += 1
        self._aggregations += 1
