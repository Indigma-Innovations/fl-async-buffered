import pytest
import torch

from fl_async.config import Settings
from fl_async.orchestrator import FedAvgOrchestrator
from fl_async.schemas import UpdateSubmission


def submission(value: float, version: int = 0, samples: int = 1) -> UpdateSubmission:
    state_dict = {
        "0.weight": torch.full((3, 2), value).tolist(),
        "0.bias": torch.full((3,), value).tolist(),
        "2.weight": torch.full((2, 3), value).tolist(),
        "2.bias": torch.full((2,), value).tolist(),
    }
    return UpdateSubmission(
        client_id="client", base_version=version, num_samples=samples, state_dict=state_dict
    )


@pytest.mark.asyncio
async def test_aggregates_when_buffer_reaches_target():
    server = FedAvgOrchestrator(Settings(buffer_target=2, input_size=2, hidden_size=3))
    first = await server.submit(submission(2, samples=1))
    second = await server.submit(submission(4, samples=3))
    model = await server.snapshot()

    assert not first.aggregated
    assert second.aggregated
    assert second.model_version == 1
    assert torch.allclose(torch.tensor(model.state_dict["0.weight"]), torch.full((3, 2), 3.5))
    assert (await server.status()).buffered == 0


@pytest.mark.asyncio
async def test_rejects_invalid_tensor_shape():
    server = FedAvgOrchestrator(Settings(input_size=2, hidden_size=3))
    update = submission(1)
    update.state_dict["0.weight"] = [[1.0]]
    with pytest.raises(ValueError, match="shape"):
        await server.submit(update)
