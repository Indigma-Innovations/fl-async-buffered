"""Five-client synthetic classification demo for FL Async."""
import argparse
import asyncio
import httpx
import torch
from fl_async.model import build_model, load_json_state, state_dict_to_json
from fl_async.schemas import ArchitectureSpec


def train(model: torch.nn.Module, features: torch.Tensor, labels: torch.Tensor, epochs: int) -> float:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_function = torch.nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_function(model(features), labels)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        return float((model(features).argmax(dim=1) == labels).float().mean())


async def run_client(
    client_id: int, x: torch.Tensor, y: torch.Tensor, url: str, epochs: int, rounds: int
) -> None:
    async with httpx.AsyncClient(base_url=url, timeout=30) as api:
        for round_num in range(1, rounds + 1):
            snapshot = (await api.get("/model")).raise_for_status().json()
            spec = ArchitectureSpec.model_validate(snapshot["architecture"])
            model = build_model(spec)
            load_json_state(model, snapshot["state_dict"])
            accuracy = await asyncio.to_thread(train, model, x, y, epochs)
            response = await api.post("/updates", json={
                "client_id": f"client-{client_id}",
                "base_version": snapshot["version"],
                "num_samples": len(x),
                "state_dict": state_dict_to_json(model),
                "metrics": {"local_accuracy": accuracy},
            })
            response.raise_for_status()
            receipt = response.json()
            print(
                f"client-{client_id} round {round_num}/{rounds}: "
                f"accuracy={accuracy:.3f}, base_version={snapshot['version']}, "
                f"model_version={receipt['model_version']}, buffered={receipt['buffered']}"
            )


async def demo(url: str, clients: int, samples: int, epochs: int, rounds: int) -> None:
    generator = torch.Generator().manual_seed(42)
    x = torch.randn((samples, 20), generator=generator)
    separator = torch.randn(20, generator=generator)
    y = (x @ separator + torch.randn(samples, generator=generator) * 0.5 > 0).long()
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)
    train_x, train_y = x[: int(samples * 0.8)], y[: int(samples * 0.8)]
    indices = torch.randperm(len(train_x), generator=generator).tensor_split(clients)
    await asyncio.gather(
        *(
            run_client(i + 1, train_x[index], train_y[index], url, epochs, rounds)
            for i, index in enumerate(indices)
        )
    )
    async with httpx.AsyncClient(base_url=url) as api:
        print("server:", (await api.get("/status")).json())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=20)
    args = parser.parse_args()
    asyncio.run(demo(args.url, args.clients, args.samples, args.epochs, args.rounds))


if __name__ == "__main__":
    main()
