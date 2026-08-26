# Async Buffered Federated Learning

A small open-source implementation of **asynchronous buffered federated learning** with PyTorch. It is one local FastAPI process, an in-memory FedAvg orchestrator, and a runnable multi-client, multi-round MLP example.

1. The server owns the PyTorch architecture specification and initializes its global model on CPU or GPU, auto-detected at startup (or forced via `FL_DEVICE`).
2. Clients fetch the architecture specification, `state_dict`, and current version from `GET /model`.
3. Each client constructs the specified `torch.nn.Sequential` model, loads the state, trains it locally with PyTorch for a configurable number of epochs, and sends the resulting `state_dict` to `POST /updates`.
4. Updates enter a buffer without waiting for the other clients.
5. Once `buffer_target` updates are available, the server atomically performs sample-weighted FedAvg on PyTorch tensors. Stale updates are down-weighted by `1 / (1 + staleness_decay * staleness)`.
6. Clients repeat steps 2-5 for a configurable number of rounds, so later rounds may fetch a model version that has already advanced past what earlier-launched clients last saw. This is the "asynchronous" and "staleness" part of the design.

State lives in memory and resets when the process restarts.

## Quick start

Requires Python 3.13 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run uvicorn fl_async.main:app --host 127.0.0.1 --port 8000
```

In another terminal, launch the synthetic PyTorch classification demo (five concurrent clients, 20 rounds of one local epoch each, by default):

```bash
uv run fl-demo
```

With the default buffer target of 3 and 5 clients × 20 rounds = 100 total updates submitted, the server aggregates every time 3 updates accumulate, so you'll see 33 aggregations (`⌊100 / 3⌋`), with 1 update left buffered at the end. To aggregate all five clients together on every round instead:

```bash
FL_BUFFER_TARGET=5 uv run uvicorn fl_async.main:app --port 8000
uv run fl-demo
```

You can also control the number of clients, samples, local epochs per round, and rounds:
```bash
uv run fl-demo --clients 5 --samples 1000 --epochs 1 --rounds 20
```


## Configuration

All settings are environment variables with the `FL_` prefix:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `BUFFER_TARGET` | `3` | Updates required to trigger aggregation |
| `INPUT_SIZE` | `20` | MLP input features |
| `HIDDEN_SIZE` | `16` | MLP hidden units |
| `OUTPUT_SIZE` | `2` | Classification classes |
| `SERVER_LEARNING_RATE` | `1.0` | Mix of averaged update into global model |
| `STALENESS_DECAY` | `1.0` | Strength of stale-update down-weighting |
| `SEED` | `42` | Initial model seed |
| `DEVICE` | *(auto-detected)* | `cpu` or `cuda`; leave unset to auto-detect a working GPU, falling back to CPU if none is available or the driver is incompatible |

For example: `FL_BUFFER_TARGET=5 FL_STALENESS_DECAY=0.5 ...`.

## API

- `GET /health` - liveness check
- `GET /model` - PyTorch architecture specification, serialized `state_dict`, and current version
- `POST /updates` - submit a trained `state_dict`, sample count, and base version
- `GET /status` - buffer and aggregation counters
- `GET /docs` - interactive OpenAPI documentation

The architecture response is declarative. Currently, the server emits a
`sequential` specification containing `linear` and `relu` layers. Clients should construct that  architecture with `build_model`, then load the accompanying state:

```python
from fl_async.model import build_model, load_json_state
from fl_async.schemas import ArchitectureSpec

snapshot = (await api.get("/model")).raise_for_status().json()
spec = ArchitectureSpec.model_validate(snapshot["architecture"])
model = build_model(spec)
load_json_state(model, snapshot["state_dict"])
```

After local PyTorch training, serialize the model with `state_dict_to_json(model)` and submit it as  the `state_dict` field. The server rejects missing, extra, incorrectly shaped, or non-finite tensors.

## Development

```bash
uv run pytest
uv run ruff check .
```


## License

This project is licensed under the **Apache License 2.0**.

See the [`LICENSE`](LICENSE) file for the full license text.

Copyright © 2026 Indigma Innovations.
