import pytest
from httpx import ASGITransport, AsyncClient

from fl_async.config import Settings
from fl_async.main import create_app


@pytest.mark.asyncio
async def test_api_exposes_only_local_fl_surface():
    app = create_app(Settings(buffer_target=1, input_size=2, hidden_size=2))
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        assert (await client.get("/health")).json() == {"status": "ok"}
        model = (await client.get("/model")).json()
        response = await client.post("/updates", json={
            "client_id": "one", "base_version": 0, "num_samples": 5,
            "state_dict": model["state_dict"],
        })
        assert response.status_code == 202
        assert response.json()["aggregated"] is True
        assert (await client.get("/status")).json()["model_version"] == 1
        assert (await client.get("/admin")).status_code == 404
