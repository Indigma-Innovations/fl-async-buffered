"""FastAPI entrypoint."""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request

from .config import Settings
from .orchestrator import FedAvgOrchestrator
from .schemas import ModelSnapshot, ServerStatus, UpdateReceipt, UpdateSubmission


def create_app(settings: Settings | None = None) -> FastAPI:
    configuration = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.orchestrator = FedAvgOrchestrator(configuration)
        yield

    app = FastAPI(title="FL Async", version="1.0.0", lifespan=lifespan)

    def orchestrator(request: Request) -> FedAvgOrchestrator:
        return request.app.state.orchestrator

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/model", response_model=ModelSnapshot)
    async def get_model(request: Request) -> ModelSnapshot:
        return await orchestrator(request).snapshot()

    @app.get("/status", response_model=ServerStatus)
    async def get_status(request: Request) -> ServerStatus:
        return await orchestrator(request).status()

    @app.post("/updates", response_model=UpdateReceipt, status_code=202)
    async def post_update(submission: UpdateSubmission, request: Request) -> UpdateReceipt:
        try:
            return await orchestrator(request).submit(submission)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return app


app = create_app()


def run() -> None:
    uvicorn.run("fl_async.main:app", host="127.0.0.1", port=8000)
