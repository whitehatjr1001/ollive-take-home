import uvicorn
from fastapi import FastAPI

from ollie_assistants.api import chat, evals, health, observability
from ollie_assistants.assistant.factory import AssistantFactory
from ollie_assistants.settings import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()
    app = FastAPI(title=resolved.app_name)
    assistant_factory = AssistantFactory(resolved)
    app.state.assistant_factory = assistant_factory
    app.state.assistants = {
        "oss": assistant_factory.create_oss(),
        "frontier": assistant_factory.create_frontier(),
    }
    app.include_router(health.router)
    if resolved.chat_enabled:
        app.include_router(chat.router)
    if resolved.evals_enabled:
        app.include_router(evals.router)
    if resolved.analytics_enabled:
        app.include_router(observability.router)
    return app


app = create_app()


def run() -> None:
    uvicorn.run("ollie_assistants.main:app", host="127.0.0.1", port=8000, reload=True)
