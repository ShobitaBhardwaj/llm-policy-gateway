from __future__ import annotations

import importlib.util
import json
from typing import Dict, Iterator

from llm_guard.config import Settings
from llm_guard.errors import GuardrailError, PolicyError
from llm_guard.models import ChatCompletionRequest, ChatCompletionResponse
from llm_guard.policy import PolicyEngine
from llm_guard.providers.echo import EchoProvider
from llm_guard.providers.openai_compatible import OpenAICompatibleProvider
from llm_guard.service import GuardrailService


def build_service_from_env() -> GuardrailService:
    settings = Settings.from_env()
    policy_engine = PolicyEngine.from_file(settings.policy_path)
    providers: Dict[str, object] = {"echo": EchoProvider()}
    if settings.openai_compat_base_url:
        providers["openai_compatible"] = OpenAICompatibleProvider(
            base_url=settings.openai_compat_base_url,
            api_key=settings.openai_compat_api_key or "",
        )
    return GuardrailService(policy_engine=policy_engine, providers=providers)


def create_app():
    if importlib.util.find_spec("fastapi") is None:
        raise RuntimeError("FastAPI is not installed. Install project dependencies before starting the API.")

    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    app = FastAPI(title="LLM Safety Middleware", version="0.1.0")
    service = build_service_from_env()
    settings = Settings.from_env()

    @app.exception_handler(GuardrailError)
    async def handle_guardrail_error(_: Request, exc: GuardrailError):
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())

    @app.exception_handler(PolicyError)
    async def handle_policy_error(_: Request, exc: PolicyError):
        payload = GuardrailError(500, "POLICY_INVALID", str(exc)).to_dict()
        return JSONResponse(status_code=500, content=payload)

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        return service.readiness()

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(request: ChatCompletionRequest):
        return service.handle_completion(request)

    @app.post("/v1/chat/completions/stream")
    async def chat_completions_stream(request: ChatCompletionRequest):
        response = service.handle_completion(request)
        return StreamingResponse(
            _sse_stream(response, settings.stream_chunk_size),
            media_type="text/event-stream",
        )

    return app


def _sse_stream(response: ChatCompletionResponse, chunk_size: int) -> Iterator[str]:
    yield _event("meta", {"request_id": response.request_id, "provider": response.provider, "model": response.model})
    for index in range(0, len(response.content), chunk_size):
        yield _event("chunk", {"content": response.content[index : index + chunk_size]})
    yield _event("done", response.model_dump(mode="json"))


def _event(event: str, payload) -> str:
    return "event: {0}\ndata: {1}\n\n".format(event, json.dumps(payload))


app = create_app() if importlib.util.find_spec("fastapi") is not None else None
