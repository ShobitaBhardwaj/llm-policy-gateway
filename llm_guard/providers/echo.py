from __future__ import annotations

import json

from llm_guard.models import ProviderRequest, ProviderResponse
from llm_guard.providers.base import LLMProvider


class EchoProvider(LLMProvider):
    name = "echo"

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        if "mock_response" in request.metadata:
            content = str(request.metadata["mock_response"])
        elif "mock_response_json" in request.metadata:
            content = json.dumps(request.metadata["mock_response_json"])
        else:
            content = request.messages[-1].content
        return ProviderResponse(content=content, raw={"provider": self.name})
