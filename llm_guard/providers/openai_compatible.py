from __future__ import annotations

from typing import Dict

import httpx

from llm_guard.models import ChatMessage, ProviderRequest, ProviderResponse
from llm_guard.providers.base import LLMProvider


class OpenAICompatibleProvider(LLMProvider):
    name = "openai_compatible"

    def __init__(self, base_url: str, api_key: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        headers = self._headers()
        payload = {
            "model": request.model,
            "messages": [message.model_dump() for message in request.messages],
            "stream": False,
        }
        payload.update(request.parameters)

        with httpx.Client(timeout=30.0) as client:
            response = client.post("{0}/chat/completions".format(self.base_url), headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        content = self._extract_content(data)
        return ProviderResponse(content=content, raw=data)

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer {0}".format(self.api_key)
        return headers

    def _extract_content(self, payload: Dict) -> str:
        choices = payload.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "".join(parts)
        return str(content)
