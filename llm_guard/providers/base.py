from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from llm_guard.models import ProviderRequest, ProviderResponse


class LLMProvider(ABC):
    name = "provider"

    @abstractmethod
    def complete(self, request: ProviderRequest) -> ProviderResponse:
        raise NotImplementedError

    def stream(self, request: ProviderRequest, chunk_size: int = 120) -> Iterator[str]:
        response = self.complete(request)
        for index in range(0, len(response.content), chunk_size):
            yield response.content[index : index + chunk_size]
