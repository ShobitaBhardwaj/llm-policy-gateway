from __future__ import annotations

import os
from typing import Optional

from pydantic import BaseModel


class Settings(BaseModel):
    policy_path: str = "config/policies/default.yaml"
    openai_compat_base_url: Optional[str] = None
    openai_compat_api_key: Optional[str] = None
    stream_chunk_size: int = 120

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            policy_path=os.getenv("POLICY_PATH", "config/policies/default.yaml"),
            openai_compat_base_url=os.getenv("OPENAI_COMPAT_BASE_URL"),
            openai_compat_api_key=os.getenv("OPENAI_COMPAT_API_KEY"),
            stream_chunk_size=int(os.getenv("STREAM_CHUNK_SIZE", "120")),
        )
