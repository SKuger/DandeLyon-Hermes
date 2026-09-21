"""Configuration, read once from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_secret: str = os.getenv("META_APP_SECRET", "")
    verify_token: str = os.getenv("META_VERIFY_TOKEN", "dev-verify-token")
    graph_token: str = os.getenv("META_GRAPH_TOKEN", "")
    phone_number_id: str = os.getenv("META_PHONE_NUMBER_ID", "")
    page_id: str = os.getenv("META_PAGE_ID", "")

    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    model: str = os.getenv("MODEL", "claude-sonnet-5")

    # Anthropic does not serve embeddings; with no key the knowledge base
    # is indexed with the offline hashing embedder instead.
    voyage_api_key: str = os.getenv("VOYAGE_API_KEY", "")
    knowledge_dir: str = os.getenv("KNOWLEDGE_DIR", "knowledge")

    redis_url: str = os.getenv("REDIS_URL", "")

    @property
    def verify_signatures(self) -> bool:
        """No secret configured means running locally without Meta.

        Refusing every request in that case would make the project
        impossible to try; this is the one place where the check relaxes,
        and it is loud about why.
        """
        return bool(self.app_secret)


settings = Settings()
