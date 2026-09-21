"""Dense embeddings, for when lexical retrieval is not enough.

This is the optional half of retrieval. The default index
([`app/rag/index.py`](index.py)) is lexical and needs no model, no key
and no network, which is what lets the project run and be tested
without an account anywhere.

An embedding model earns its cost when users ask for "money back" and
the document says "refund" — vocabulary that does not overlap but means
the same thing. Lexical search cannot bridge that and will confidently
return nothing. Both implement the same `VectorIndex` Protocol, so the
choice is one line of wiring.
"""

from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    dimensions: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class VoyageEmbedder:
    """Anthropic does not serve embeddings; their docs point at Voyage."""

    def __init__(
        self, api_key: str, model: str = "voyage-3", dimensions: int = 1024
    ) -> None:
        self._api_key = api_key
        self._model = model
        self.dimensions = dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import httpx

        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(
                "https://api.voyageai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"input": texts, "model": self._model},
            )
            response.raise_for_status()
            payload = response.json()

        # The API does not promise input order back.
        ordered = sorted(payload["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in ordered]
