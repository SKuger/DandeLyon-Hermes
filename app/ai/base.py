"""The AI layer, behind an interface.

Two reasons this is a Protocol and not a direct SDK call:

- The repository has to run with no API key, or nobody can try it.
- Model providers fail. They time out, they rate-limit, they return
  nonsense. Everything above this boundary keeps working when they do.
"""

from __future__ import annotations

from typing import Protocol


class ReplyProvider(Protocol):
    async def reply(
        self, history: list[tuple[str, str]], message: str
    ) -> str: ...


class EchoProvider:
    """Deterministic provider: no key, no network, no cost.

    This is the default, which is what makes `docker compose up` and the
    test suite work out of the box.
    """

    async def reply(self, history: list[tuple[str, str]], message: str) -> str:
        turn = len([role for role, _ in history if role == "user"]) + 1
        return f"(turn {turn}) You said: {message}"
