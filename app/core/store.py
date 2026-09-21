"""Conversation state and idempotency.

Two problems, one storage layer:

1. A conversation is a sequence. Answering "how much is it?" needs the
   previous turns, so the last N turns are kept per conversation.
2. Providers retry a webhook when they do not get a fast 200. The same
   message arrives two or three times, and a bot that answers twice is
   immediately obvious to the person on the other side.

`InMemoryStore` is the default so the project runs with no
infrastructure. `RedisStore` is the same interface backed by Redis, for
when more than one worker is running and memory is no longer shared.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Protocol

# Keep the prompt bounded: older turns stop being useful and start
# costing tokens.
MAX_TURNS = 10

# How long a message id is remembered for deduplication. Provider
# retries arrive within seconds; a few minutes is plenty.
SEEN_TTL_SECONDS = 600


class ConversationStore(Protocol):
    async def already_processed(self, message_id: str) -> bool: ...
    async def mark_processed(self, message_id: str) -> None: ...
    async def history(self, conversation_id: str) -> list[tuple[str, str]]: ...
    async def append(
        self, conversation_id: str, role: str, text: str
    ) -> None: ...
    async def open_handoff(self, conversation_id: str, reason: str) -> bool: ...
    async def awaiting_human(self, conversation_id: str) -> bool: ...


class InMemoryStore:
    """Single-process store. Good for tests and for running locally."""

    def __init__(self) -> None:
        self._seen: dict[str, float] = {}
        self._turns: dict[str, deque[tuple[str, str]]] = {}
        self._handoffs: dict[str, str] = {}

    def _expire_seen(self) -> None:
        cutoff = time.monotonic() - SEEN_TTL_SECONDS
        for message_id, stamp in list(self._seen.items()):
            if stamp < cutoff:
                del self._seen[message_id]

    async def already_processed(self, message_id: str) -> bool:
        self._expire_seen()
        return message_id in self._seen

    async def mark_processed(self, message_id: str) -> None:
        self._seen[message_id] = time.monotonic()

    async def history(self, conversation_id: str) -> list[tuple[str, str]]:
        return list(self._turns.get(conversation_id, ()))

    async def append(self, conversation_id: str, role: str, text: str) -> None:
        turns = self._turns.setdefault(conversation_id, deque(maxlen=MAX_TURNS))
        turns.append((role, text))

    async def open_handoff(self, conversation_id: str, reason: str) -> bool:
        if conversation_id in self._handoffs:
            return False
        self._handoffs[conversation_id] = reason
        return True

    async def awaiting_human(self, conversation_id: str) -> bool:
        return conversation_id in self._handoffs


class RedisStore:
    """Shared store, so replicas deduplicate against each other.

    `SET NX` is what makes deduplication safe under concurrency: two
    workers can receive the same retry at the same moment, and exactly
    one of them wins the key.
    """

    def __init__(self, redis) -> None:  # redis.asyncio.Redis
        self._redis = redis

    async def already_processed(self, message_id: str) -> bool:
        was_set = await self._redis.set(
            f"seen:{message_id}", "1", nx=True, ex=SEEN_TTL_SECONDS
        )
        return not was_set

    async def mark_processed(self, message_id: str) -> None:
        # already_processed() claims the key atomically, so there is
        # nothing left to do here.
        return None

    async def history(self, conversation_id: str) -> list[tuple[str, str]]:
        raw = await self._redis.lrange(f"turns:{conversation_id}", 0, -1)
        turns = []
        for item in raw:
            text = item.decode() if isinstance(item, bytes) else item
            role, _, content = text.partition("|")
            turns.append((role, content))
        return turns

    async def append(self, conversation_id: str, role: str, text: str) -> None:
        key = f"turns:{conversation_id}"
        await self._redis.rpush(key, f"{role}|{text}")
        await self._redis.ltrim(key, -MAX_TURNS, -1)

    async def open_handoff(self, conversation_id: str, reason: str) -> bool:
        # Same `SET NX` trick as deduplication, for the same reason: the
        # model can emit the same tool call twice, and two workers can
        # run it at once. Exactly one of them opens the handoff.
        return bool(
            await self._redis.set(
                f"handoff:{conversation_id}", reason, nx=True
            )
        )

    async def awaiting_human(self, conversation_id: str) -> bool:
        return bool(await self._redis.exists(f"handoff:{conversation_id}"))
