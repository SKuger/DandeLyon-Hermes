"""Real model provider.

Kept deliberately small. The interesting engineering is not the API
call, it is what happens around it: a bounded timeout, and a reply the
user still receives when the model does not answer.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a customer service assistant answering on WhatsApp, "
    "Instagram and Messenger. Keep replies short: two or three "
    "sentences, no formatting. If you do not know something, say so and "
    "offer to pass the question to a person."
)

# Past this point the user has been waiting too long for a chat message.
REPLY_TIMEOUT_SECONDS = 20

FALLBACK = (
    "Sorry, I could not answer that right now. A person from the team "
    "will follow up shortly."
)


class AnthropicProvider:
    def __init__(self, api_key: str, model: str) -> None:
        # Imported here so the package stays optional: the project runs
        # without the SDK installed as long as EchoProvider is used.
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model

    async def reply(self, history: list[tuple[str, str]], message: str) -> str:
        messages = [
            {"role": "assistant" if role == "bot" else "user", "content": text}
            for role, text in history
        ]
        messages.append({"role": "user", "content": message})

        try:
            async with asyncio.timeout(REPLY_TIMEOUT_SECONDS):
                response = await self._client.messages.create(
                    model=self._model,
                    max_tokens=300,
                    system=SYSTEM_PROMPT,
                    messages=messages,
                )
        except Exception as exc:  # noqa: BLE001
            # Includes TimeoutError from the block above.
            # A failed model call must not become silence on the user's
            # phone. Log it, answer something honest, move on.
            logger.warning("model call failed: %s", exc)
            return FALLBACK

        return "".join(
            block.text for block in response.content if block.type == "text"
        ).strip() or FALLBACK
