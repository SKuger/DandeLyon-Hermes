"""Anthropic behind the tool-calling boundary.

All this class does is translate. The project's `Message` types go in,
the vendor's content blocks go out, and the agent loop never learns
which vendor is on the other side. Swapping providers should mean
writing another file this size, not editing the loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.agent.llm import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a customer service assistant answering on WhatsApp, "
    "Instagram and Messenger. Keep replies short: two or three "
    "sentences, no formatting. Use a tool when it would give you a fact "
    "you do not have; do not guess dates or account details. If the user "
    "wants a person, escalate instead of apologising repeatedly."
)

# One step of the loop, not the whole turn. The loop may take several.
STEP_TIMEOUT_SECONDS = 20


class AnthropicAgentProvider:
    def __init__(self, api_key: str, model: str) -> None:
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model

    async def respond(
        self, messages: list[Message], tools: list[dict[str, Any]]
    ) -> AssistantMessage:
        try:
            async with asyncio.timeout(STEP_TIMEOUT_SECONDS):
                response = await self._client.messages.create(
                    model=self._model,
                    max_tokens=600,
                    system=SYSTEM_PROMPT,
                    tools=tools,
                    messages=_to_vendor(messages),
                )
        except Exception as exc:  # noqa: BLE001
            # Returning text rather than raising ends the loop cleanly
            # with something the user can read. Raising here would lose
            # the turn entirely.
            logger.warning("model step failed: %s", exc)
            return AssistantMessage(
                text=(
                    "Sorry, I could not answer that right now. A person "
                    "from the team will follow up shortly."
                )
            )

        text = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        calls = tuple(
            ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
            for block in response.content
            if block.type == "tool_use"
        )
        return AssistantMessage(text=text, tool_calls=calls)


def _to_vendor(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, UserMessage):
            out.append({"role": "user", "content": message.text})

        elif isinstance(message, AssistantMessage):
            blocks: list[dict[str, Any]] = []
            if message.text:
                blocks.append({"type": "text", "text": message.text})
            blocks.extend(
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                }
                for call in message.tool_calls
            )
            if blocks:
                out.append({"role": "assistant", "content": blocks})

        elif isinstance(message, ToolResultMessage):
            # Tool results are sent as a user turn. That is the vendor's
            # convention, not ours, which is why it is confined to here.
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": result.call_id,
                            "content": result.content,
                            "is_error": result.is_error,
                        }
                        for result in message.results
                    ],
                }
            )
    return out
