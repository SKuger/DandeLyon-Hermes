"""The agent loop: ask, run what was asked for, ask again.

The loop itself is a dozen lines. Everything else in this module exists
because of what goes wrong around it, and that is the part worth
reading:

- **The step budget is the bill, not a safety net.** Every iteration is
  another model call with a longer prompt. An agent that "keeps trying"
  is an agent that spends without a ceiling, so the ceiling is explicit
  and low, and running out of it is logged as the incident it is.
- **A tool that hangs is worse than a tool that fails.** The person is
  waiting on a phone. Each call gets its own timeout, and a timeout is
  fed back as an observation so the model can pick something else.
- **A failed tool is information.** `ToolError` becomes text the model
  reads. It does not end the turn, because "that record does not exist"
  is often exactly what the user needed to hear.
- **The loop never returns empty.** A model that stops with no text is
  a bug, but silence on WhatsApp is the user's problem, not ours.
"""

from __future__ import annotations

import asyncio
import logging

from app.agent.llm import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolCallingProvider,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)
from app.agent.tools import ToolError, ToolRegistry
from app.core.models import InboundMessage

logger = logging.getLogger(__name__)

# Four is enough for "check the time, then look something up, then
# answer". Beyond that the model is usually looping, not working.
MAX_STEPS = 4

# Shorter than the model timeout: a tool is a database or an HTTP call,
# not a generation.
TOOL_TIMEOUT_SECONDS = 10

EXHAUSTED = (
    "Sorry, I could not finish that. A person from the team will follow "
    "up shortly."
)

EMPTY = (
    "Sorry, I did not catch that. Could you say it another way?"
)


async def run_tool(
    registry: ToolRegistry,
    name: str,
    arguments: dict,
    timeout: float = TOOL_TIMEOUT_SECONDS,
) -> tuple[str, bool]:
    """Run one tool call. Returns (what to tell the model, is_error).

    Shared by every runtime on purpose. Two agent implementations that
    disagree about what a timeout means are two different products, and
    the swap this project advertises would not be a swap.
    """
    try:
        async with asyncio.timeout(timeout):
            return await registry.run(name, arguments), False
    except ToolError as exc:
        logger.info("tool %s refused: %s", name, exc)
        return str(exc), True
    except TimeoutError:
        logger.warning("tool %s timed out after %ss", name, timeout)
        return (
            f"{name} took too long and was cancelled. Do not call it "
            f"again this turn."
        ), True


class AgentLoop:
    """Turns a tool-calling model into a single reply string.

    `tools_for` builds the registry per message rather than once at
    startup, because tools are bound to a conversation: escalating has
    to escalate *this* conversation.
    """

    def __init__(
        self,
        provider: ToolCallingProvider,
        tools_for,  # Callable[[InboundMessage], ToolRegistry]
        max_steps: int = MAX_STEPS,
        tool_timeout: float = TOOL_TIMEOUT_SECONDS,
    ) -> None:
        self._provider = provider
        self._tools_for = tools_for
        self._max_steps = max_steps
        self._tool_timeout = tool_timeout

    async def respond(
        self, message: InboundMessage, history: list[tuple[str, str]]
    ) -> str:
        registry: ToolRegistry = self._tools_for(message)
        schemas = registry.schemas()
        messages: list[Message] = [
            UserMessage(text) if role == "user" else AssistantMessage(text)
            for role, text in history
        ]
        messages.append(UserMessage(message.text))

        for step in range(self._max_steps):
            turn = await self._provider.respond(messages, schemas)

            if not turn.tool_calls:
                return turn.text.strip() or EMPTY

            messages.append(turn)
            results = [
                await self._run(registry, call) for call in turn.tool_calls
            ]
            messages.append(ToolResultMessage(tuple(results)))

            logger.info(
                "agent step %d/%d on %s: %s",
                step + 1,
                self._max_steps,
                message.conversation_id,
                ", ".join(call.name for call in turn.tool_calls),
            )

        # The model asked for a tool on every single step and never wrote
        # an answer. That is a prompt or a tool-design problem, and it
        # should be visible as one.
        logger.warning(
            "agent budget exhausted on %s after %d steps",
            message.conversation_id,
            self._max_steps,
        )
        return EXHAUSTED

    async def _run(self, registry: ToolRegistry, call: ToolCall) -> ToolResult:
        content, is_error = await run_tool(
            registry, call.name, call.arguments, self._tool_timeout
        )
        return ToolResult(
            call_id=call.id, content=content, is_error=is_error
        )
