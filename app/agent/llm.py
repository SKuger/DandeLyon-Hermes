"""The model boundary, widened to allow tool calls.

`ReplyProvider` answers with a sentence. A tool-using model answers with
either a sentence or a request to run something, and the conversation
then has to carry that request and its result back into the next call.

These types are the project's own, not the SDK's. Translating to a
vendor shape is five lines inside each provider; leaking the vendor
shape into the loop would make the loop untestable without a key, which
is exactly what this repository refuses to do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    id: str
    """Correlates the call with its result. Providers require the pairing."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False
    """A failed tool is still an observation, not an exception, once the
    loop has decided to keep going."""


@dataclass(frozen=True)
class UserMessage:
    text: str


@dataclass(frozen=True)
class AssistantMessage:
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class ToolResultMessage:
    results: tuple[ToolResult, ...] = field(default_factory=tuple)


Message = UserMessage | AssistantMessage | ToolResultMessage


class ToolCallingProvider(Protocol):
    async def respond(
        self, messages: list[Message], tools: list[dict[str, Any]]
    ) -> AssistantMessage: ...


class ScriptedToolProvider:
    """Deterministic stand-in for a model that can call tools.

    Not a mock in the test-double sense: it is the default provider, so
    `docker compose up` gives a working tool-using agent with no API key
    and no network. It routes on keywords, which is a terrible way to
    build a product and a very good way to exercise every branch of the
    loop in CI.
    """

    #: keyword -> (tool name, arguments)
    ROUTES: tuple[tuple[tuple[str, ...], str, dict[str, Any]], ...] = (
        (
            ("human", "person", "agent", "someone", "manager"),
            "escalate_to_human",
            {"reason": "The user asked to speak to a person."},
        ),
        (("time", "date", "today", "now"), "current_time", {}),
    )

    async def respond(
        self, messages: list[Message], tools: list[dict[str, Any]]
    ) -> AssistantMessage:
        available = {tool["name"] for tool in tools}

        # A tool has just run: report it and stop. One hop is enough to
        # prove the loop closes.
        last = messages[-1] if messages else None
        if isinstance(last, ToolResultMessage):
            return AssistantMessage(text=_render(last.results))

        question = _last_user_text(messages)
        text = question.lower()
        for keywords, name, arguments in self.ROUTES:
            if name in available and any(word in text for word in keywords):
                return AssistantMessage(
                    tool_calls=(
                        ToolCall(id=f"call_{name}", name=name,
                                 arguments=dict(arguments)),
                    )
                )

        # Anything else is a question for the knowledge base. A real
        # model decides this; here it is the fallback, which is enough to
        # exercise retrieval end to end without a key.
        if "search_knowledge_base" in available:
            return AssistantMessage(
                tool_calls=(
                    ToolCall(
                        id="call_search",
                        name="search_knowledge_base",
                        arguments={"query": question},
                    ),
                )
            )

        turn = sum(1 for m in messages if isinstance(m, UserMessage))
        return AssistantMessage(text=f"(turn {turn}) You said: {question}")


def _last_user_text(messages: list[Message]) -> str:
    for message in reversed(messages):
        if isinstance(message, UserMessage):
            return message.text
    return ""


CANNOT_HELP = (
    "I do not have that information. Would you like me to pass you to a "
    "person?"
)

PASSAGE_CHARS = 320


def _render(results: tuple[ToolResult, ...]) -> str:
    """Turn tool output into something a person can read.

    A model does this with judgement. This does it with three rules,
    which is the honest cost of having no model: tool results are
    written for a reader, and an unread instruction must never reach the
    user's phone.
    """
    from app.agent.tools import NO_MATCH

    rendered = []
    for result in results:
        if result.is_error or result.content.startswith(NO_MATCH):
            rendered.append(CANNOT_HELP)
        elif result.content.startswith("["):
            # A retrieved passage: drop the citation header, keep prose.
            _, _, body = result.content.partition("\n")
            text = " ".join(body.split())
            rendered.append(
                text[:PASSAGE_CHARS].rstrip() + "…"
                if len(text) > PASSAGE_CHARS
                else text
            )
        else:
            rendered.append(result.content)
    return " ".join(rendered)
