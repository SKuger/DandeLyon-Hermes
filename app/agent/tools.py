"""What the model is allowed to do, and what happens when it goes wrong.

A tool is the point where a language model stops producing text and
starts having effects. That changes the failure modes: a bad sentence is
embarrassing, a bad tool call books the wrong appointment twice.

So the contract here is deliberately narrow:

- Tools declare their arguments as JSON Schema. The model gets that
  schema, not a prose description of what to type.
- Tools raise `ToolError` for anything the model could plausibly fix by
  trying again differently. Everything else is a bug and propagates.
- A tool never returns `None`. The loop needs something to feed back.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol


class ToolError(Exception):
    """A tool failed in a way the model should be told about.

    Bad arguments, a missing record, a refused action. The loop turns
    this into an observation and lets the model try something else.
    Anything that is not a `ToolError` is treated as a defect and is
    allowed to crash the turn.
    """


class Tool(Protocol):
    name: str
    description: str
    parameters: dict[str, Any]
    """JSON Schema for the arguments. Providers send this to the model.

    Implementations declare those same arguments as named parameters on
    `run`, so that a hallucinated or missing argument is caught by
    Python's own binding rather than by a hand-written check that would
    drift from the schema.
    """

    async def run(self, **kwargs: Any) -> str: ...


class ToolRegistry:
    """The set of tools available for one conversation.

    Held per-pipeline rather than globally so that a future version can
    give different channels different capabilities without rewriting the
    loop.
    """

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in self._tools.values()
        ]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    async def run(self, name: str, arguments: dict[str, Any]) -> str:
        """Run a tool by name, converting expected failures into text.

        The model picks the name, so an unknown name is not a bug on our
        side: it is a hallucination, and the fix is to tell the model
        which names exist.
        """
        tool = self._tools.get(name)
        if tool is None:
            known = ", ".join(sorted(self._tools)) or "none"
            raise ToolError(f"no tool named {name!r}. Available: {known}")
        try:
            return await tool.run(**arguments)
        except ToolError:
            raise
        except TypeError as exc:
            # Wrong or missing arguments against the declared schema.
            raise ToolError(f"{name} rejected those arguments: {exc}") from exc


class CurrentTime:
    """The model has no clock. Anything about "today" needs this."""

    name = "current_time"
    description = (
        "Return the current UTC date and time in ISO 8601 format. Use this "
        "before any reasoning that depends on today's date."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def run(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")


class EscalateToHuman:
    """Hand the conversation to a person and stop answering.

    The most important tool in a support bot is the one that admits the
    bot should stop. It is also the one with a real side effect, so it
    is written to be safe to call twice: the second call reports the
    existing handoff instead of opening another one.
    """

    name = "escalate_to_human"
    description = (
        "Hand this conversation to a human agent. Use it when the user asks "
        "for a person, is angry, or asks something you cannot answer with "
        "the other tools. After calling it, tell the user a person will "
        "take over."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "One short sentence on why a human is needed.",
            }
        },
        "required": ["reason"],
    }

    def __init__(self, store, conversation_id: str) -> None:
        self._store = store
        self._conversation_id = conversation_id

    async def run(self, reason: str) -> str:
        created = await self._store.open_handoff(self._conversation_id, reason)
        if not created:
            return "This conversation is already waiting for a human agent."
        return "A human agent has been notified and will take over."


NO_MATCH = (
    "Nothing in the knowledge base matches that question. Tell the user "
    "you do not have that information and offer to pass them to a person."
)
"""What retrieval says when it comes back empty.

Addressed to the model, not the user: it is an instruction to admit
ignorance rather than a sentence to repeat. Exported so a provider that
cannot read can still recognise it.
"""


class SearchKnowledgeBase:
    """Retrieval, with an explicit way to come back empty.

    The failure that matters here is not a bad search, it is a search
    that returns three loosely related paragraphs and lets the model
    build an answer out of them. When nothing clears the threshold this
    says so in plain words, which is the instruction the model needs to
    admit it does not know.
    """

    name = "search_knowledge_base"
    description = (
        "Search the support knowledge base for an answer. Returns the "
        "matching passages with their sources, or tells you that nothing "
        "matched. If nothing matched, say you do not know and offer to "
        "pass the question to a person. Never answer from memory."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The user's question, in their own words. Do not "
                    "shorten it to keywords."
                ),
            }
        },
        "required": ["query"],
    }

    def __init__(self, index, limit: int = 3) -> None:
        self._index = index
        self._limit = limit

    async def run(self, query: str) -> str:
        if not query.strip():
            raise ToolError("query must not be empty")

        hits = await self._index.search(query, limit=self._limit)
        if not hits:
            return NO_MATCH

        return "\n\n".join(
            f"[{chunk.citation}] (score {score:.2f})\n{chunk.text}"
            for chunk, score in hits
        )
