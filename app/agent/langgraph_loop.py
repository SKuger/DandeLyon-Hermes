"""The same agent, as a LangGraph state machine.

This exists to make an architectural claim checkable. The project says
the runtime sits behind a boundary and that a framework is a swappable
implementation detail; this is the other implementation, satisfying the
same `Responder` contract, driving the same `ToolRegistry`, with the
same failure semantics through the same `run_tool`.

What changes is who owns the control flow. `AgentLoop` owns it in a
`for`. Here the graph does, and the budget stops being a loop counter
and becomes a recursion limit.

What does not change is the tools. Writing a tool twice, once per
framework, would be the sign that the boundary was decorative.

Optional. LangGraph is not in `requirements.txt`, because the default
path has to run with no dependencies beyond FastAPI. Install it with:

    pip install -r requirements-langgraph.txt
"""

from __future__ import annotations

import logging

from app.agent.loop import EMPTY, EXHAUSTED, MAX_STEPS, TOOL_TIMEOUT_SECONDS
from app.agent.loop import run_tool
from app.agent.tools import ToolRegistry
from app.core.models import InboundMessage

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a customer service assistant answering on WhatsApp, "
    "Instagram and Messenger. Keep replies short: two or three "
    "sentences, no formatting. Use a tool when it would give you a fact "
    "you do not have; do not guess dates or account details."
)


class LangGraphLoop:
    """A `Responder` backed by a compiled LangGraph.

    The graph is compiled per message. Tools are bound to a
    conversation — escalating has to escalate *this* one — and a
    compiled graph closes over them. Hoisting the compile out would mean
    threading the registry through the graph's config, which buys
    microseconds and costs the clarity this file exists to show.
    """

    def __init__(
        self,
        chat_model,  # a LangChain BaseChatModel supporting bind_tools
        tools_for,  # Callable[[InboundMessage], ToolRegistry]
        max_steps: int = MAX_STEPS,
        tool_timeout: float = TOOL_TIMEOUT_SECONDS,
    ) -> None:
        self._model = chat_model
        self._tools_for = tools_for
        self._max_steps = max_steps
        self._tool_timeout = tool_timeout

    async def respond(
        self, message: InboundMessage, history: list[tuple[str, str]]
    ) -> str:
        from langchain_core.messages import AIMessage, HumanMessage
        from langgraph.errors import GraphRecursionError

        registry = self._tools_for(message)
        graph = self._build(registry)

        messages = [
            HumanMessage(text) if role == "user" else AIMessage(text)
            for role, text in history
        ]
        messages.append(HumanMessage(message.text))

        try:
            # Each step is two nodes, model then tools, plus the final
            # model call that answers. Exceeding it means the model kept
            # asking for tools and never wrote a reply.
            final = await graph.ainvoke(
                {"messages": messages},
                config={"recursion_limit": self._max_steps * 2 + 1},
            )
        except GraphRecursionError:
            logger.warning(
                "langgraph budget exhausted on %s after %d steps",
                message.conversation_id,
                self._max_steps,
            )
            return EXHAUSTED

        return _text_of(final["messages"][-1]) or EMPTY

    def _build(self, registry: ToolRegistry):
        from langchain_core.messages import SystemMessage, ToolMessage
        from langgraph.graph import END, START, MessagesState, StateGraph

        model = self._model.bind_tools(_as_langchain_tools(registry))
        timeout = self._tool_timeout

        async def call_model(state):
            reply = await model.ainvoke(
                [SystemMessage(SYSTEM_PROMPT), *state["messages"]]
            )
            return {"messages": [reply]}

        async def call_tools(state):
            last = state["messages"][-1]
            results = []
            for call in last.tool_calls:
                content, is_error = await run_tool(
                    registry, call["name"], call["args"], timeout
                )
                results.append(
                    ToolMessage(
                        content=content,
                        tool_call_id=call["id"],
                        status="error" if is_error else "success",
                    )
                )
            return {"messages": results}

        def route(state):
            last = state["messages"][-1]
            return "tools" if getattr(last, "tool_calls", None) else END

        graph = StateGraph(MessagesState)
        graph.add_node("model", call_model)
        graph.add_node("tools", call_tools)
        graph.add_edge(START, "model")
        graph.add_conditional_edges("model", route, ["tools", END])
        graph.add_edge("tools", "model")
        return graph.compile()


def _as_langchain_tools(registry: ToolRegistry) -> list[dict]:
    """Our schemas, in the shape `bind_tools` expects.

    `ToolRegistry.schemas()` already emits Anthropic's
    `name`/`description`/`input_schema`, which LangChain accepts
    directly. Nine lines of translation is the entire cost of not
    writing every tool twice.
    """
    return [
        {
            "name": schema["name"],
            "description": schema["description"],
            "input_schema": schema["input_schema"],
        }
        for schema in registry.schemas()
    ]


def _text_of(message) -> str:
    """Anthropic returns content as blocks; other providers as a string."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    return " ".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ).strip()
