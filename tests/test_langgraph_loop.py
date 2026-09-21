"""The LangGraph runtime, held to the same contract as the hand-written one.

Skipped when LangGraph is not installed: it is an optional dependency,
and the default suite has to pass on a machine that only installed
`requirements-dev.txt`.
"""

import pytest

pytest.importorskip("langgraph")

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from app.agent.langgraph_loop import LangGraphLoop  # noqa: E402
from app.agent.loop import EXHAUSTED  # noqa: E402
from app.agent.tools import CurrentTime, ToolError, ToolRegistry  # noqa: E402
from app.core.models import Channel, InboundMessage  # noqa: E402


class ScriptedChatModel(BaseChatModel):
    """Replays a list of replies and, unlike the stock fake, binds tools."""

    replies: list = []
    seen_tools: list = []
    served: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        self.seen_tools = list(tools)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        template = self.replies[min(self.served, len(self.replies) - 1)]
        self.served += 1

        # Every reply needs its own id. LangGraph's `add_messages` reducer
        # deduplicates by id, so handing back the same object twice
        # replaces the earlier copy instead of appending it — the graph
        # then sees a ToolMessage last, routes to END, and a loop that
        # should have run out of budget quietly succeeds instead.
        reply = template.model_copy(update={"id": f"ai_{self.served}"})
        if reply.tool_calls:
            reply = reply.model_copy(
                update={
                    "tool_calls": [
                        {**tool_call, "id": f"call_{self.served}"}
                        for tool_call in reply.tool_calls
                    ]
                }
            )
        return ChatResult(generations=[ChatGeneration(message=reply)])


def make_message(text: str = "hello") -> InboundMessage:
    return InboundMessage(
        message_id="m1",
        conversation_id=InboundMessage.build_conversation_id(
            Channel.WHATSAPP, "573001112233"
        ),
        channel=Channel.WHATSAPP,
        sender_id="573001112233",
        text=text,
    )


def call(name: str, arguments: dict | None = None) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": arguments or {}, "id": "call_1"}],
    )


@pytest.mark.asyncio
async def test_it_answers_without_touching_a_tool():
    model = ScriptedChatModel(replies=[AIMessage(content="We are open until 6pm.")])
    loop = LangGraphLoop(model, lambda _: ToolRegistry([CurrentTime()]))

    assert await loop.respond(make_message(), []) == "We are open until 6pm."


@pytest.mark.asyncio
async def test_our_tools_reach_the_model_without_being_rewritten():
    """The claim this file exists to check: tools are written once."""
    model = ScriptedChatModel(replies=[AIMessage(content="ok")])
    loop = LangGraphLoop(model, lambda _: ToolRegistry([CurrentTime()]))

    await loop.respond(make_message(), [])

    assert [tool["name"] for tool in model.seen_tools] == ["current_time"]
    assert "input_schema" in model.seen_tools[0]


@pytest.mark.asyncio
async def test_a_tool_call_runs_and_the_answer_comes_back():
    model = ScriptedChatModel(
        replies=[call("current_time"), AIMessage(content="It is midday.")]
    )
    loop = LangGraphLoop(model, lambda _: ToolRegistry([CurrentTime()]))

    assert await loop.respond(make_message(), []) == "It is midday."
    assert model.served == 2


@pytest.mark.asyncio
async def test_a_failing_tool_does_not_end_the_turn():
    """Same semantics as the hand-written loop, because it is the same code."""

    class Exploding:
        name = "lookup"
        description = "look something up"
        parameters = {"type": "object", "properties": {}, "required": []}

        async def run(self, **kwargs):
            raise ToolError("no such record")

    model = ScriptedChatModel(
        replies=[call("lookup"), AIMessage(content="I could not find it.")]
    )
    loop = LangGraphLoop(model, lambda _: ToolRegistry([Exploding()]))

    assert await loop.respond(make_message(), []) == "I could not find it."


@pytest.mark.asyncio
async def test_a_model_that_only_calls_tools_runs_out_of_budget():
    """The budget survives the move from a `for` to a recursion limit."""
    model = ScriptedChatModel(replies=[call("current_time")])
    loop = LangGraphLoop(
        model, lambda _: ToolRegistry([CurrentTime()]), max_steps=3
    )

    assert await loop.respond(make_message(), []) == EXHAUSTED
