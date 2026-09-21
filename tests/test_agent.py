"""The agent loop, tested through its failure modes.

The happy path is one test. The rest of this file is the reason the
loop exists in the shape it does.
"""

import asyncio

import pytest

from app.agent.llm import AssistantMessage, ScriptedToolProvider, ToolCall
from app.agent.loop import EMPTY, EXHAUSTED, AgentLoop
from app.agent.tools import (
    CurrentTime,
    EscalateToHuman,
    ToolError,
    ToolRegistry,
)
from app.core.models import Channel, InboundMessage
from app.core.pipeline import Pipeline
from app.core.sender import LoggingSender
from app.core.store import InMemoryStore


def make_message(message_id: str = "m1", text: str = "hello") -> InboundMessage:
    return InboundMessage(
        message_id=message_id,
        conversation_id=InboundMessage.build_conversation_id(
            Channel.WHATSAPP, "573001112233"
        ),
        channel=Channel.WHATSAPP,
        sender_id="573001112233",
        text=text,
    )


def build(provider=None, store=None, **kwargs):
    store = store or InMemoryStore()

    def tools_for(message):
        return ToolRegistry(
            [CurrentTime(), EscalateToHuman(store, message.conversation_id)]
        )

    loop = AgentLoop(
        provider=provider or ScriptedToolProvider(),
        tools_for=tools_for,
        **kwargs,
    )
    return loop, store


# --- tools ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_tool_name_is_reported_not_raised():
    """The model chooses the name, so a wrong one is its mistake to fix."""
    registry = ToolRegistry([CurrentTime()])

    with pytest.raises(ToolError) as exc:
        await registry.run("send_refund", {})

    assert "current_time" in str(exc.value)


@pytest.mark.asyncio
async def test_wrong_arguments_become_a_tool_error():
    registry = ToolRegistry([EscalateToHuman(InMemoryStore(), "c1")])

    with pytest.raises(ToolError):
        await registry.run("escalate_to_human", {})  # `reason` is required


@pytest.mark.asyncio
async def test_escalating_twice_does_not_open_two_handoffs():
    """Models repeat themselves, and webhooks arrive twice."""
    store = InMemoryStore()
    registry = ToolRegistry([EscalateToHuman(store, "c1")])

    first = await registry.run("escalate_to_human", {"reason": "angry"})
    second = await registry.run("escalate_to_human", {"reason": "angry"})

    assert "notified" in first
    assert "already" in second


# --- the loop ------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_tool_call_comes_back_as_an_answer():
    loop, _ = build()

    reply = await loop.respond(make_message(text="what time is it?"), [])

    assert reply
    assert "T" in reply  # an ISO timestamp came back through the loop


@pytest.mark.asyncio
async def test_a_failing_tool_does_not_end_the_turn():
    """"That record does not exist" is often the answer the user needed."""

    class Exploding:
        name = "lookup"
        description = "look something up"
        parameters = {"type": "object", "properties": {}, "required": []}

        async def run(self, **kwargs):
            raise ToolError("no such record")

    class CallsLookupOnce:
        def __init__(self):
            self.calls = 0

        async def respond(self, messages, tools):
            self.calls += 1
            if self.calls == 1:
                return AssistantMessage(
                    tool_calls=(ToolCall(id="1", name="lookup", arguments={}),)
                )
            return AssistantMessage(text="I could not find that record.")

    loop = AgentLoop(
        provider=CallsLookupOnce(), tools_for=lambda m: ToolRegistry([Exploding()])
    )

    reply = await loop.respond(make_message(), [])

    assert reply == "I could not find that record."


@pytest.mark.asyncio
async def test_a_hanging_tool_is_cancelled_and_reported():
    """The user is waiting on a phone. A tool does not get to hang forever."""

    class Hangs:
        name = "slow"
        description = "never returns"
        parameters = {"type": "object", "properties": {}, "required": []}

        async def run(self, **kwargs):
            await asyncio.sleep(60)
            return "done"

    seen = []

    class WatchesResults:
        async def respond(self, messages, tools):
            if not seen:
                seen.append(True)
                return AssistantMessage(
                    tool_calls=(ToolCall(id="1", name="slow", arguments={}),)
                )
            observation = messages[-1].results[0]
            return AssistantMessage(
                text=f"error={observation.is_error}"
            )

    loop = AgentLoop(
        provider=WatchesResults(),
        tools_for=lambda m: ToolRegistry([Hangs()]),
        tool_timeout=0.05,
    )

    reply = await loop.respond(make_message(), [])

    assert reply == "error=True"


@pytest.mark.asyncio
async def test_a_model_that_only_calls_tools_runs_out_of_budget():
    """Every step is another billed call. The ceiling has to be real."""

    class NeverAnswers:
        def __init__(self):
            self.steps = 0

        async def respond(self, messages, tools):
            self.steps += 1
            return AssistantMessage(
                tool_calls=(ToolCall(id="1", name="current_time", arguments={}),)
            )

    provider = NeverAnswers()
    loop = AgentLoop(
        provider=provider,
        tools_for=lambda m: ToolRegistry([CurrentTime()]),
        max_steps=3,
    )

    reply = await loop.respond(make_message(), [])

    assert reply == EXHAUSTED
    assert provider.steps == 3


@pytest.mark.asyncio
async def test_an_empty_answer_is_never_sent_as_silence():
    class SaysNothing:
        async def respond(self, messages, tools):
            return AssistantMessage(text="   ")

    loop = AgentLoop(
        provider=SaysNothing(), tools_for=lambda m: ToolRegistry([])
    )

    assert await loop.respond(make_message(), []) == EMPTY


# --- handoff, end to end -------------------------------------------------


@pytest.mark.asyncio
async def test_the_bot_goes_quiet_once_a_human_has_the_conversation():
    """Two voices answering the same customer is worse than a slow reply."""
    store, sender = InMemoryStore(), LoggingSender()
    loop, _ = build(store=store)
    pipeline = Pipeline(store, loop, sender)

    await pipeline.handle(make_message("m1", "I want to talk to a person"))
    assert len(sender.sent) == 1

    await pipeline.handle(make_message("m2", "are you there?"))
    await pipeline.handle(make_message("m3", "hello?"))

    assert len(sender.sent) == 1
    assert await store.awaiting_human(
        InboundMessage.build_conversation_id(Channel.WHATSAPP, "573001112233")
    )
