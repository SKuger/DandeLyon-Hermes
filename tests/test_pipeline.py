"""Retries, context and failure behaviour."""

import pytest

from app.ai.base import EchoProvider
from app.core.models import Channel, InboundMessage
from app.core.pipeline import Pipeline, PlainResponder
from app.core.sender import LoggingSender
from app.core.store import MAX_TURNS, InMemoryStore


def make_message(message_id: str = "m1", text: str = "hola") -> InboundMessage:
    return InboundMessage(
        message_id=message_id,
        conversation_id=InboundMessage.build_conversation_id(
            Channel.WHATSAPP, "573001112233"
        ),
        channel=Channel.WHATSAPP,
        sender_id="573001112233",
        text=text,
    )


def build() -> tuple[Pipeline, LoggingSender, InMemoryStore]:
    store, sender = InMemoryStore(), LoggingSender()
    return Pipeline(store, PlainResponder(EchoProvider()), sender), sender, store


@pytest.mark.asyncio
async def test_a_message_gets_exactly_one_reply():
    pipeline, sender, _ = build()

    await pipeline.handle(make_message())

    assert len(sender.sent) == 1
    assert sender.sent[0].recipient_id == "573001112233"
    assert sender.sent[0].channel is Channel.WHATSAPP


@pytest.mark.asyncio
async def test_a_retried_message_is_not_answered_twice():
    """Meta resends when the 200 is slow. The user must not see two replies."""
    pipeline, sender, _ = build()
    message = make_message()

    await pipeline.handle(message)
    await pipeline.handle(message)
    await pipeline.handle(message)

    assert len(sender.sent) == 1


@pytest.mark.asyncio
async def test_different_messages_are_both_answered():
    pipeline, sender, _ = build()

    await pipeline.handle(make_message("m1"))
    await pipeline.handle(make_message("m2"))

    assert len(sender.sent) == 2


@pytest.mark.asyncio
async def test_history_is_carried_between_turns():
    pipeline, sender, _ = build()

    await pipeline.handle(make_message("m1", "first"))
    await pipeline.handle(make_message("m2", "second"))

    # EchoProvider numbers the turn from the history it was given, so a
    # reply of "turn 2" proves the first turn was still there.
    assert sender.sent[1].text.startswith("(turn 2)")


@pytest.mark.asyncio
async def test_history_stays_bounded():
    """Otherwise the prompt grows forever, and so does the bill."""
    pipeline, _, store = build()

    for index in range(MAX_TURNS + 10):
        await pipeline.handle(make_message(f"m{index}", f"message {index}"))

    history = await store.history(
        InboundMessage.build_conversation_id(Channel.WHATSAPP, "573001112233")
    )
    assert len(history) == MAX_TURNS


@pytest.mark.asyncio
async def test_a_failing_provider_does_not_send_a_broken_reply():
    """A provider blowing up must not turn into garbage on the user's phone.

    The pipeline lets the error out on purpose: the caller in main.py
    logs it. What must not happen is a half-built message going out.
    """

    class BrokenProvider:
        async def reply(self, history, message):
            raise RuntimeError("model is down")

    store, sender = InMemoryStore(), LoggingSender()
    pipeline = Pipeline(store, PlainResponder(BrokenProvider()), sender)

    with pytest.raises(RuntimeError):
        await pipeline.handle(make_message())

    assert sender.sent == []
