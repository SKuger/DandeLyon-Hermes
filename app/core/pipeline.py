"""What happens to a message after the webhook has already answered 200.

This is the part that does not run inside the request. The webhook
returns immediately; this runs afterwards, where it is allowed to take
as long as the model takes.
"""

from __future__ import annotations

import logging
from typing import Protocol

from app.ai.base import ReplyProvider
from app.core.models import InboundMessage, OutboundMessage
from app.core.store import ConversationStore

logger = logging.getLogger(__name__)


class Responder(Protocol):
    """Whatever decides what to say.

    Takes the whole message and not just its text, because anything that
    can act on the conversation needs to know which conversation it is
    acting on.
    """

    async def respond(
        self, message: InboundMessage, history: list[tuple[str, str]]
    ) -> str: ...


class PlainResponder:
    """Adapts a plain `ReplyProvider` to the `Responder` interface.

    Keeps the no-tools path available: a bot that only answers questions
    should not have to carry an agent loop it never uses.
    """

    def __init__(self, provider: ReplyProvider) -> None:
        self._provider = provider

    async def respond(
        self, message: InboundMessage, history: list[tuple[str, str]]
    ) -> str:
        return await self._provider.reply(history, message.text)


class Pipeline:
    def __init__(
        self,
        store: ConversationStore,
        responder: Responder,
        sender,  # object with `async def send(OutboundMessage)`
    ) -> None:
        self._store = store
        self._responder = responder
        self._sender = sender

    async def handle(self, message: InboundMessage) -> OutboundMessage | None:
        # The retry guard. Claimed before any work happens, so a duplicate
        # that arrives while the first copy is still being processed is
        # dropped instead of producing a second reply.
        if await self._store.already_processed(message.message_id):
            logger.info("duplicate ignored: %s", message.message_id)
            return None
        await self._store.mark_processed(message.message_id)

        # Once a person has taken the conversation, the bot stops
        # talking. Two voices answering the same customer is worse than
        # a slow answer, and the handoff is the one state the model is
        # not allowed to overrule.
        if await self._store.awaiting_human(message.conversation_id):
            logger.info(
                "handoff in progress, staying quiet: %s",
                message.conversation_id,
            )
            await self._store.append(
                message.conversation_id, "user", message.text
            )
            return None

        history = await self._store.history(message.conversation_id)
        text = await self._responder.respond(message, history)

        await self._store.append(message.conversation_id, "user", message.text)
        await self._store.append(message.conversation_id, "bot", text)

        reply = OutboundMessage(
            conversation_id=message.conversation_id,
            channel=message.channel,
            recipient_id=message.sender_id,
            text=text,
        )
        await self._sender.send(reply)
        return reply
