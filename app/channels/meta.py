"""Adapters for the three Meta channels.

WhatsApp Cloud API and the Messenger/Instagram Send API share a webhook
envelope but disagree on everything inside it. WhatsApp nests messages
under `entry[].changes[].value.messages[]`; Messenger and Instagram use
`entry[].messaging[]` with a different field layout.

Each adapter absorbs that difference so the rest of the service does not
have to know about it.
"""

from __future__ import annotations

from typing import Any

from app.core.models import Channel, InboundMessage, OutboundMessage


def _text_of(message: dict[str, Any]) -> str:
    """WhatsApp puts the body under `text.body`; the others under `text`."""
    text = message.get("text")
    if isinstance(text, dict):
        return text.get("body", "")
    return text or ""


class WhatsAppAdapter:
    channel = Channel.WHATSAPP

    def parse(self, payload: dict[str, Any]) -> list[InboundMessage]:
        messages: list[InboundMessage] = []
        for entry in payload.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                for raw in value.get("messages", []):
                    # Status updates ride the same webhook. Skip anything
                    # that is not a text message we can answer.
                    if raw.get("type") not in (None, "text"):
                        continue
                    text = _text_of(raw)
                    if not text:
                        continue
                    sender_id = raw.get("from", "")
                    messages.append(
                        InboundMessage(
                            message_id=raw["id"],
                            conversation_id=InboundMessage.build_conversation_id(
                                self.channel, sender_id
                            ),
                            channel=self.channel,
                            sender_id=sender_id,
                            text=text,
                        )
                    )
        return messages

    def render(self, message: OutboundMessage) -> dict[str, Any]:
        return {
            "messaging_product": "whatsapp",
            "to": message.recipient_id,
            "type": "text",
            "text": {"body": message.text},
        }


class _MessagingAdapter:
    """Shared shape for Messenger and Instagram."""

    channel: Channel

    def parse(self, payload: dict[str, Any]) -> list[InboundMessage]:
        messages: list[InboundMessage] = []
        for entry in payload.get("entry", []):
            for event in entry.get("messaging", []):
                raw = event.get("message")
                if not raw:
                    # Postbacks, reads and echoes arrive here too.
                    continue
                if raw.get("is_echo"):
                    # Our own reply coming back. Answering it would loop.
                    continue
                text = _text_of(raw)
                if not text:
                    continue
                sender_id = event.get("sender", {}).get("id", "")
                messages.append(
                    InboundMessage(
                        message_id=raw["mid"],
                        conversation_id=InboundMessage.build_conversation_id(
                            self.channel, sender_id
                        ),
                        channel=self.channel,
                        sender_id=sender_id,
                        text=text,
                    )
                )
        return messages

    def render(self, message: OutboundMessage) -> dict[str, Any]:
        return {
            "recipient": {"id": message.recipient_id},
            "message": {"text": message.text},
            "messaging_type": "RESPONSE",
        }


class MessengerAdapter(_MessagingAdapter):
    channel = Channel.MESSENGER


class InstagramAdapter(_MessagingAdapter):
    channel = Channel.INSTAGRAM


ADAPTERS = {
    Channel.WHATSAPP: WhatsAppAdapter(),
    Channel.MESSENGER: MessengerAdapter(),
    Channel.INSTAGRAM: InstagramAdapter(),
}
