"""Internal message model.

Every channel speaks a different dialect. Everything below this module
works with `InboundMessage` only: the conversation logic never learns
whether a message arrived from WhatsApp, Instagram or Messenger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class Channel(str, Enum):
    WHATSAPP = "whatsapp"
    INSTAGRAM = "instagram"
    MESSENGER = "messenger"


@dataclass(frozen=True)
class InboundMessage:
    """A message after normalization, regardless of where it came from."""

    message_id: str
    """Provider-side id. Used for idempotency: providers retry."""

    conversation_id: str
    """Stable per (channel, user). Two channels never share a conversation."""

    channel: Channel
    sender_id: str
    text: str
    received_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @staticmethod
    def build_conversation_id(channel: Channel, sender_id: str) -> str:
        return f"{channel.value}:{sender_id}"


@dataclass(frozen=True)
class OutboundMessage:
    conversation_id: str
    channel: Channel
    recipient_id: str
    text: str
