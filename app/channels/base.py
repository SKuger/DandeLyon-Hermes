"""Channel adapters.

An adapter has one job: turn a provider payload into `InboundMessage`
objects, and turn an `OutboundMessage` back into whatever the provider
expects. Adding a channel means adding one file here — nothing else in
the codebase changes.
"""

from __future__ import annotations

from typing import Any, Protocol

from app.core.models import InboundMessage, OutboundMessage


class ChannelAdapter(Protocol):
    """What every channel must implement."""

    def parse(self, payload: dict[str, Any]) -> list[InboundMessage]:
        """Extract messages from a provider payload.

        A single webhook call can carry several messages, or none at all
        (delivery receipts, read receipts, status updates). Returning an
        empty list is normal and must not be treated as an error.
        """
        ...

    def render(self, message: OutboundMessage) -> dict[str, Any]:
        """Build the provider-shaped body used to send a reply."""
        ...
