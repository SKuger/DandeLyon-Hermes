"""Sending replies back to the provider."""

from __future__ import annotations

import logging

import httpx

from app.channels.meta import ADAPTERS
from app.core.models import OutboundMessage

logger = logging.getLogger(__name__)


class LoggingSender:
    """Default sender: prints the reply instead of calling Meta.

    Lets the whole flow be exercised end to end without credentials.
    """

    def __init__(self) -> None:
        self.sent: list[OutboundMessage] = []

    async def send(self, message: OutboundMessage) -> None:
        self.sent.append(message)
        logger.info(
            "[%s -> %s] %s",
            message.channel.value,
            message.recipient_id,
            message.text,
        )


class GraphApiSender:
    """Real sender against the Meta Graph API."""

    def __init__(self, token: str, phone_number_id: str, page_id: str) -> None:
        self._token = token
        self._phone_number_id = phone_number_id
        self._page_id = page_id

    def _url(self, message: OutboundMessage) -> str:
        base = "https://graph.facebook.com/v21.0"
        if message.channel.value == "whatsapp":
            return f"{base}/{self._phone_number_id}/messages"
        return f"{base}/{self._page_id}/messages"

    async def send(self, message: OutboundMessage) -> None:
        body = ADAPTERS[message.channel].render(message)
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                self._url(message),
                json=body,
                headers={"Authorization": f"Bearer {self._token}"},
            )
        if response.is_error:
            # Worth logging loudly: from the user's side this looks like
            # the bot simply never answered.
            logger.error(
                "send failed (%s): %s", response.status_code, response.text
            )
