"""The three channels must collapse into one internal shape."""

from app.channels.meta import InstagramAdapter, MessengerAdapter, WhatsAppAdapter
from app.core.models import Channel


def test_whatsapp_payload_is_normalized():
    payload = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {
                                    "id": "wamid.1",
                                    "from": "573001112233",
                                    "type": "text",
                                    "text": {"body": "hola"},
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }

    [message] = WhatsAppAdapter().parse(payload)

    assert message.message_id == "wamid.1"
    assert message.channel is Channel.WHATSAPP
    assert message.text == "hola"
    assert message.conversation_id == "whatsapp:573001112233"


def test_messenger_payload_is_normalized():
    payload = {
        "entry": [
            {
                "messaging": [
                    {
                        "sender": {"id": "PSID-9"},
                        "message": {"mid": "m.1", "text": "hola"},
                    }
                ]
            }
        ]
    }

    [message] = MessengerAdapter().parse(payload)

    assert message.channel is Channel.MESSENGER
    assert message.text == "hola"
    assert message.conversation_id == "messenger:PSID-9"


def test_instagram_uses_the_same_shape_as_messenger():
    payload = {
        "entry": [
            {
                "messaging": [
                    {
                        "sender": {"id": "IGSID-4"},
                        "message": {"mid": "m.2", "text": "hola"},
                    }
                ]
            }
        ]
    }

    [message] = InstagramAdapter().parse(payload)

    assert message.channel is Channel.INSTAGRAM
    assert message.conversation_id == "instagram:IGSID-4"


def test_echo_of_our_own_reply_is_ignored():
    """Without this the bot answers itself, forever."""
    payload = {
        "entry": [
            {
                "messaging": [
                    {
                        "sender": {"id": "PAGE"},
                        "message": {
                            "mid": "m.3",
                            "text": "our reply",
                            "is_echo": True,
                        },
                    }
                ]
            }
        ]
    }

    assert MessengerAdapter().parse(payload) == []


def test_status_updates_produce_no_messages():
    """Delivery and read receipts share the webhook. They are not input."""
    payload = {
        "entry": [
            {"changes": [{"value": {"statuses": [{"id": "wamid.1"}]}}]}
        ]
    }

    assert WhatsAppAdapter().parse(payload) == []


def test_same_user_on_two_channels_is_two_conversations():
    assert (
        WhatsAppAdapter().parse(
            {
                "entry": [
                    {
                        "changes": [
                            {
                                "value": {
                                    "messages": [
                                        {
                                            "id": "a",
                                            "from": "X",
                                            "text": {"body": "hi"},
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                ]
            }
        )[0].conversation_id
        != MessengerAdapter().parse(
            {
                "entry": [
                    {
                        "messaging": [
                            {
                                "sender": {"id": "X"},
                                "message": {"mid": "b", "text": "hi"},
                            }
                        ]
                    }
                ]
            }
        )[0].conversation_id
    )
