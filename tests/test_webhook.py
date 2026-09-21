"""The HTTP contract with Meta."""

import hashlib
import hmac
import json

from fastapi.testclient import TestClient

from app import main
from app.core.security import is_valid_signature

client = TestClient(main.app)

PAYLOAD = {
    "entry": [
        {
            "changes": [
                {
                    "value": {
                        "messages": [
                            {
                                "id": "wamid.test",
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


def test_health():
    assert client.get("/health").status_code == 200


def test_verification_echoes_the_challenge():
    response = client.get(
        "/webhook/whatsapp",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": main.settings.verify_token,
            "hub.challenge": "12345",
        },
    )
    assert response.status_code == 200
    assert response.text == "12345"


def test_verification_rejects_a_wrong_token():
    response = client.get(
        "/webhook/whatsapp",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "wrong",
            "hub.challenge": "12345",
        },
    )
    assert response.status_code == 403


def test_webhook_acknowledges_immediately():
    assert client.post("/webhook/whatsapp", json=PAYLOAD).status_code == 200


def test_unknown_channel_is_rejected():
    assert client.post("/webhook/telegram", json=PAYLOAD).status_code == 422


def test_signature_accepts_a_correctly_signed_body():
    secret, body = "s3cret", json.dumps(PAYLOAD).encode()
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

    assert is_valid_signature(body, f"sha256={digest}", secret)


def test_signature_rejects_a_tampered_body():
    secret, body = "s3cret", json.dumps(PAYLOAD).encode()
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

    assert not is_valid_signature(b'{"entry": []}', f"sha256={digest}", secret)


def test_signature_rejects_a_missing_header():
    assert not is_valid_signature(b"{}", None, "s3cret")
