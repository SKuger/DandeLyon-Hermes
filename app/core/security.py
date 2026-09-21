"""Webhook signature verification.

The webhook endpoint is public: anyone who finds the URL can POST to it.
Meta signs every request with the app secret, so the signature is the
only thing separating a real message from a forged one.
"""

from __future__ import annotations

import hashlib
import hmac


def is_valid_signature(body: bytes, header: str | None, app_secret: str) -> bool:
    """Check the `X-Hub-Signature-256` header against the raw body.

    The raw body matters: re-serializing the parsed JSON changes bytes
    (key order, spacing) and the signature stops matching.
    """
    if not header or not header.startswith("sha256="):
        return False

    expected = hmac.new(
        app_secret.encode("utf-8"), body, hashlib.sha256
    ).hexdigest()

    # compare_digest, not ==, so the comparison time does not leak
    # how many leading characters were correct.
    return hmac.compare_digest(expected, header.removeprefix("sha256="))
