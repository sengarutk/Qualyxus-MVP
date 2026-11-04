"""AES-256-GCM sealing for federated model payloads.

Payloads are encrypted with a pre-shared 256-bit key that never travels with the
payload. GCM authenticates the ciphertext *and* the metadata bound as associated data
(node id, version, plaintext digest), so tampering with either is detected on decrypt.

Key resolution: ``QUALYXUS_FEDERATED_KEY`` (url-safe base64 of 32 bytes) or the file at
``federated_learning.key_path`` in the config. Generate one with::

    python -m src.training.federated_crypto --generate > secrets/federated.key
"""
from __future__ import annotations

import base64
import os
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

KEY_ENV = "QUALYXUS_FEDERATED_KEY"


class FederatedKeyError(RuntimeError):
    pass


def _decode_key(text: str) -> bytes:
    try:
        key = base64.urlsafe_b64decode(text.strip().encode("ascii"))
    except Exception as exc:
        raise FederatedKeyError(f"Federated key is not valid base64: {exc}") from exc
    if len(key) != 32:
        raise FederatedKeyError(f"Federated key must be 32 bytes (got {len(key)})")
    return key


def load_key(key_path: str | None = None) -> bytes:
    env = os.getenv(KEY_ENV)
    if env:
        return _decode_key(env)
    if key_path and Path(key_path).is_file():
        return _decode_key(Path(key_path).read_text(encoding="utf-8"))
    raise FederatedKeyError(
        f"No federated key configured: set {KEY_ENV} or federated_learning.key_path"
    )


def generate_key() -> str:
    return base64.urlsafe_b64encode(AESGCM.generate_key(bit_length=256)).decode("ascii")


def seal(plaintext: bytes, key: bytes, associated_data: bytes) -> tuple[str, str]:
    nonce = os.urandom(12)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, associated_data)
    return base64.b64encode(nonce).decode("ascii"), base64.b64encode(ciphertext).decode("ascii")


def open_sealed(nonce_b64: str, ciphertext_b64: str, key: bytes, associated_data: bytes) -> bytes:
    return AESGCM(key).decrypt(base64.b64decode(nonce_b64), base64.b64decode(ciphertext_b64), associated_data)


if __name__ == "__main__":
    import sys

    if "--generate" in sys.argv:
        print(generate_key())
    else:
        print(__doc__)
