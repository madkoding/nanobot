"""AES-128-ECB encryption / decryption helpers for WeChat media (matches pic-decrypt.ts / aes-ecb.ts)."""

from __future__ import annotations

import base64
import re
from contextlib import suppress

from loguru import logger


def _parse_aes_key(aes_key_b64: str) -> bytes:
    """Parse a base64-encoded AES key, handling both encodings seen in the wild.

    From ``pic-decrypt.ts parseAesKey``:

    * ``base64(raw 16 bytes)``            → images (media.aes_key)
    * ``base64(hex string of 16 bytes)``  → file / voice / video

    In the second case base64-decoding yields 32 ASCII hex chars which must
    then be parsed as hex to recover the actual 16-byte key.
    """
    decoded = base64.b64decode(aes_key_b64)
    if len(decoded) == 16:
        return decoded
    if len(decoded) == 32 and re.fullmatch(rb"[0-9a-fA-F]{32}", decoded):
        # hex-encoded key: base64 → hex string → raw bytes
        return bytes.fromhex(decoded.decode("ascii"))
    raise ValueError(
        f"aes_key must decode to 16 raw bytes or 32-char hex string, got {len(decoded)} bytes"
    )


def _encrypt_aes_ecb(data: bytes, aes_key_b64: str) -> bytes:
    """Encrypt data with AES-128-ECB and PKCS7 padding for CDN upload."""
    try:
        key = _parse_aes_key(aes_key_b64)
    except Exception as e:
        logger.warning("Failed to parse AES key for encryption, sending raw: {}", e)
        return data

    # PKCS7 padding
    pad_len = 16 - len(data) % 16
    padded = data + bytes([pad_len] * pad_len)

    with suppress(ImportError):
        from Crypto.Cipher import AES

        cipher = AES.new(key, AES.MODE_ECB)
        return cipher.encrypt(padded)

    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

        cipher_obj = Cipher(algorithms.AES(key), modes.ECB())
        encryptor = cipher_obj.encryptor()
        return encryptor.update(padded) + encryptor.finalize()
    except ImportError:
        logger.warning("Cannot encrypt media: install 'pycryptodome' or 'cryptography'")
        return data


def _decrypt_aes_ecb(data: bytes, aes_key_b64: str) -> bytes:
    """Decrypt AES-128-ECB media data.

    ``aes_key_b64`` is always base64-encoded (caller converts hex keys first).
    """
    try:
        key = _parse_aes_key(aes_key_b64)
    except Exception as e:
        logger.warning("Failed to parse AES key, returning raw data: {}", e)
        return data

    decrypted: bytes | None = None

    with suppress(ImportError):
        from Crypto.Cipher import AES

        cipher = AES.new(key, AES.MODE_ECB)
        decrypted = cipher.decrypt(data)

    if decrypted is None:
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

            cipher_obj = Cipher(algorithms.AES(key), modes.ECB())
            decryptor = cipher_obj.decryptor()
            decrypted = decryptor.update(data) + decryptor.finalize()
        except ImportError:
            logger.warning("Cannot decrypt media: install 'pycryptodome' or 'cryptography'")
            return data

    return _pkcs7_unpad_safe(decrypted)


def _pkcs7_unpad_safe(data: bytes, block_size: int = 16) -> bytes:
    """Safely remove PKCS7 padding when valid; otherwise return original bytes."""
    if not data:
        return data
    if len(data) % block_size != 0:
        return data
    pad_len = data[-1]
    if pad_len < 1 or pad_len > block_size:
        return data
    if data[-pad_len:] != bytes([pad_len]) * pad_len:
        return data
    return data[:-pad_len]
