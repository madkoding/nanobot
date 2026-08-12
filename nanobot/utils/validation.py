"""Input validation utilities for nanobot SDK."""

from __future__ import annotations

import re
from typing import Final

# Maximum lengths for input validation
MAX_SESSION_KEY_LENGTH: Final[int] = 64
MAX_CHAT_ID_LENGTH: Final[int] = 128
MAX_SENDER_ID_LENGTH: Final[int] = 128
MAX_MESSAGE_LENGTH: Final[int] = 50000
MAX_CHANNEL_LENGTH: Final[int] = 64
MAX_MEDIA_PATH_LENGTH: Final[int] = 512

# Patterns for safe identifiers
SESSION_KEY_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-zA-Z0-9_:-]+$")
IDENTIFIER_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-zA-Z0-9_-]+$")


class ValidationError(ValueError):
    """Raised when input validation fails."""

    pass


def validate_session_key(session_key: str) -> str:
    """Validate and sanitize session key.

    Args:
        session_key: The session key to validate.

    Returns:
        The validated session key.

    Raises:
        ValidationError: If the session key is invalid.
    """
    if not session_key:
        raise ValidationError("Session key cannot be empty")

    if len(session_key) > MAX_SESSION_KEY_LENGTH:
        raise ValidationError(
            f"Session key too long: {len(session_key)} > {MAX_SESSION_KEY_LENGTH}"
        )

    if not SESSION_KEY_PATTERN.match(session_key):
        raise ValidationError(
            "Session key contains invalid characters. "
            "Only alphanumeric, underscore, colon, and hyphen allowed."
        )

    return session_key


def validate_identifier(value: str, name: str, max_length: int = 128) -> str:
    """Validate a generic identifier (chat_id, sender_id, channel).

    Args:
        value: The identifier value to validate.
        name: Name of the field for error messages.
        max_length: Maximum allowed length.

    Returns:
        The validated identifier.

    Raises:
        ValidationError: If the identifier is invalid.
    """
    if not value:
        raise ValidationError(f"{name} cannot be empty")

    if len(value) > max_length:
        raise ValidationError(f"{name} too long: {len(value)} > {max_length}")

    if not IDENTIFIER_PATTERN.match(value):
        raise ValidationError(
            f"{name} contains invalid characters. "
            "Only alphanumeric, underscore, and hyphen allowed."
        )

    return value


def validate_chat_id(chat_id: str) -> str:
    """Validate chat ID.

    Args:
        chat_id: The chat ID to validate.

    Returns:
        The validated chat ID.

    Raises:
        ValidationError: If the chat ID is invalid.
    """
    return validate_identifier(chat_id, "Chat ID", MAX_CHAT_ID_LENGTH)


def validate_sender_id(sender_id: str) -> str:
    """Validate sender ID.

    Args:
        sender_id: The sender ID to validate.

    Returns:
        The validated sender ID.

    Raises:
        ValidationError: If the sender ID is invalid.
    """
    return validate_identifier(sender_id, "Sender ID", MAX_SENDER_ID_LENGTH)


def validate_channel(channel: str) -> str:
    """Validate channel name.

    Args:
        channel: The channel name to validate.

    Returns:
        The validated channel name.

    Raises:
        ValidationError: If the channel name is invalid.
    """
    return validate_identifier(channel, "Channel", MAX_CHANNEL_LENGTH)


def sanitize_input(message: str, max_length: int = MAX_MESSAGE_LENGTH) -> str:
    """Sanitize user input message.

    Args:
        message: The message to sanitize.
        max_length: Maximum allowed length.

    Returns:
        The sanitized message.

    Raises:
        ValidationError: If the message is too long.
    """
    if not isinstance(message, str):
        raise ValidationError("Message must be a string")

    if len(message) > max_length:
        raise ValidationError(f"Message too long: {len(message)} > {max_length}")

    # Strip leading/trailing whitespace but preserve internal formatting
    return message.strip()


def validate_media_path(path: str) -> str:
    """Validate media file path to prevent path traversal.

    Args:
        path: The media path to validate.

    Returns:
        The validated path (unchanged if valid).

    Raises:
        ValidationError: If the path is invalid or dangerous.
    """
    from pathlib import Path

    if not path:
        raise ValidationError("Media path cannot be empty")

    if len(path) > MAX_MEDIA_PATH_LENGTH:
        raise ValidationError(f"Media path too long: {len(path)} > {MAX_MEDIA_PATH_LENGTH}")

    # Check for path traversal attempts in the original path string
    # Don't resolve/normalize absolute paths to preserve cross-platform compatibility
    path_obj = Path(path)

    # Only check for '..' components in the path as given
    # Absolute paths like /tmp/image.png are fine
    # Relative paths with .. like ../../etc/passwd are not
    if ".." in path_obj.parts:
        raise ValidationError("Media path cannot contain '..' components")

    # Return the original path unchanged to maintain test compatibility
    return path


def validate_media_paths(paths: list[str] | None) -> list[str] | None:
    """Validate a list of media paths.

    Args:
        paths: List of media paths to validate.

    Returns:
        List of validated paths, or None if input was None.

    Raises:
        ValidationError: If any path is invalid.
    """
    if paths is None:
        return None

    validated = []
    for path in paths:
        validated.append(validate_media_path(path))

    return validated
