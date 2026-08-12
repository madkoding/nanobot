"""Twitter / X setup validation."""

from __future__ import annotations

from typing import Any

import httpx

from nanobot.channels.contracts import ChannelValidationContext
from nanobot.channels.validation import (
    check,
    int_value,
    required_checks,
    status_from_checks,
    string_value,
)


def _probe_bearer(token: str) -> tuple[str, str]:
    """Return (status, detail) for a bearer token probe against /2/users/me."""
    try:
        resp = httpx.get(
            "https://api.twitter.com/2/users/me",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10.0,
        )
    except Exception as exc:
        return "warn", f"Could not reach X API now: {exc}"
    if resp.status_code == 200:
        return "pass", "Bearer token authenticated."
    if resp.status_code == 401:
        return "fail", "Bearer token was rejected (401)."
    return "warn", f"X API returned {resp.status_code}: {resp.text[:160]}"


def validate(
    values: dict[str, Any],
    context: ChannelValidationContext,
) -> dict[str, Any]:
    checks, missing = required_checks("twitter", values)
    bot = string_value(values.get("botUsername")).lstrip("@")
    if bot:
        checks.append(
            check("bot_username", "Bot username", "pass", f"Will poll mentions of @{bot}.")
        )
    interval = int_value(values.get("pollIntervalSeconds")) or 900
    if interval < 60:
        checks.append(
            check("poll_interval", "Poll interval", "fail", "Minimum poll interval is 60 seconds.")
        )
    else:
        checks.append(
            check(
                "poll_interval",
                "Poll interval",
                "pass",
                f"Polling every {interval} seconds (~{interval // 60} min).",
            )
        )

    bearer = string_value(values.get("bearerToken"))
    api_key = string_value(values.get("apiKey"))
    api_secret = string_value(values.get("apiKeySecret"))
    access_token = string_value(values.get("accessToken"))
    access_secret = string_value(values.get("accessTokenSecret"))
    if all([bearer, api_key, api_secret, access_token, access_secret]):
        status, detail = _probe_bearer(bearer)
        checks.append(check("bearer_token", "Bearer token", status, detail))
    else:
        checks.append(
            check(
                "oauth_credentials",
                "OAuth 1.0a credentials",
                "warn",
                "Read + write both require all four OAuth fields.",
            )
        )

    identity = {"account": f"@{bot}" if bot else ""}
    return status_from_checks("twitter", checks, missing, identity=identity)


__all__ = ["validate"]
