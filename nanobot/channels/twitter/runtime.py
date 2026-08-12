"""Twitter / X channel — OAuth 1.0a user context + bearer token for read.

Reads mentions of the configured bot via GET /2/tweets/search/recent and
posts replies via POST /2/tweets (OAuth 1.0a user context, since /2/tweets
write endpoints require user auth). Polls every ``poll_interval_seconds``
(default 900s = 15min) and dedupes by tweet id.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import time
import urllib.parse
from typing import Any

import httpx
from pydantic import Field

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import Base
from nanobot.utils.helpers import split_message

TWITTER_BASE = "https://api.twitter.com"
TWITTER_MAX_TWEET_LEN = 280
TWITTER_DEFAULT_INTERVAL = 900  # 15 min
TWITTER_MIN_INTERVAL = 60
TWITTER_RATE_LIMIT_BACKOFF_MAX = 900
TWITTER_TWEET_FIELDS = ",".join(
    [
        "author_id",
        "conversation_id",
        "in_reply_to_user_id",
        "created_at",
        "text",
    ]
)
TWITTER_USER_FIELDS = ",".join(["username", "name"])
TWITTER_EXPANSIONS = "author_id"


class TwitterConfig(Base):
    """Twitter / X channel configuration."""

    enabled: bool = False
    consent_granted: bool = False

    bearer_token: str = ""
    api_key: str = ""
    api_key_secret: str = ""
    access_token: str = ""
    access_token_secret: str = ""

    bot_username: str = ""
    poll_interval_seconds: int = TWITTER_DEFAULT_INTERVAL
    search_query: str = ""
    language: str = "en"
    allow_from: list[str] = Field(default_factory=list)
    group_policy: str = "open"
    max_mentions_per_poll: int = 20
    reply_prefix: str = ""

    send_progress: bool = True
    send_tool_hints: bool = True


def _percent_encode(value: str) -> str:
    """RFC 3986 percent-encoding for OAuth 1.0a."""
    return urllib.parse.quote(value, safe="~")


def _oauth1_sign(
    *,
    method: str,
    url: str,
    params: dict[str, str],
    api_key: str,
    api_key_secret: str,
    access_token: str,
    access_token_secret: str,
) -> dict[str, str]:
    """Build OAuth 1.0a Authorization header (user context)."""
    oauth_params = {
        "oauth_consumer_key": api_key,
        "oauth_nonce": base64.urlsafe_b64encode(
            hashlib.sha1(str(time.time_ns()).encode()).digest()
        )[:32].decode(),
        "oauth_signature_method": "HMAC-SHA1",
        "oauth_timestamp": str(int(time.time())),
        "oauth_token": access_token,
        "oauth_version": "1.0",
    }
    all_params = {**oauth_params, **params}
    param_str = "&".join(
        f"{_percent_encode(k)}={_percent_encode(v)}" for k, v in sorted(all_params.items())
    )
    base_str = "&".join(
        [
            method.upper(),
            _percent_encode(url.split("?", 1)[0]),
            _percent_encode(param_str),
        ]
    )
    signing_key = f"{_percent_encode(api_key_secret)}&{_percent_encode(access_token_secret)}"
    sig = base64.b64encode(
        hmac.new(signing_key.encode(), base_str.encode(), hashlib.sha1).digest()
    ).decode()
    oauth_params["oauth_signature"] = sig
    auth_header = "OAuth " + ", ".join(
        f'{_percent_encode(k)}="{_percent_encode(v)}"' for k, v in oauth_params.items()
    )
    return {"Authorization": auth_header}


class TwitterChannel(BaseChannel):
    """Twitter / X channel."""

    name = "twitter"
    display_name = "Twitter / X"
    send_progress = False  # no streaming on Twitter — single tweet is the unit

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return TwitterConfig().model_dump(by_alias=True)

    def __init__(self, config: Any, bus: MessageBus):
        if isinstance(config, dict):
            config = TwitterConfig.model_validate(config)
        super().__init__(config, bus)
        self.config: TwitterConfig = config
        self._self_user_id: str | None = None
        self._self_username: str | None = None
        self._seen_tweet_ids: set[str] = set()
        self._MAX_SEEN = 50_000
        self._http: httpx.AsyncClient | None = None
        self._poll_task: asyncio.Task | None = None

    # --- lifecycle ---

    async def start(self) -> None:
        if not self.config.consent_granted:
            self.logger.warning(
                "Twitter channel disabled: consent_granted is false. "
                "Set channels.twitter.consentGranted=true after explicit user permission."
            )
            return
        if not self._validate_config():
            return

        self._http = httpx.AsyncClient(
            base_url=TWITTER_BASE,
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

        me = await self._fetch_self()
        if not me:
            await self._close_http()
            return
        self._self_user_id, self._self_username = me
        self.logger.info(
            "Twitter bot connected as @{} ({})", self._self_username, self._self_user_id
        )

        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        try:
            await self._poll_task
        finally:
            self._poll_task = None

    async def stop(self) -> None:
        self._running = False
        task = self._poll_task
        if task and task is not asyncio.current_task():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await self._close_http()

    async def _close_http(self) -> None:
        if self._http:
            await self._http.aclose()
            self._http = None

    # --- polling ---

    async def _poll_loop(self) -> None:
        interval = max(TWITTER_MIN_INTERVAL, int(self.config.poll_interval_seconds))
        while self._running:
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                break
            except Exception:
                self.logger.exception("Twitter poll failed")
            if not self._running:
                break
            await asyncio.sleep(interval)

    async def _poll_once(self) -> None:
        if not self._http or not self._self_user_id:
            return
        params = self._build_search_params()
        params["tweet.fields"] = TWITTER_TWEET_FIELDS
        params["expansions"] = TWITTER_EXPANSIONS
        params["user.fields"] = TWITTER_USER_FIELDS
        url = f"/2/tweets/search/recent?{urllib.parse.urlencode(params)}"
        try:
            resp = await self._http.get(
                url,
                headers={"Authorization": f"Bearer {self.config.bearer_token}"},
            )
        except Exception:
            self.logger.warning("Twitter search request failed")
            return
        if resp.status_code == 429:
            reset = resp.headers.get("x-rate-limit-reset")
            wait = max(60, int(reset) - int(time.time())) if reset else 60
            self.logger.warning("Twitter rate limited, sleeping {}s", wait)
            await asyncio.sleep(min(wait, TWITTER_RATE_LIMIT_BACKOFF_MAX))
            return
        if resp.status_code >= 400:
            self.logger.warning("Twitter search returned {}: {}", resp.status_code, resp.text[:200])
            return
        payload = resp.json()
        tweets = payload.get("data") or []
        users_by_id = {u["id"]: u for u in (payload.get("includes") or {}).get("users", [])}
        new_count = 0
        for tweet in tweets:
            tid = tweet.get("id")
            if not tid or tid in self._seen_tweet_ids:
                continue
            self._remember_tweet(tid)
            author = users_by_id.get(tweet.get("author_id", ""), {})
            sender_id = author.get("username", "") or tweet.get("author_id", "")
            if not self._is_allowed_sender(sender_id):
                continue
            text = tweet.get("text", "")
            in_reply_to = tweet.get("in_reply_to_user_id")
            # Skip self-replies to own tweets (handled via outbound, not inbound).
            if in_reply_to == self._self_user_id and tweet.get("author_id") == self._self_user_id:
                continue
            meta = {
                "twitter": {
                    "tweet_id": tid,
                    "conversation_id": tweet.get("conversation_id"),
                    "author_id": tweet.get("author_id"),
                    "author_username": sender_id,
                    "author_name": author.get("name", ""),
                    "created_at": tweet.get("created_at"),
                },
                "message_id": tid,
            }
            await self._handle_message(
                sender_id=sender_id,
                chat_id=tid,
                content=text,
                metadata=meta,
                is_dm=False,
            )
            new_count += 1
        if new_count:
            self.logger.info("Twitter poll: delivered {} new mentions", new_count)

    def _build_search_params(self) -> dict[str, str]:
        bot = self.config.bot_username.lstrip("@")
        if self.config.search_query.strip():
            query = self.config.search_query.strip()
        else:
            query = f"@{bot} -from:{bot}"
        params: dict[str, str] = {
            "query": query,
            "max_results": str(max(10, min(100, int(self.config.max_mentions_per_poll)))),
            "sort_order": "recency",
        }
        # NOTE: X /2/tweets/search/recent rejects 'lang' as a top-level query
        # param — language filtering goes inside the query string (e.g. "lang:en").
        return params

    def _is_allowed_sender(self, username: str) -> bool:
        if self.config.group_policy == "open":
            return True
        allow = self.config.allow_from or []
        if not allow:
            return True
        if "*" in allow:
            return True
        return username in {a.lstrip("@") for a in allow}

    def _remember_tweet(self, tid: str) -> None:
        self._seen_tweet_ids.add(tid)
        if len(self._seen_tweet_ids) > self._MAX_SEEN:
            # Evict oldest half — set is unordered, so drop a slice.
            keep = list(self._seen_tweet_ids)[len(self._seen_tweet_ids) // 2 :]
            self._seen_tweet_ids = set(keep)

    # --- self / send ---

    async def _fetch_self(self) -> tuple[str, str] | None:
        if not self._http:
            return None
        url = f"{TWITTER_BASE}/2/users/me"
        headers = _oauth1_sign(
            method="GET",
            url=url,
            params={"user.fields": TWITTER_USER_FIELDS},
            api_key=self.config.api_key,
            api_key_secret=self.config.api_key_secret,
            access_token=self.config.access_token,
            access_token_secret=self.config.access_token_secret,
        )
        try:
            resp = await self._http.get(
                "/2/users/me", headers=headers, params={"user.fields": TWITTER_USER_FIELDS}
            )
        except Exception as exc:
            self.logger.error("Twitter auth probe failed: {}", exc)
            return None
        if resp.status_code >= 400:
            self.logger.error("Twitter auth probe failed: {} {}", resp.status_code, resp.text[:200])
            return None
        data = resp.json().get("data") or {}
        return data.get("id"), data.get("username")

    def _validate_config(self) -> bool:
        cfg = self.config
        missing = []
        if not cfg.bearer_token:
            missing.append("bearerToken")
        if not cfg.api_key:
            missing.append("apiKey")
        if not cfg.api_key_secret:
            missing.append("apiKeySecret")
        if not cfg.access_token:
            missing.append("accessToken")
        if not cfg.access_token_secret:
            missing.append("accessTokenSecret")
        if not cfg.bot_username:
            missing.append("botUsername")
        if missing:
            self.logger.error("Twitter channel not configured, missing: {}", ", ".join(missing))
            return False
        return True

    async def send(self, msg: OutboundMessage) -> None:
        if not self.config.consent_granted:
            self.logger.warning("Skip Twitter send: consent_granted is false")
            return
        if not self._http:
            self.logger.warning("Twitter client not initialized")
            return
        if not msg.content:
            return

        reply_to = msg.chat_id
        meta = msg.metadata or {}
        tw_meta = meta.get("twitter") if isinstance(meta.get("twitter"), dict) else {}
        explicit_reply = tw_meta.get("reply_to") or meta.get("reply_to") or reply_to
        text = self._apply_prefix(msg.content)
        for chunk in split_message(text, TWITTER_MAX_TWEET_LEN):
            await self._post_tweet(chunk, reply_to_tweet_id=explicit_reply)
            # Subsequent chunks go as standalone tweets (no thread), matching
            # how other channels handle multi-chunk sends.
            explicit_reply = None

    def _apply_prefix(self, text: str) -> str:
        prefix = self.config.reply_prefix or ""
        if not prefix:
            return text
        if text.startswith(prefix):
            return text
        budget = TWITTER_MAX_TWEET_LEN - len(prefix)
        return f"{prefix}{text[:budget]}"

    async def _post_tweet(self, text: str, *, reply_to_tweet_id: str | None) -> None:
        url = f"{TWITTER_BASE}/2/tweets"
        body: dict[str, Any] = {"text": text}
        if reply_to_tweet_id:
            body["reply"] = {"in_reply_to_tweet_id": reply_to_tweet_id}
        body_str = json.dumps(body)
        headers = _oauth1_sign(
            method="POST",
            url=url,
            params={},
            api_key=self.config.api_key,
            api_key_secret=self.config.api_key_secret,
            access_token=self.config.access_token,
            access_token_secret=self.config.access_token_secret,
        )
        headers["Content-Type"] = "application/json"
        try:
            resp = await self._http.post(url, headers=headers, content=body_str)
        except Exception:
            self.logger.exception("Twitter POST /2/tweets transport error")
            raise
        if resp.status_code >= 400:
            self.logger.error(
                "Twitter POST /2/tweets failed: {} {}", resp.status_code, resp.text[:300]
            )
            raise RuntimeError(f"Twitter post failed: {resp.status_code}")
        self.logger.info("Twitter tweet posted ({} chars)", len(text))
