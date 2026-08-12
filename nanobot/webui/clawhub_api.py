"""ClawHub registry integration for the WebUI.

Talks to the public ClawHub registry (https://clawhub.ai) so the Skills
settings section can browse and install skills without leaving the WebUI.

Search/trending hit the public JSON API; install downloads the skill zip
and extracts it into the workspace skills directory (same layout the
``clawhub`` CLI produces with ``--workdir``).
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

import httpx

CLAWHUB_API = "https://clawhub.ai"
_SEARCH_PATH = "/api/v1/search"
_TRENDING_PATH = "/api/v1/trending"
_DOWNLOAD_PATH = "/api/v1/download"
_TIMEOUT = 15.0
_MAX_RESULTS = 50
_SOURCE_FILE = ".clawhub-source.json"

_CATALOG_TTL_SECONDS = 600.0
_BROWSE_MAX_PAGE_SIZE = 50
_BROWSE_PAGE_CONCURRENCY = 12

# Full-catalog cache: ``{"items": [...], "fetched_at": monotonic}``. The
# trending endpoint paginates with opaque cursors; fetching every page takes
# a few seconds, so we cache the normalized, sorted catalog.
_catalog_cache: dict[str, Any] = {}
_catalog_lock = threading.Lock()
# Catalog fetch state: "idle" | "loading" | "ready" | "error". The fetch
# runs in a background daemon thread so the WebUI never blocks on it.
_catalog_state: dict[str, Any] = {"status": "idle"}


class ClawhubError(Exception):
    """Raised when the ClawHub registry call fails."""


def _get(path: str, params: dict[str, Any]) -> dict[str, Any]:
    try:
        response = httpx.get(
            f"{CLAWHUB_API}{path}",
            params=params,
            timeout=_TIMEOUT,
            follow_redirects=True,
        )
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPStatusError as exc:
        raise ClawhubError(f"ClawHub responded {exc.response.status_code}") from exc
    except httpx.HTTPError as exc:
        raise ClawhubError(f"Could not reach ClawHub: {exc}") from exc
    except ValueError as exc:
        raise ClawhubError("ClawHub returned an invalid response") from exc
    if not isinstance(data, dict):
        raise ClawhubError("ClawHub returned an invalid response")
    return data


def _split_reference(reference: str) -> tuple[str, str]:
    """Split an install reference like ``owner/slug`` into (owner, slug).

    Handles skills.sh references (``skills-sh:owner/repo/slug`` or
    ``owner/repo@slug``) by stripping the prefix, so the owner shown in
    the UI is the GitHub user/org (``vercel-labs``), not
    ``skills-sh:vercel-labs``.
    """
    ref = reference.strip()
    if ref.startswith("skills-sh:"):
        ref = ref[len("skills-sh:") :].strip()
    parts = ref.split("/")
    if len(parts) >= 2:
        return parts[0], parts[-1]
    return "", reference


def _summary_payload(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize a ClawHub search/trending item into a safe summary."""
    install = item.get("install") or {}
    reference = install.get("reference") or item.get("slug") or ""
    owner, slug = _split_reference(reference)
    metrics = item.get("metrics") or {}
    rolling = metrics.get("rolling60DayInstalls")
    lifetime = metrics.get("lifetimeInstalls")
    # Trending items do not carry rolling60DayInstalls; fall back to the
    # best available install metric so sorting stays meaningful.
    installs = int(rolling) if rolling is not None else int(lifetime or 0)
    return {
        "slug": slug,
        "owner": owner,
        "reference": reference,
        "name": item.get("displayName") or slug,
        "description": (item.get("summary") or "").strip(),
        "installs_60d": installs,
        "lifetime_installs": int(lifetime or 0),
        "downloads": int(item.get("downloads") or 0),
        "kind": install.get("kind") or "clawhub",
    }


_INSTALLABLE_KINDS = {"clawhub", "skills-sh"}


def clawhub_search(query: str, limit: int = 20) -> list[dict[str, Any]]:
    """Search the ClawHub registry by natural language query.

    Only installable kinds are returned: ``clawhub`` (downloaded from the
    ClawHub download API) and ``skills-sh`` (downloaded from their GitHub
    source repository, see :func:`skills_sh_install`). Results are sorted
    most-installed first (60-day installs when available, lifetime
    otherwise) and truncated to ``limit``.
    """
    query = query.strip()
    if not query:
        return []
    data = _get(
        _SEARCH_PATH,
        {"q": query, "limit": max(1, min(limit, _MAX_RESULTS))},
    )
    results = data.get("results") or []
    payloads = [
        _summary_payload(item)
        for item in results
        if isinstance(item, dict) and (item.get("install") or {}).get("kind") in _INSTALLABLE_KINDS
    ]
    payloads.sort(key=lambda s: s["installs_60d"], reverse=True)
    return payloads[:limit]


def clawhub_trending(limit: int = 20) -> list[dict[str, Any]]:
    """Return trending skills from the ClawHub registry, most installed first.

    Only installable kinds are returned (see :func:`clawhub_search`). The
    trending endpoint only returns the current window (100 items), so the
    ranking is limited to that window.
    """
    data = _get(
        _TRENDING_PATH,
        {"limit": max(1, min(limit, _MAX_RESULTS))},
    )
    items = data.get("items") or []
    payloads = [
        _summary_payload(item)
        for item in items
        if isinstance(item, dict) and (item.get("install") or {}).get("kind") in _INSTALLABLE_KINDS
    ]
    payloads.sort(key=lambda s: s["lifetime_installs"], reverse=True)
    return payloads[:limit]


def clawhub_browse(page: int = 1, page_size: int = 50) -> dict[str, Any]:
    """Browse the whole ClawHub catalog ordered by lifetime installs.

    Never blocks on the network: the catalog is fetched in a background
    thread (see :func:`_catalog_items`). While the first fetch is running
    the payload carries ``loading: True`` so the UI can show a spinner and
    poll until the catalog is ready.
    """
    page = max(1, int(page))
    page_size = max(1, min(int(page_size), _BROWSE_MAX_PAGE_SIZE))
    items, status = _catalog_items()
    total = len(items)
    start = (page - 1) * page_size
    chunk = items[start : start + page_size]
    payload: dict[str, Any] = {
        "results": chunk,
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": max(1, math.ceil(total / page_size)) if total else 1,
        "loading": status == "loading",
    }
    if status == "error":
        payload["error"] = _catalog_state.get("error") or "ClawHub catalog is unavailable"
    return payload


def _catalog_items() -> tuple[list[dict[str, Any]], str]:
    """Return (items, status) without ever blocking on the network.

    status is "ready" when the cached catalog is fresh, "loading" while a
    background fetch is running, and "error" after a failed fetch. On the
    first call (or after a TTL expiry) a background fetch is started and
    stale data (or an empty list) is returned immediately.
    """
    now = time.monotonic()
    with _catalog_lock:
        cached = _catalog_cache.get("items")
        fetched_at = _catalog_cache.get("fetched_at", 0)
        status = _catalog_state.get("status", "idle")
        if cached is not None and now - fetched_at < _CATALOG_TTL_SECONDS:
            return cached, "ready"
        if status == "loading":
            return cached or [], "loading"
        if status == "error":
            return cached or [], "error"
        # idle: kick off a background fetch; serve stale data (or nothing
        # on the very first load) without blocking the caller.
        _catalog_state["status"] = "loading"
    threading.Thread(
        target=_background_catalog_fetch,
        name="clawhub-catalog-prefetch",
        daemon=True,
    ).start()
    return cached or [], "ready" if cached is not None else "loading"


def _background_catalog_fetch() -> None:
    """Fetch the full catalog in a background thread and cache it."""
    try:
        items = _fetch_catalog()
    except ClawhubError as exc:
        with _catalog_lock:
            _catalog_state["status"] = "error"
            _catalog_state["error"] = str(exc)
        return
    with _catalog_lock:
        _catalog_cache["items"] = items
        _catalog_cache["fetched_at"] = time.monotonic()
        _catalog_state["status"] = "ready"


def _fetch_catalog() -> list[dict[str, Any]]:
    """Fetch every trending page (concurrently) and normalize the results."""
    data = _get(_TRENDING_PATH, {"limit": _BROWSE_MAX_PAGE_SIZE})
    first_items = data.get("items") or []
    total = int(data.get("totalItems") or 0)
    page_size = len(first_items) or _BROWSE_MAX_PAGE_SIZE
    snapshot = _cursor_snapshot(data.get("nextCursor")) or data.get("snapshotId")
    offsets = list(range(page_size, total, page_size))

    pages = _fetch_pages_concurrent(snapshot, offsets)
    raw_items: list[dict[str, Any]] = list(first_items)
    for page_items in pages:
        if page_items is None:
            # A concurrent fetch failed; fall back to sequential walking.
            return _fetch_catalog_sequential(data)
        raw_items.extend(page_items)

    payloads = [
        _summary_payload(item)
        for item in raw_items
        if isinstance(item, dict) and (item.get("install") or {}).get("kind") in _INSTALLABLE_KINDS
    ]
    payloads.sort(key=lambda s: s["lifetime_installs"], reverse=True)
    return payloads


def _fetch_pages_concurrent(
    snapshot: Any,
    offsets: list[int],
) -> list[list[dict[str, Any]] | None]:
    """Fetch trending pages concurrently, one request per offset."""

    async def fetch_one(offset: int) -> list[dict[str, Any]] | None:
        cursor = _build_cursor(snapshot, offset)
        try:
            data = await asyncio.to_thread(
                _get,
                _TRENDING_PATH,
                {"limit": _BROWSE_MAX_PAGE_SIZE, "cursor": cursor},
            )
            return data.get("items") or []
        except ClawhubError:
            return None

    async def run() -> list[list[dict[str, Any]] | None]:
        semaphore = asyncio.Semaphore(_BROWSE_PAGE_CONCURRENCY)

        async def limited(offset: int) -> list[dict[str, Any]] | None:
            async with semaphore:
                return await fetch_one(offset)

        return await asyncio.gather(*(limited(o) for o in offsets))

    if not offsets:
        return []
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running event loop; drive the coroutine ourselves.
        return asyncio.run(run())
    # Already inside an event loop (e.g. the WebUI handler); fall back to
    # sequential fetches instead of nesting a second loop.
    return [fetch_one_sequential(snapshot, o) for o in offsets]


def fetch_one_sequential(snapshot: Any, offset: int) -> list[dict[str, Any]] | None:
    cursor = _build_cursor(snapshot, offset)
    try:
        data = _get(_TRENDING_PATH, {"limit": _BROWSE_MAX_PAGE_SIZE, "cursor": cursor})
        return data.get("items") or []
    except ClawhubError:
        return None


def _fetch_catalog_sequential(seed: dict[str, Any]) -> list[dict[str, Any]]:
    """Sequential fallback for fetching the full trending catalog."""
    items: list[dict[str, Any]] = list(seed.get("items") or [])
    cursor = seed.get("nextCursor")
    while cursor:
        data = _get(_TRENDING_PATH, {"limit": _BROWSE_MAX_PAGE_SIZE, "cursor": cursor})
        batch = data.get("items") or []
        if not batch:
            break
        items.extend(batch)
        cursor = data.get("nextCursor")
    payloads = [
        _summary_payload(item)
        for item in items
        if isinstance(item, dict) and (item.get("install") or {}).get("kind") in _INSTALLABLE_KINDS
    ]
    payloads.sort(key=lambda s: s["lifetime_installs"], reverse=True)
    return payloads


def _cursor_snapshot(cursor: str | None) -> Any:
    """Extract the snapshot id from a trending cursor, if decodable."""
    if not cursor:
        return None
    try:
        decoded = json.loads(base64.urlsafe_b64decode(cursor + "=="))
        if isinstance(decoded, dict):
            return decoded.get("s")
    except (ValueError, TypeError):
        return None
    return None


def _build_cursor(snapshot: Any, offset: int) -> str:
    """Build a trending pagination cursor for the given offset.

    Trending cursors encode ``{v, s (snapshot), o (offset), e, p}`` and the
    server validates them against the active snapshot, so we can jump to
    any offset within the snapshot.
    """
    payload = {"v": 1, "s": snapshot, "o": offset, "e": offset, "p": True}
    raw = json.dumps(payload).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _clawhub_download_install(reference: str, skills_dir: Path) -> dict[str, Any]:
    """Download and install a ClawHub skill into ``skills_dir``.

    ``reference`` is the install reference (``owner/slug``). The zip is
    extracted into ``skills_dir/<slug>/``, matching the layout the
    ``clawhub`` CLI produces.
    """
    reference = reference.strip()
    if not reference or "/" not in reference:
        raise ClawhubError("Invalid skill reference")
    owner, slug = _split_reference(reference)
    if not owner or not slug:
        raise ClawhubError("Invalid skill reference")

    try:
        response = httpx.get(
            f"{CLAWHUB_API}{_DOWNLOAD_PATH}",
            params={"slug": slug, "ownerHandle": owner},
            timeout=_TIMEOUT,
            follow_redirects=True,
        )
        response.raise_for_status()
        raw = response.content
    except httpx.HTTPStatusError as exc:
        raise ClawhubError(f"ClawHub responded {exc.response.status_code}") from exc
    except httpx.HTTPError as exc:
        raise ClawhubError(f"Could not reach ClawHub: {exc}") from exc

    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
    except (TypeError, zipfile.BadZipFile) as exc:
        raise ClawhubError("ClawHub returned an invalid skill archive") from exc

    target = skills_dir / slug
    target.mkdir(parents=True, exist_ok=True)
    for member in archive.namelist():
        # Guard against zip-slip: never write outside the target directory.
        resolved = (target / member).resolve()
        if not resolved.is_relative_to(target.resolve()):
            raise ClawhubError("Skill archive contains unsafe paths")
        if member.endswith("/"):
            resolved.mkdir(parents=True, exist_ok=True)
            continue
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_bytes(archive.read(member))

    _write_source_meta(skills_dir, slug, reference, "clawhub")
    return {"slug": slug, "installed": True, "path": str(target)}


def _write_source_meta(skills_dir: Path, slug: str, reference: str, kind: str) -> None:
    """Record where a skill came from so it can be updated later."""
    meta = {"reference": reference, "kind": kind}
    (skills_dir / slug / _SOURCE_FILE).write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )


def _read_source_meta(skills_dir: Path, slug: str) -> dict[str, str] | None:
    """Return the recorded install source for ``slug``, or None."""
    meta_path = skills_dir / slug / _SOURCE_FILE
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    reference = data.get("reference")
    kind = data.get("kind")
    if not isinstance(reference, str) or not reference:
        return None
    return {"reference": reference, "kind": str(kind) if isinstance(kind, str) else "clawhub"}


def skills_sh_install(reference: str, skills_dir: Path) -> dict[str, Any]:
    """Install a ``skills-sh`` skill from its GitHub source repository.

    The skills.sh API itself requires a Vercel OIDC token, but skills are
    plain folders in GitHub repos, so we install them straight from the
    source: ``skills-sh:owner/repo/slug`` (or ``owner/repo@slug``) maps to
    the GitHub repo ``owner/repo`` and the skill folder ``<slug>`` inside
    it (conventionally under ``skills/<slug>/``). Files are downloaded from
    the repo tree and extracted into ``skills_dir/<slug>/``.
    """
    ref = reference.strip()
    if ref.startswith("skills-sh:"):
        ref = ref[len("skills-sh:") :].strip()
    if "@" in ref:
        owner_repo, slug = ref.rsplit("@", 1)
    else:
        parts = ref.split("/")
        if len(parts) != 3:
            raise ClawhubError("Invalid skills-sh reference (expected owner/repo/slug)")
        owner_repo, slug = "/".join(parts[:2]), parts[2]
    owner, sep, repo = owner_repo.partition("/")
    if not sep or not owner or not repo or not slug:
        raise ClawhubError("Invalid skills-sh reference (expected owner/repo/slug)")

    branch = _github_default_branch(owner, repo)
    try:
        tree = _github_get_json(
            f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
        )
    except ClawhubError as exc:
        raise ClawhubError(f"Could not read the skill repository: {exc}") from exc
    entries = tree.get("tree") if isinstance(tree, dict) else None
    if not isinstance(entries, list):
        raise ClawhubError("Could not read the skill repository tree")

    skill_paths = _skill_folder_paths(entries, slug)
    if not skill_paths:
        raise ClawhubError(
            f"Skill '{slug}' not found in {owner}/{repo} (looked for a '{slug}/SKILL.md' folder)"
        )
    prefix = skill_paths[0]  # deterministic: prefers skills/<slug>, then */<slug>
    files = [entry["path"] for entry in entries if entry["path"].startswith(prefix + "/")]

    target = skills_dir / slug
    target.mkdir(parents=True, exist_ok=True)
    for path in files:
        if not path.endswith("/"):
            resolved = (target / path[len(prefix) + 1 :]).resolve()
            if not resolved.is_relative_to(target.resolve()):
                raise ClawhubError("Skill source contains unsafe paths")
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_bytes(
                _github_get_raw(f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}")
            )

    _write_source_meta(skills_dir, slug, ref, "skills-sh")
    return {"slug": slug, "installed": True, "path": str(target), "source": "skills-sh"}


def _safe_skill_dirs(skills_dir: Path) -> list[Path]:
    """Return the workspace skill folders (directories containing SKILL.md)."""
    if not skills_dir.is_dir():
        return []
    return [d for d in skills_dir.iterdir() if d.is_dir() and (d / "SKILL.md").exists()]


def clawhub_update_all(skills_dir: Path) -> dict[str, Any]:
    """Re-install every ClawHub/skills.sh skill that has a recorded source.

    Existing skills without a ``.clawhub-source.json`` marker (e.g. added
    before the marker existed) are skipped and reported as ``skipped`` so
    the UI can tell the user they were not updated.
    """
    updated: list[str] = []
    skipped: list[str] = []
    errors: list[dict[str, str]] = []

    for folder in _safe_skill_dirs(skills_dir):
        slug = folder.name
        meta = _read_source_meta(skills_dir, slug)
        if meta is None:
            skipped.append(slug)
            continue
        try:
            if meta["kind"] == "skills-sh":
                skills_sh_install(meta["reference"], skills_dir)
            else:
                _clawhub_download_install(meta["reference"], skills_dir)
            updated.append(slug)
        except ClawhubError as exc:
            errors.append({"slug": slug, "error": str(exc)})

    return {"updated": updated, "skipped": skipped, "errors": errors}


def _github_default_branch(owner: str, repo: str) -> str:
    """Resolve the repository's default branch (needed for raw URLs)."""
    data = _github_get_json(f"https://api.github.com/repos/{owner}/{repo}")
    branch = data.get("default_branch")
    if not isinstance(branch, str) or not branch:
        raise ClawhubError("Could not resolve the repository default branch")
    return branch


def _github_get_json(url: str) -> dict[str, Any]:
    try:
        response = httpx.get(url, timeout=_TIMEOUT, follow_redirects=True)
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPStatusError as exc:
        raise ClawhubError(f"GitHub responded {exc.response.status_code}") from exc
    except httpx.HTTPError as exc:
        raise ClawhubError(f"Could not reach GitHub: {exc}") from exc
    except ValueError as exc:
        raise ClawhubError("GitHub returned an invalid response") from exc
    if not isinstance(data, dict):
        raise ClawhubError("GitHub returned an invalid response")
    return data


def _github_get_raw(url: str) -> bytes:
    try:
        response = httpx.get(url, timeout=_TIMEOUT, follow_redirects=True)
        response.raise_for_status()
        return response.content
    except httpx.HTTPStatusError as exc:
        raise ClawhubError(f"GitHub responded {exc.response.status_code}") from exc
    except httpx.HTTPError as exc:
        raise ClawhubError(f"Could not reach GitHub: {exc}") from exc


def _skill_folder_paths(entries: list[dict[str, Any]], slug: str) -> list[str]:
    """Return candidate skill folder paths (containing SKILL.md) sorted by preference.

    The skills.sh convention is ``skills/<slug>/SKILL.md``; we prefer that,
    then any ``*/<slug>/SKILL.md``, then ``<slug>/SKILL.md`` at the root.
    """
    skl = slug.lower()
    candidates: list[str] = []
    for entry in entries:
        path = entry.get("path") or ""
        if path.endswith(f"/{slug}/SKILL.md") or path.endswith(f"/{skl}/SKILL.md"):
            folder = path[: -len("/SKILL.md")]
            candidates.append(folder)
    if not candidates:
        return []

    def _rank(folder: str) -> tuple[int, int]:
        parts = folder.split("/")
        if len(parts) == 2 and parts[0].lower() == "skills":
            return (0, 0)
        return (1, len(parts))

    candidates.sort(key=_rank)
    return candidates


def clawhub_install(reference: str, skills_dir: Path) -> dict[str, Any]:
    """Install a skill from ClawHub or skills.sh by reference.

    References starting with ``skills-sh:`` (or ``owner/repo@slug``) are
    installed from their GitHub source; anything else goes through the
    ClawHub download API.
    """
    if reference.strip().startswith("skills-sh:") or (
        reference.count("/") == 1 and "@" in reference
    ):
        return skills_sh_install(reference, skills_dir)
    return _clawhub_download_install(reference, skills_dir)
