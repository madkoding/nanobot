"""Text-to-speech tool using Microsoft edge-tts."""

from __future__ import annotations

import asyncio
import secrets
import shutil
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool, ToolResult, tool_parameters
from nanobot.agent.tools.context import current_request_session_key
from nanobot.agent.tools.schema import StringSchema, tool_parameters_schema
from nanobot.config.paths import get_media_dir

DEFAULT_VOICE = "es-CL-CatalinaNeural"
DEFAULT_RATE = "+0%"
DEFAULT_VOLUME = "+0%"
DEFAULT_PITCH = "+8Hz"

# ponytail: re-encode edge-tts MP3 to a WhatsApp-mobile-friendly MP3.
# Edge-tts emits 24kHz mono 48kbps which many mobile clients refuse to download.
# 44.1kHz stereo 64kbps is the most universally accepted audio format.
REENCODE = True
MP3_BITRATE = "64k"
MP3_SAMPLE_RATE = 44100


@tool_parameters(
    tool_parameters_schema(
        text=StringSchema("Text to synthesize into speech."),
        voice=StringSchema(
            f"edge-tts voice id. Default '{DEFAULT_VOICE}'. "
            "Run `edge-tts --list-voices` to see all available voices."
        ),
        rate=StringSchema(
            f"Speech rate adjustment (e.g. '+10%', '-5%'). Default '{DEFAULT_RATE}'."
        ),
        volume=StringSchema(f"Volume adjustment. Default '{DEFAULT_VOLUME}'."),
        pitch=StringSchema(
            f"Pitch adjustment in Hz (e.g. '+2Hz', '-3Hz'). Default '{DEFAULT_PITCH}' (higher for a younger, more expressive sound)."
        ),
    )
)
class TtsTool(Tool):
    """Synthesize speech from text and return a local audio file path."""

    @property
    def name(self) -> str:
        return "tts"

    @property
    def description(self) -> str:
        return "Text-to-speech via edge-tts. Returns audio path. Use message tool with media=[path] to deliver."

    async def execute(
        self,
        text: str,
        voice: str | None = None,
        rate: str | None = None,
        volume: str | None = None,
        pitch: str | None = None,
        **_kwargs: Any,
    ) -> str:
        text = (text or "").strip()
        if not text:
            return ToolResult.error("Error: text is required")

        try:
            import edge_tts  # type: ignore[import-not-found]
        except ImportError as exc:
            return ToolResult.error(f"Error: edge-tts is not installed ({exc.__class__.__name__})")

        chosen_voice = (voice or DEFAULT_VOICE).strip() or DEFAULT_VOICE
        chosen_rate = rate or DEFAULT_RATE
        chosen_volume = volume or DEFAULT_VOLUME
        chosen_pitch = pitch or DEFAULT_PITCH

        # ponytail: per-session tag, no global lock needed; switch to a
        # semaphore if TTS contention with other tools becomes visible.
        session_key = current_request_session_key()
        workspace_tag = session_key.replace(":", "_").replace("/", "_") if session_key else "shared"
        media_dir = get_media_dir("tts")
        media_dir.mkdir(parents=True, exist_ok=True)
        token = secrets.token_hex(4)
        mp3_path = media_dir / f"tts_{workspace_tag}_{token}.mp3"

        try:
            communicate = edge_tts.Communicate(
                text,
                voice=chosen_voice,
                rate=chosen_rate,
                volume=chosen_volume,
                pitch=chosen_pitch,
            )
            await communicate.save(str(mp3_path))
        except Exception as exc:  # noqa: BLE001 - surface upstream errors verbatim
            return ToolResult.error(f"Error: TTS synthesis failed: {exc}")

        if not mp3_path.is_file() or mp3_path.stat().st_size == 0:
            return ToolResult.error("Error: TTS produced no audio output")

        out_path, note = await _maybe_reencode(mp3_path, REENCODE)
        if out_path != mp3_path:
            try:
                mp3_path.unlink()
            except OSError:
                pass

        return (
            f"TTS audio saved to {out_path}\n"
            f"voice={chosen_voice} rate={chosen_rate} volume={chosen_volume} pitch={chosen_pitch}\n"
            f"{note}\n"
            f"Attach it via the 'message' tool with media=[{out_path}]."
        )


async def _maybe_reencode(mp3_path: Path, enabled: bool) -> tuple[Path, str]:
    """Re-encode the raw edge-tts MP3 to a WhatsApp-mobile-friendly MP3.

    Falls back to the original MP3 if ffmpeg/libmp3lame is unavailable.
    """
    if not enabled:
        return mp3_path, "Raw MP3 from edge-tts."

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return mp3_path, "ffmpeg unavailable; shipped raw MP3."

    tmp_out = Path(tempfile.gettempdir()) / f"tts_{secrets.token_hex(4)}.mp3"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(mp3_path),
        "-vn",
        # Standard stereo MP3 that WhatsApp mobile reliably downloads and plays.
        "-c:a",
        "libmp3lame",
        "-b:a",
        MP3_BITRATE,
        "-ar",
        str(MP3_SAMPLE_RATE),
        "-ac",
        "2",
        "-id3v2_version",
        "0",
        "-write_id3v1",
        "0",
        str(tmp_out),
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return mp3_path, "ffmpeg binary missing; shipped raw MP3."

    stdout, stderr = await proc.communicate()
    if proc.returncode != 0 or not tmp_out.is_file() or tmp_out.stat().st_size == 0:
        logger.warning(
            "ffmpeg MP3 re-encode failed: {}", (stderr or b"").decode("utf-8", "replace")
        )
        tmp_out.unlink(missing_ok=True)
        return mp3_path, "MP3 re-encode failed; shipped raw MP3."

    try:
        tmp_out.replace(mp3_path)
    except OSError as exc:
        logger.warning("could not move re-encoded MP3 into place: {}", exc)
        tmp_out.unlink(missing_ok=True)
        return mp3_path, "MP3 output move failed; shipped raw MP3."

    return (
        mp3_path,
        f"Re-encoded to WhatsApp-friendly MP3 ({MP3_BITRATE}, {MP3_SAMPLE_RATE}Hz stereo).",
    )
