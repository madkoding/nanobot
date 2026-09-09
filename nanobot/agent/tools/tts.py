"""Text-to-speech tool: local neural TTS via Supertonic 3, with edge-tts fallback."""

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
from nanobot.agent.tools.schema import NumberSchema, StringSchema, tool_parameters_schema
from nanobot.config.paths import get_media_dir

DEFAULT_ENGINE = "supertonic"
DEFAULT_VOICE = "F1"  # Supertonic 3 built-in voices: M1..M5 (male), F1..F5 (female)
DEFAULT_LANG = "es"
DEFAULT_SPEED = 1.05
SUPERTONIC_VOICES = frozenset(f"{g}{i}" for g in ("M", "F") for i in range(1, 6))

# edge-tts fallback defaults (kept for backward compatibility)
EDGE_DEFAULT_VOICE = "es-CL-CatalinaNeural"
EDGE_DEFAULT_RATE = "+0%"
EDGE_DEFAULT_VOLUME = "+0%"
EDGE_DEFAULT_PITCH = "+8Hz"

# ponytail: re-encode edge-tts MP3 to a WhatsApp-mobile-friendly MP3.
# Edge-tts emits 24kHz mono 48kbps which many mobile clients refuse to download.
# 44.1kHz stereo 64kbps is the most universally accepted audio format.
REENCODE = True
MP3_BITRATE = "64k"
MP3_SAMPLE_RATE = 44100

# Supertonic 3 engine is loaded once per process (~1s, ~400MB model in RAM).
_engine_lock = asyncio.Lock()
_supertonic_engine: Any = None


def _find_ffmpeg() -> str | None:
    """Locate an ffmpeg binary: PATH first, then the imageio-ffmpeg static build."""
    bin_path = shutil.which("ffmpeg")
    if bin_path:
        return bin_path
    try:
        import imageio_ffmpeg  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - any ffmpeg discovery failure means "no ffmpeg"
        return None


async def _ffmpeg_to_mp3(src: Path, dst: Path) -> tuple[bool, str]:
    """Re-encode src to a WhatsApp-mobile-friendly MP3 at dst.

    Returns (ok, note). On failure dst is removed and note explains why.
    """
    ffmpeg_bin = _find_ffmpeg()
    if not ffmpeg_bin:
        return False, "ffmpeg unavailable"

    cmd = [
        ffmpeg_bin,
        "-y",
        "-loglevel", "error",
        "-i", str(src),
        "-vn",
        # Standard stereo MP3 that WhatsApp mobile reliably downloads and plays.
        "-c:a", "libmp3lame",
        "-b:a", MP3_BITRATE,
        "-ar", str(MP3_SAMPLE_RATE),
        "-ac", "2",
        "-id3v2_version", "0",
        "-write_id3v1", "0",
        str(dst),
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return False, "ffmpeg binary missing"

    stdout, stderr = await proc.communicate()
    if proc.returncode != 0 or not dst.is_file() or dst.stat().st_size == 0:
        logger.warning("ffmpeg MP3 re-encode failed: {}", (stderr or b"").decode("utf-8", "replace"))
        dst.unlink(missing_ok=True)
        return False, "MP3 re-encode failed"
    return True, ""


async def _get_supertonic_engine() -> Any:
    """Return the lazily-loaded Supertonic 3 TTS engine (process-wide singleton)."""
    global _supertonic_engine
    if _supertonic_engine is not None:
        return _supertonic_engine
    async with _engine_lock:
        if _supertonic_engine is not None:
            return _supertonic_engine
        try:
            from supertonic import TTS  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                f"supertonic is not installed ({exc.__class__.__name__})"
            ) from exc
        # Model load is CPU-bound; keep the event loop responsive.
        _supertonic_engine = await asyncio.to_thread(TTS)
        return _supertonic_engine


# Kokoro-82M: 82M-param StyleTTS2-based model, Apache 2.0, 54 voices,
# runs on CPU (~1.2s load, >3x real-time synthesis). Loaded once per (lang)
# process and cached; each lang code has its own pipeline (G2P differs).
# Voces es: ef_dora (f), em_alex/em_santa (m). Sample rate fijo 24 kHz.
KOKORO_DEFAULT_VOICE = "ef_dora"
KOKORO_VOICES = frozenset(
    {
        # es (e):
        "ef_dora", "em_alex", "em_santa",
        # en (a):
        "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
        "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah",
        "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir",
        "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa",
        # fr (f):
        "ff_siwis",
        # hi (h):
        "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
        # it (i):
        "if_sara", "im_nicola",
        # ja (j):
        "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro",
        "jm_kumo",
        # pt (p):
        "pf_dora", "pm_alex", "pm_santa",
        # zh (z):
        "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
        "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
    }
)
# El modelo (Kokoro-82M) no clona voces: todos los voice ids nacen de un
# prefijo de idioma (a=en, b=inglés británico, e=es, f=fr, h=hi, i=it, j=ja,
# p=pt, z=zh) — el lang se deriva del prefijo, no requiere config extra.
KOKORO_SAMPLE_RATE = 24000

_kokoro_lock = asyncio.Lock()
_kokoro_pipelines: dict[str, Any] = {}


def _kokoro_lang_for_voice(voice: str) -> str:
    """Map a Kokoro voice prefix to the KPipeline lang_code (G2P)."""
    first = voice[0] if voice else ""
    return {
        "a": "a",  # en
        "b": "b",  # en (británico)
        "e": "e",  # es
        "f": "f",  # fr
        "h": "h",  # hi
        "i": "i",  # it
        "j": "j",  # ja
        "p": "p",  # pt
        "z": "z",  # zh
    }.get(first, "e")


async def _get_kokoro_pipeline(lang_code: str) -> Any:
    """Return a lazily-loaded Kokoro KPipeline for a lang code (cached per lang)."""
    global _kokoro_pipelines
    pipeline = _kokoro_pipelines.get(lang_code)
    if pipeline is not None:
        return pipeline
    async with _kokoro_lock:
        pipeline = _kokoro_pipelines.get(lang_code)
        if pipeline is not None:
            return pipeline
        try:
            from kokoro import KPipeline  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                f"kokoro is not installed ({exc.__class__.__name__}); "
                f"install with: uv pip install kokoro"
            ) from exc
        # Model load is CPU-bound; keep the event loop responsive.
        pipeline = await asyncio.to_thread(KPipeline, lang_code=lang_code)
        _kokoro_pipelines[lang_code] = pipeline
        return pipeline


@tool_parameters(
    tool_parameters_schema(
        text=StringSchema("Text to synthesize into speech."),
        voice=StringSchema(
            f"Voice id. Supertonic 3 built-ins: 'M1'..'M5' (male), 'F1'..'F5' (female). "
            f"Kokoro voices (es): 'ef_dora' (f), 'em_alex'/'em_santa' (m), default '{KOKORO_DEFAULT_VOICE}' "
            f"when engine='kokoro'. An edge-tts voice id (e.g. 'es-CL-CatalinaNeural') "
            f"auto-switches to the edge engine."
        ),
        engine=StringSchema(
            f"TTS engine: 'supertonic' (local neural, default), 'kokoro' (Kokoro-82M, local) "
            f"or 'edge' (Microsoft edge-tts). Default '{DEFAULT_ENGINE}'."
        ),
        lang=StringSchema(
            f"Language code for Supertonic (e.g. 'es', 'en', 'na' auto-detect). "
            f"Default '{DEFAULT_LANG}'. For Kokoro the language comes from the voice prefix "
            f"(ef_/em_ = es) and is ignored."
        ),
        speed=NumberSchema(
            f"Speech speed multiplier for Supertonic. Default {DEFAULT_SPEED}."
        ),
        rate=StringSchema(f"edge-tts rate (e.g. '+10%', '-5%'). Default '{EDGE_DEFAULT_RATE}'."),
        volume=StringSchema(f"edge-tts volume. Default '{EDGE_DEFAULT_VOLUME}'."),
        pitch=StringSchema(f"edge-tts pitch. Default '{EDGE_DEFAULT_PITCH}'."),
    )
)
class TtsTool(Tool):
    """Synthesize speech from text and return a local audio file path."""

    @property
    def name(self) -> str:
        return "tts"

    @property
    def description(self) -> str:
        return (
            "Text-to-speech via Supertonic 3 (local neural TTS, default), Kokoro-82M "
            "(local) or edge-tts fallback. Returns audio path. Use message tool with "
            "media=[path] to deliver."
        )

    async def execute(
        self,
        text: str,
        voice: str | None = None,
        engine: str | None = None,
        lang: str | None = None,
        speed: float | None = None,
        rate: str | None = None,
        volume: str | None = None,
        pitch: str | None = None,
        **_kwargs: Any,
    ) -> str:
        text = (text or "").strip()
        if not text:
            return ToolResult.error("Error: text is required")

        chosen_engine = (engine or DEFAULT_ENGINE).strip().lower()
        if chosen_engine not in ("supertonic", "kokoro", "edge"):
            return ToolResult.error(
                f"Error: unknown engine '{chosen_engine}' "
                f"(use 'supertonic', 'kokoro' or 'edge')"
            )

        if chosen_engine == "supertonic":
            chosen_voice = (voice or DEFAULT_VOICE).strip() or DEFAULT_VOICE
            # An edge-tts voice id with the supertonic engine → auto-fallback to edge.
            if chosen_voice not in SUPERTONIC_VOICES:
                chosen_engine = "edge"
        elif chosen_engine == "kokoro":
            chosen_voice = (voice or KOKORO_DEFAULT_VOICE).strip() or KOKORO_DEFAULT_VOICE
            # An edge-tts voice id (e.g. es-CL-*) or a Supertonic voice (M1..F5)
            # with the kokoro engine → auto-fallback to edge.
            if chosen_voice not in KOKORO_VOICES:
                chosen_engine = "edge"
        else:
            chosen_voice = (voice or EDGE_DEFAULT_VOICE).strip() or EDGE_DEFAULT_VOICE

        # ponytail: per-session tag, no global lock needed; switch to a
        # semaphore if TTS contention with other tools becomes visible.
        session_key = current_request_session_key()
        workspace_tag = (
            session_key.replace(":", "_").replace("/", "_") if session_key else "shared"
        )
        media_dir = get_media_dir("tts")
        media_dir.mkdir(parents=True, exist_ok=True)
        token = secrets.token_hex(4)

        if chosen_engine == "supertonic":
            return await self._synthesize_supertonic(
                text, chosen_voice, lang, speed, media_dir, workspace_tag, token
            )
        if chosen_engine == "kokoro":
            return await self._synthesize_kokoro(
                text, chosen_voice, media_dir, workspace_tag, token
            )
        return await self._synthesize_edge(
            text, chosen_voice, rate, volume, pitch, media_dir, workspace_tag, token
        )

    async def _synthesize_supertonic(
        self,
        text: str,
        voice: str,
        lang: str | None,
        speed: float | None,
        media_dir: Path,
        workspace_tag: str,
        token: str,
    ) -> str:
        try:
            engine = await _get_supertonic_engine()
        except RuntimeError as exc:
            return ToolResult.error(f"Error: {exc}")

        chosen_lang = (lang or DEFAULT_LANG).strip() or DEFAULT_LANG
        chosen_speed = float(speed) if speed is not None else DEFAULT_SPEED

        wav_path = media_dir / f"tts_{workspace_tag}_{token}.wav"
        try:
            style = await asyncio.to_thread(engine.get_voice_style, voice)
            wav, dur = await asyncio.to_thread(
                engine.synthesize,
                text,
                voice_style=style,
                lang=chosen_lang,
                speed=chosen_speed,
            )
            await asyncio.to_thread(engine.save_audio, wav, str(wav_path))
        except Exception as exc:  # noqa: BLE001 - surface upstream errors verbatim
            return ToolResult.error(f"Error: Supertonic synthesis failed: {exc}")

        if not wav_path.is_file() or wav_path.stat().st_size == 0:
            return ToolResult.error("Error: TTS produced no audio output")

        try:
            duration = float(dur[0])
        except (TypeError, ValueError, IndexError):
            duration = 0.0

        # Telegram no muestra la duración de los WAV (los trata como documentos);
        # re-encode a MP3 para que el cliente muestre el largo del audio.
        mp3_path = media_dir / f"tts_{workspace_tag}_{token}.mp3"
        ok, note = await _ffmpeg_to_mp3(wav_path, mp3_path)
        if ok:
            try:
                wav_path.unlink()
            except OSError:
                pass
            out_path = mp3_path
            format_note = f"Re-encoded to MP3 ({MP3_BITRATE}, {MP3_SAMPLE_RATE}Hz stereo)."
        else:
            out_path = wav_path
            format_note = f"MP3 re-encode skipped ({note}); shipped WAV."

        return (
            f"TTS audio saved to {out_path}\n"
            f"engine=supertonic voice={voice} lang={chosen_lang} speed={chosen_speed} "
            f"duration={duration:.1f}s\n"
            f"{format_note}\n"
            f"Attach it via the 'message' tool with media=[{out_path}]."
        )

    async def _synthesize_kokoro(
        self,
        text: str,
        voice: str,
        media_dir: Path,
        workspace_tag: str,
        token: str,
    ) -> str:
        lang_code = _kokoro_lang_for_voice(voice)
        try:
            pipeline = await _get_kokoro_pipeline(lang_code)
        except RuntimeError as exc:
            return ToolResult.error(f"Error: {exc}")

        wav_path = media_dir / f"tts_{workspace_tag}_{token}.wav"
        try:
            gen = await asyncio.to_thread(pipeline, text, voice=voice)
            # Consume el primer chunk (y siguientes) en un hilo: la síntesis es
            # CPU-bound (torch) y no debe bloquear el event loop.
            def _synthesize() -> tuple[list[float], float]:
                audio_parts: list[list[float]] = []
                duration = 0.0
                for _, _, audio in gen:
                    chunk = audio.tolist() if hasattr(audio, "tolist") else list(audio)
                    audio_parts.append(chunk)
                    duration += len(chunk) / KOKORO_SAMPLE_RATE
                samples: list[float] = []
                for part in audio_parts:
                    samples.extend(part)
                return samples, duration

            samples, duration = await asyncio.to_thread(_synthesize)
            if not samples:
                return ToolResult.error("Error: Kokoro produced no audio output")
            # Guardar WAV 16-bit PCM 24 kHz.
            import array

            pcm = array.array("h", (int(max(-1.0, min(1.0, s)) * 32767) for s in samples))
            with wav_path.open("wb") as fh:
                self._write_wav_header(fh, KOKORO_SAMPLE_RATE, len(pcm))
                pcm.tofile(fh)
        except Exception as exc:  # noqa: BLE001 - surface upstream errors verbatim
            return ToolResult.error(f"Error: Kokoro synthesis failed: {exc}")

        if not wav_path.is_file() or wav_path.stat().st_size == 0:
            return ToolResult.error("Error: TTS produced no audio output")

        # Telegram no muestra la duración de los WAV; re-encode a MP3.
        mp3_path = media_dir / f"tts_{workspace_tag}_{token}.mp3"
        ok, note = await _ffmpeg_to_mp3(wav_path, mp3_path)
        if ok:
            try:
                wav_path.unlink()
            except OSError:
                pass
            out_path = mp3_path
            format_note = f"Re-encoded to MP3 ({MP3_BITRATE}, {MP3_SAMPLE_RATE}Hz stereo)."
        else:
            out_path = wav_path
            format_note = f"MP3 re-encode skipped ({note}); shipped WAV."

        return (
            f"TTS audio saved to {out_path}\n"
            f"engine=kokoro voice={voice} lang={lang_code} duration={duration:.1f}s\n"
            f"{format_note}\n"
            f"Attach it via the 'message' tool with media=[{out_path}]."
        )

    @staticmethod
    def _write_wav_header(fh: Any, sample_rate: int, n_samples: int) -> None:
        """Write a minimal RIFF/WAVE header for 16-bit mono PCM."""
        import struct

        data_size = n_samples * 2
        fh.write(b"RIFF")
        fh.write(struct.pack("<I", 36 + data_size))
        fh.write(b"WAVEfmt ")
        fh.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
        fh.write(b"data")
        fh.write(struct.pack("<I", data_size))

    async def _synthesize_edge(
        self,
        text: str,
        voice: str,
        rate: str | None,
        volume: str | None,
        pitch: str | None,
        media_dir: Path,
        workspace_tag: str,
        token: str,
    ) -> str:
        try:
            import edge_tts  # type: ignore[import-not-found]
        except ImportError as exc:
            return ToolResult.error(
                f"Error: edge-tts is not installed ({exc.__class__.__name__})"
            )

        chosen_rate = rate or EDGE_DEFAULT_RATE
        chosen_volume = volume or EDGE_DEFAULT_VOLUME
        chosen_pitch = pitch or EDGE_DEFAULT_PITCH

        mp3_path = media_dir / f"tts_{workspace_tag}_{token}.mp3"
        try:
            communicate = edge_tts.Communicate(
                text,
                voice=voice,
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
            f"engine=edge voice={voice} rate={chosen_rate} volume={chosen_volume} pitch={chosen_pitch}\n"
            f"{note}\n"
            f"Attach it via the 'message' tool with media=[{out_path}]."
        )


async def _maybe_reencode(mp3_path: Path, enabled: bool) -> tuple[Path, str]:
    """Re-encode the raw edge-tts MP3 to a WhatsApp-mobile-friendly MP3.

    Falls back to the original MP3 if ffmpeg/libmp3lame is unavailable.
    """
    if not enabled:
        return mp3_path, "Raw MP3 from edge-tts."

    tmp_out = Path(tempfile.gettempdir()) / f"tts_{secrets.token_hex(4)}.mp3"
    ok, note = await _ffmpeg_to_mp3(mp3_path, tmp_out)
    if not ok:
        return mp3_path, f"{note}; shipped raw MP3."

    try:
        tmp_out.replace(mp3_path)
    except OSError as exc:
        logger.warning("could not move re-encoded MP3 into place: {}", exc)
        tmp_out.unlink(missing_ok=True)
        return mp3_path, "MP3 output move failed; shipped raw MP3."

    return mp3_path, f"Re-encoded to WhatsApp-friendly MP3 ({MP3_BITRATE}, {MP3_SAMPLE_RATE}Hz stereo)."


