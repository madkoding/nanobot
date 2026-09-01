import sys
import types
from pathlib import Path

import pytest

from nanobot.agent.tools.tts import TtsTool


class _FakeStyle:
    pass


class _FakeEngine:
    """Minimal stand-in for supertonic.TTS."""

    sample_rate = 44100

    def __init__(self) -> None:
        self.saved: list[tuple[object, str]] = []

    def get_voice_style(self, voice: str) -> _FakeStyle:
        return _FakeStyle()

    def synthesize(self, text, voice_style=None, lang=None, speed=None):
        n = int(0.5 * self.sample_rate)
        return [0.0] * n, [0.5]

    def save_audio(self, wav, output_path: str) -> None:
        self.saved.append((wav, output_path))
        Path(output_path).write_bytes(b"WAVDATA")


def _install_fake_supertonic(monkeypatch, engine: _FakeEngine) -> None:
    mod = types.ModuleType("supertonic")
    mod.TTS = lambda *args, **kwargs: engine  # noqa: ARG005 - fake ctor
    monkeypatch.setitem(sys.modules, "supertonic", mod)
    # Reset the process-wide singleton so the fake engine is used.
    monkeypatch.setattr("nanobot.agent.tools.tts._supertonic_engine", None)


def _install_fake_edge_tts(monkeypatch, written: dict[str, Path], payload: bytes = b"MP3DATA") -> None:
    mod = types.ModuleType("edge_tts")

    class _FakeCommunicate:
        def __init__(self, text, voice=None, rate=None, volume=None, pitch=None):
            self.text = text
            self.kwargs = {"voice": voice, "rate": rate, "volume": volume, "pitch": pitch}

        async def save(self, path):
            written["path"] = Path(path)
            Path(path).write_bytes(payload)

    mod.Communicate = _FakeCommunicate
    monkeypatch.setitem(sys.modules, "edge_tts", mod)


@pytest.mark.asyncio
async def test_tts_tool_returns_mp3_when_ffmpeg_missing(monkeypatch, tmp_path) -> None:
    fake_media = tmp_path / "media"
    monkeypatch.setattr("nanobot.agent.tools.tts.get_media_dir", lambda *_: fake_media)
    monkeypatch.setattr("shutil.which", lambda name: None)

    written: dict[str, Path] = {}
    _install_fake_edge_tts(monkeypatch, written)

    tool = TtsTool()
    result = await tool.execute(text="hola", voice="es-CL-CatalinaNeural")

    out = written["path"]
    assert out.exists()
    assert out.suffix == ".mp3"
    assert fake_media in out.parents
    assert "es-CL-CatalinaNeural" in result
    assert str(out) in result
    assert "raw MP3" in result or "ffmpeg unavailable" in result


@pytest.mark.asyncio
async def test_tts_tool_rejects_empty_text() -> None:
    tool = TtsTool()
    result = await tool.execute(text="   ")
    assert result.startswith("Error:")


@pytest.mark.asyncio
async def test_tts_tool_import_error(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "edge_tts", None)
    tool = TtsTool()
    result = await tool.execute(text="hola", engine="edge")
    assert result.startswith("Error:")
    assert "edge-tts" in result


@pytest.mark.asyncio
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="fake ffmpeg is a POSIX shell script (#!/bin/sh)",
)
async def test_tts_tool_reencode_pipeline(monkeypatch, tmp_path) -> None:
    """Simulates the ffmpeg re-encode pipeline and asserts the MP3 is replaced."""
    fake_media = tmp_path / "media"
    monkeypatch.setattr("nanobot.agent.tools.tts.get_media_dir", lambda *_: fake_media)

    fake_ffmpeg = tmp_path / "ffmpeg"
    fake_ffmpeg.write_text(
        "#!/bin/sh\nlast_mp3=\"\"\nfor arg in \"$@\"; do\n  case \"$arg\" in\n    *.mp3) last_mp3=\"$arg\" ;;\n  esac\ndone\nprintf 'MP3MP3MP3' > \"$last_mp3\"\nexit 0\n",
        encoding="utf-8",
    )
    fake_ffmpeg.chmod(0o755)
    monkeypatch.setattr("shutil.which", lambda name: str(fake_ffmpeg) if name == "ffmpeg" else None)

    written: dict[str, Path] = {}
    _install_fake_edge_tts(monkeypatch, written)

    tool = TtsTool()
    result = await tool.execute(text="hola", voice="es-CL-CatalinaNeural")

    out = written["path"]
    assert out.exists()
    assert out.suffix == ".mp3"
    assert "WhatsApp-friendly MP3" in result
    assert str(out) in result


@pytest.mark.asyncio
async def test_tts_tool_supertonic_default_engine(monkeypatch, tmp_path) -> None:
    """Default engine is Supertonic: WAV output, voice/lang/speed in result."""
    fake_media = tmp_path / "media"
    monkeypatch.setattr("nanobot.agent.tools.tts.get_media_dir", lambda *_: fake_media)

    engine = _FakeEngine()
    _install_fake_supertonic(monkeypatch, engine)

    tool = TtsTool()
    result = await tool.execute(text="hola")

    assert "engine=supertonic" in result
    assert "voice=F1" in result
    assert "lang=es" in result
    assert "duration=0.5s" in result
    assert ".wav" in result
    assert engine.saved, "save_audio was never called"
    out = Path(engine.saved[0][1])
    assert out.exists()
    assert out.suffix == ".wav"
    assert fake_media in out.parents


@pytest.mark.asyncio
async def test_tts_tool_supertonic_custom_voice_lang_speed(monkeypatch, tmp_path) -> None:
    fake_media = tmp_path / "media"
    monkeypatch.setattr("nanobot.agent.tools.tts.get_media_dir", lambda *_: fake_media)

    engine = _FakeEngine()
    _install_fake_supertonic(monkeypatch, engine)

    tool = TtsTool()
    result = await tool.execute(
        text="hola", voice="M3", lang="en", speed=1.2, engine="supertonic"
    )

    assert "engine=supertonic" in result
    assert "voice=M3" in result
    assert "lang=en" in result
    assert "speed=1.2" in result


@pytest.mark.asyncio
async def test_tts_tool_supertonic_import_error(monkeypatch, tmp_path) -> None:
    fake_media = tmp_path / "media"
    monkeypatch.setattr("nanobot.agent.tools.tts.get_media_dir", lambda *_: fake_media)
    monkeypatch.setitem(sys.modules, "supertonic", None)
    monkeypatch.setattr("nanobot.agent.tools.tts._supertonic_engine", None)

    tool = TtsTool()
    result = await tool.execute(text="hola")

    assert result.startswith("Error:")
    assert "supertonic" in result


@pytest.mark.asyncio
async def test_tts_tool_supertonic_mp3_reencode(monkeypatch, tmp_path) -> None:
    """With ffmpeg available, the WAV is re-encoded to MP3 and the WAV removed."""
    fake_media = tmp_path / "media"
    monkeypatch.setattr("nanobot.agent.tools.tts.get_media_dir", lambda *_: fake_media)

    fake_ffmpeg = tmp_path / "ffmpeg"
    fake_ffmpeg.write_text(
        "#!/bin/sh\nlast_mp3=\"\"\nfor arg in \"$@\"; do\n  case \"$arg\" in\n    *.mp3) last_mp3=\"$arg\" ;;\n  esac\ndone\nprintf 'MP3MP3MP3' > \"$last_mp3\"\nexit 0\n",
        encoding="utf-8",
    )
    fake_ffmpeg.chmod(0o755)
    monkeypatch.setattr("nanobot.agent.tools.tts._find_ffmpeg", lambda: str(fake_ffmpeg))

    engine = _FakeEngine()
    _install_fake_supertonic(monkeypatch, engine)

    tool = TtsTool()
    result = await tool.execute(text="hola")

    assert "Re-encoded to MP3" in result
    assert ".mp3" in result
    wav_path = Path(engine.saved[0][1])
    assert not wav_path.exists(), "WAV should be removed after MP3 re-encode"
    mp3_path = wav_path.with_suffix(".mp3")
    assert mp3_path.exists()
    assert mp3_path.read_bytes() == b"MP3MP3MP3"


@pytest.mark.asyncio
async def test_tts_tool_unknown_engine_rejected() -> None:
    tool = TtsTool()
    result = await tool.execute(text="hola", engine="klingon")
    assert result.startswith("Error:")
    assert "klingon" in result


@pytest.mark.asyncio
async def test_tts_tool_edge_voice_auto_fallback(monkeypatch, tmp_path) -> None:
    """An edge-tts voice id with the default engine falls back to edge-tts."""
    fake_media = tmp_path / "media"
    monkeypatch.setattr("nanobot.agent.tools.tts.get_media_dir", lambda *_: fake_media)
    monkeypatch.setattr("shutil.which", lambda name: None)

    written: dict[str, Path] = {}
    _install_fake_edge_tts(monkeypatch, written)

    tool = TtsTool()
    result = await tool.execute(text="hola", voice="es-CL-CatalinaNeural")

    assert "engine=edge" in result
    assert "es-CL-CatalinaNeural" in result
    assert written["path"].suffix == ".mp3"


@pytest.mark.asyncio
async def test_tts_tool_explicit_edge_engine(monkeypatch, tmp_path) -> None:
    fake_media = tmp_path / "media"
    monkeypatch.setattr("nanobot.agent.tools.tts.get_media_dir", lambda *_: fake_media)
    monkeypatch.setattr("shutil.which", lambda name: None)

    written: dict[str, Path] = {}
    _install_fake_edge_tts(monkeypatch, written)

    tool = TtsTool()
    result = await tool.execute(text="hola", engine="edge")

    assert "engine=edge" in result
    assert "es-CL-CatalinaNeural" in result
    assert written["path"].suffix == ".mp3"
