import sys
import types
from pathlib import Path

import pytest

from nanobot.agent.tools.tts import TtsTool


def _install_fake_edge_tts(
    monkeypatch, written: dict[str, Path], payload: bytes = b"MP3DATA"
) -> None:
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
    result = await tool.execute(text="hola")
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
        '#!/bin/sh\nlast_mp3=""\nfor arg in "$@"; do\n  case "$arg" in\n    *.mp3) last_mp3="$arg" ;;\n  esac\ndone\nprintf \'MP3MP3MP3\' > "$last_mp3"\nexit 0\n',
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
