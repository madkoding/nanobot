"""Tests for the WebUI Projects controller (storage + CRUD + files)."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from nanobot.webui.projects import (
    ProjectError,
    WebUIProjectsController,
)


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Use a tmp data dir for project storage."""
    d = tmp_path / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _data_url(mime: str, payload: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(payload).decode()}"


def _create(controller: WebUIProjectsController, name: str, instructions: str = ""):
    return controller.create_project(name, instructions)


def test_create_project_persists_metadata(data_dir: Path) -> None:
    c = WebUIProjectsController(data_dir)
    summary = c.create_project("Demo", "follow the demo steps")
    assert summary.id == "demo"
    assert summary.name == "Demo"
    assert summary.instructions_md == "follow the demo steps"
    assert summary.file_count == 0
    assert summary.byte_count == 0
    pdir = data_dir / "projects" / "demo"
    assert (pdir / "project.json").is_file()
    assert (pdir / "files").is_dir()


def test_create_project_rejects_blank_name(data_dir: Path) -> None:
    c = WebUIProjectsController(data_dir)
    with pytest.raises(ProjectError):
        c.create_project("   ", "")


def test_create_project_unique_id_on_collision(data_dir: Path) -> None:
    c = WebUIProjectsController(data_dir)
    a = c.create_project("Demo", "")
    b = c.create_project("Demo", "")
    assert a.id == "demo"
    assert b.id == "demo-2"


def test_list_projects_returns_summaries_sorted(data_dir: Path) -> None:
    c = WebUIProjectsController(data_dir)
    c.create_project("Alpha", "")
    c.create_project("Beta", "")
    listed = [s.name for s in c.list_projects()]
    assert listed == ["Alpha", "Beta"]


def test_update_project_preserves_id(data_dir: Path) -> None:
    c = WebUIProjectsController(data_dir)
    s = c.create_project("Old name", "old instructions")
    updated = c.update_project(s.id, "New name", "new instructions")
    assert updated.id == s.id
    assert updated.name == "New name"
    assert updated.instructions_md == "new instructions"
    assert updated.updated_at_ms >= s.updated_at_ms


def test_update_project_rejects_blank_name(data_dir: Path) -> None:
    c = WebUIProjectsController(data_dir)
    s = c.create_project("Foo", "")
    with pytest.raises(ProjectError):
        c.update_project(s.id, "", "x")


def test_update_project_404(data_dir: Path) -> None:
    c = WebUIProjectsController(data_dir)
    with pytest.raises(ProjectError):
        c.update_project("missing", "x", "y")


def test_delete_project_removes_files(data_dir: Path) -> None:
    c = WebUIProjectsController(data_dir)
    s = c.create_project("Tmp", "")
    f = c.add_file(s.id, "a.txt", _data_url("text/plain", b"hello"))
    data_path = data_dir / "projects" / s.id / "files" / f"{f.id}.bin"
    assert data_path.is_file()
    c.delete_project(s.id)
    assert not (data_dir / "projects" / s.id).exists()


def test_add_file_stores_payload(data_dir: Path) -> None:
    c = WebUIProjectsController(data_dir)
    s = c.create_project("Tmp", "")
    f = c.add_file(s.id, "x.txt", _data_url("text/plain", b"hello world"))
    assert f.size == len(b"hello world")
    assert f.mime_type == "text/plain"
    pdir = data_dir / "projects" / s.id
    bin_path = pdir / "files" / f"{f.id}.bin"
    meta_path = pdir / "files" / f"{f.id}.meta.json"
    assert bin_path.is_file()
    assert bin_path.read_bytes() == b"hello world"
    assert meta_path.is_file()


def test_add_file_rejects_invalid_data_url(data_dir: Path) -> None:
    c = WebUIProjectsController(data_dir)
    s = c.create_project("Tmp", "")
    with pytest.raises(ProjectError):
        c.add_file(s.id, "x.txt", "not-a-data-url")


def test_add_file_rejects_blank_name(data_dir: Path) -> None:
    c = WebUIProjectsController(data_dir)
    s = c.create_project("Tmp", "")
    with pytest.raises(ProjectError):
        c.add_file(s.id, "", _data_url("text/plain", b"x"))


def test_add_file_invalid_base64(data_dir: Path) -> None:
    c = WebUIProjectsController(data_dir)
    s = c.create_project("Tmp", "")
    with pytest.raises(ProjectError):
        c.add_file(s.id, "x.txt", "data:text/plain;base64,***not-valid***")


def test_list_files_returns_metadata_only(data_dir: Path) -> None:
    c = WebUIProjectsController(data_dir)
    s = c.create_project("Tmp", "")
    c.add_file(s.id, "a.txt", _data_url("text/plain", b"a"))
    c.add_file(s.id, "b.bin", _data_url("application/octet-stream", b"\x00\x01"))
    files = c.list_files(s.id)
    assert {f.name for f in files} == {"a.txt", "b.bin"}
    assert all(f.size > 0 for f in files)


def test_read_file_round_trips_payload(data_dir: Path) -> None:
    c = WebUIProjectsController(data_dir)
    s = c.create_project("Tmp", "")
    f = c.add_file(s.id, "a.txt", _data_url("text/plain", b"payload-x"))
    payload, info = c.read_file(s.id, f.id)
    assert payload == b"payload-x"
    assert info.id == f.id
    assert info.size == len(b"payload-x")


def test_read_file_404(data_dir: Path) -> None:
    c = WebUIProjectsController(data_dir)
    s = c.create_project("Tmp", "")
    with pytest.raises(ProjectError):
        c.read_file(s.id, "missing-file-id")


def test_delete_file_removes_payload(data_dir: Path) -> None:
    c = WebUIProjectsController(data_dir)
    s = c.create_project("Tmp", "")
    f = c.add_file(s.id, "a.txt", _data_url("text/plain", b"x"))
    c.delete_file(s.id, f.id)
    assert c.list_files(s.id) == []
    assert not (data_dir / "projects" / s.id / "files" / f"{f.id}.bin").exists()


def test_delete_file_404(data_dir: Path) -> None:
    c = WebUIProjectsController(data_dir)
    s = c.create_project("Tmp", "")
    with pytest.raises(ProjectError):
        c.delete_file(s.id, "missing")


def test_add_file_touches_project_updated_at(data_dir: Path) -> None:
    c = WebUIProjectsController(data_dir)
    s = c.create_project("Tmp", "")
    before = c.get_project(s.id).updated_at_ms
    c.add_file(s.id, "a.txt", _data_url("text/plain", b"x"))
    after = c.get_project(s.id).updated_at_ms
    assert after >= before


def test_payload_builders_no_io(data_dir: Path) -> None:
    from nanobot.webui.projects import (
        project_detail_payload,
        project_file_payload,
        projects_list_payload,
    )

    c = WebUIProjectsController(data_dir)
    s = c.create_project("Demo", "instr")
    f = c.add_file(s.id, "x.txt", _data_url("text/plain", b"hi"))
    listing = projects_list_payload(c)
    assert listing["projects"][0]["id"] == s.id
    detail = project_detail_payload(c, s.id)
    assert detail["id"] == s.id
    assert len(detail["files"]) == 1
    file_payload = project_file_payload(c, s.id, f.id)
    assert file_payload["name"] == "x.txt"
    assert file_payload["data_url"].startswith("data:text/plain;base64,")


def test_corrupt_project_json_raises(data_dir: Path) -> None:
    """A project whose project.json is unparseable should be skipped on list and raise on get."""
    c = WebUIProjectsController(data_dir)
    pdir = data_dir / "projects" / "broken"
    pdir.mkdir(parents=True)
    (pdir / "project.json").write_text("not-json{", encoding="utf-8")
    assert c.list_projects() == []
    with pytest.raises(ProjectError):
        c.get_project("broken")


def test_add_folder_persists_and_lists(data_dir: Path) -> None:
    c = WebUIProjectsController(data_dir)
    s = c.create_project("Demo", "")
    f = c.add_folder(s.id, "/tmp/alpha")
    assert f.path == "/tmp/alpha"
    assert c.list_folders(s.id) == [f]
    assert c.get_project(s.id).folder_count == 1


def test_add_folder_rejects_blank_and_duplicate(data_dir: Path) -> None:
    c = WebUIProjectsController(data_dir)
    s = c.create_project("Demo", "")
    with pytest.raises(ProjectError):
        c.add_folder(s.id, "   ")
    c.add_folder(s.id, "/tmp/alpha")
    with pytest.raises(ProjectError):
        c.add_folder(s.id, "/tmp/alpha")


def test_remove_folder(data_dir: Path) -> None:
    c = WebUIProjectsController(data_dir)
    s = c.create_project("Demo", "")
    c.add_folder(s.id, "/tmp/alpha")
    c.add_folder(s.id, "/tmp/beta")
    c.remove_folder(s.id, "/tmp/alpha")
    assert [f.path for f in c.list_folders(s.id)] == ["/tmp/beta"]
    with pytest.raises(ProjectError):
        c.remove_folder(s.id, "/tmp/missing")


def test_folder_round_trips_after_reload(data_dir: Path) -> None:
    c = WebUIProjectsController(data_dir)
    s = c.create_project("Demo", "")
    c.add_folder(s.id, "/tmp/alpha")
    c2 = WebUIProjectsController(data_dir)
    assert [f.path for f in c2.list_folders(s.id)] == ["/tmp/alpha"]


def test_folder_in_detail_payload(data_dir: Path) -> None:
    from nanobot.webui.projects import project_detail_payload

    c = WebUIProjectsController(data_dir)
    s = c.create_project("Demo", "")
    c.add_folder(s.id, "/tmp/alpha")
    detail = project_detail_payload(c, s.id)
    assert detail["folders"] == [
        {"path": "/tmp/alpha", "created_at_ms": detail["folders"][0]["created_at_ms"]}
    ]
