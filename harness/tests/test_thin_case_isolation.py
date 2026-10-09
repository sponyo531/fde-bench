"""瘦 case：agent 启动前 information.md 必须离开磁盘；仓库原件绝不能被删。"""

from __future__ import annotations

import json
from pathlib import Path

from harness import cli
from harness.run import RunSpec


def _run_dir(tmp_path: Path) -> Path:
    rd = tmp_path / "run"; rd.mkdir()
    (rd / "manifest.json").write_text(json.dumps({"run_id": "x"}), encoding="utf-8")
    return rd


def test_purges_information_in_thin_case(tmp_path, monkeypatch):
    monkeypatch.setenv("DELIVER_THIN_CASE", "1")
    case = tmp_path / "003_case"; case.mkdir()
    (case / "instruction.md").write_text("i", encoding="utf-8")
    (case / "information.md").write_text("secret conventions", encoding="utf-8")
    rd = _run_dir(tmp_path)
    cli._purge_hidden_knowledge_if_thin(RunSpec(case=case, condition="Hidden", model="direct/glm-5.3"), rd, False)
    assert not (case / "information.md").exists()
    assert json.loads((rd / "manifest.json").read_text())["information_purged"] is True


def test_never_touches_a_real_case_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DELIVER_THIN_CASE", "1")
    case = tmp_path / "003_case"; case.mkdir()
    (case / "information.md").write_text("x", encoding="utf-8")
    (case / "gt.json").write_text("{}", encoding="utf-8")            # 有 gt.json = 仓库原件
    cli._purge_hidden_knowledge_if_thin(RunSpec(case=case, condition="Hidden", model="direct/glm-5.3"), _run_dir(tmp_path), False)
    assert (case / "information.md").exists()


def test_noop_without_flag(tmp_path, monkeypatch):
    monkeypatch.delenv("DELIVER_THIN_CASE", raising=False)
    case = tmp_path / "003_case"; case.mkdir()
    (case / "information.md").write_text("x", encoding="utf-8")
    cli._purge_hidden_knowledge_if_thin(RunSpec(case=case, condition="Hidden", model="direct/glm-5.3"), _run_dir(tmp_path), False)
    assert (case / "information.md").exists()
