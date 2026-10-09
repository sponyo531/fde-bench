"""workspace 产物审计：输入文件不应被误记为 Agent 交付物。"""

from pathlib import Path

from harness.run import workspace_produced, workspace_snapshot


def test_snapshot_excludes_input_and_runtime_files(tmp_path: Path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "input.csv").write_text("x", encoding="utf-8")
    (tmp_path / ".opencode").mkdir()
    (tmp_path / ".opencode" / "state.db").write_text("x", encoding="utf-8")
    (tmp_path / "existing.txt").write_text("before", encoding="utf-8")
    before = workspace_snapshot(tmp_path)

    (tmp_path / "data" / "input.csv").write_text("changed", encoding="utf-8")
    (tmp_path / ".opencode" / "state.db").write_text("changed", encoding="utf-8")
    (tmp_path / "existing.txt").write_text("after", encoding="utf-8")
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "result.csv").write_text("ok", encoding="utf-8")

    assert workspace_produced(tmp_path, before) == ["existing.txt", "output/result.csv"]

