from pathlib import Path

from harness.case_catalog import discover_cases, resolve_case_spec


def _root(tmp_path: Path) -> Path:
    for name in ("001_demo", "001_demo_clean", "002_other_clean"):
        d = tmp_path / name
        d.mkdir()
        for f in ("gt.json", "instruction.md", "information.md"):
            (d / f).write_text("{}")
        (d / "data").mkdir()
    return tmp_path


def test_discover_cases_filters_raw(tmp_path):
    root = _root(tmp_path)
    assert discover_cases(root) == ["001_demo_clean", "002_other_clean"]


def test_resolve_raw_subset_name_to_clean(tmp_path):
    root = _root(tmp_path)
    assert resolve_case_spec(["001_demo"], root) == ["001_demo_clean"]
