"""考点泄漏检查的判据。

这个脚本要在 58 个 case 上跑，误报会淹没真问题、漏报会放走真泄漏，
两个方向都得钉住。用例全部来自实测踩过的坑。

    python3 -m harness.tests.test_leak_check
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness.tools.leak_check import check_case, significant_numbers   # noqa: E402


def _case(gt: dict, instruction: str) -> Path:
    d = Path(tempfile.mkdtemp())
    (d / "gt.json").write_text(json.dumps(gt, ensure_ascii=False), encoding="utf-8")
    (d / "instruction.md").write_text(instruction, encoding="utf-8")
    return d


def test_catches_real_leak():
    """03 实测：gt 说「数据里没给这套速度参数，agent 需澄清」，
    而 instruction 白写了 70/35/120。"""
    c = _case(
        {"澄清项·时长速度分段": "出仓段按 70 km/h，网点间按 35 km/h，每站加 120 秒装卸。"},
        "干线段大概 70 km/h、网点之间本地路 35 km/h、每站装卸时间是 120 秒。",
    )
    found = check_case(c)
    assert found, "真泄漏没抓到"
    assert set(found[0]["leaked"]) == {"70", "35", "120"}, found
    print("✓ 抓到真泄漏（70/35/120）")


def test_ignores_substring_match():
    """'3' 不该命中 '43%'——子串匹配曾让产品牌号整片误报。"""
    c = _case({"澄清项·换算": "吨车与吨节换算按 3 计。"}, "43% 仓最多存 1500 吨。")
    assert check_case(c) == [], check_case(c)
    print("✓ 子串不算命中（3 ≠ 43）")


def test_ignores_dates_and_horizons():
    """客户交代活儿时会说时间范围，那是任务设定不是泄漏（49_port 曾整片误报）。"""
    c = _case(
        {"澄清项·筛船": "挑出计划离泊落在 2026-06-15 到 06-18 这 72 小时内的船。"},
        "排一下接下来 72 小时的港口作业，从 2026-06-15 00:00 算起。",
    )
    assert check_case(c) == [], check_case(c)
    print("✓ 日期与时间跨度不算泄漏")


def test_ignores_hard_constraints():
    """硬约束允许写进业务需求，只查考点。"""
    c = _case({"硬约束·HC1": "每辆车总重量不超过 7125 kg。"}, "每辆最大载重 7125 公斤。")
    assert check_case(c) == [], check_case(c)
    print("✓ 硬约束不计入")


def test_ignores_ordinals():
    c = _case({"澄清项·步骤": "第 1 步做 A，第 2 步做 B。"}, "分 1 步 2 步来做。")
    assert check_case(c) == [], check_case(c)
    print("✓ 纯序号不算卡点值")


def test_significant_numbers_keeps_units_and_long_values():
    got = significant_numbers("载重 7125 kg，每站 120 秒，速度 70 km/h，第 3 项")
    assert {"7125", "120", "70"} <= got, got
    assert "3" not in got, got
    print("✓ 带单位 / 三位以上数值被保留，裸序号被丢弃")


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
    print("\nall leak-check tests passed")
