"""E5 Oracle 条件（按考点注入前 k 条答案）。

这条通路的错误都是静默的：注入条数错了照样能跑出分，只是曲线形状是假的；
泄漏了考点名则等于告诉 agent「这几点正是评分要考的」。故必须有测试。

    python3 -m harness.tests.test_oracle
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness.conditions import (build_prompt, is_oracle,       # noqa: E402
                                oracle_conditions)

GT = {
    "目标": "把网点分配给车辆",
    "产物形式": "routes.json",
    "澄清项·甲": "速度按 70 km/h 算",
    "澄清项·乙": "仓库是重量为空那行",
    "澄清项·丙": "距离用 Haversine",
    "硬约束·HC1": "载重不超过 7125",
}


def _case() -> Path:
    d = Path(tempfile.mkdtemp())
    (d / "gt.json").write_text(json.dumps(GT, ensure_ascii=False), encoding="utf-8")
    (d / "instruction.md").write_text("帮我规划配送路线。", encoding="utf-8")
    (d / "information.md").write_text("## 业务背景\n配送场景\n", encoding="utf-8")
    (d / "data").mkdir()
    (d / "data" / "customers.csv").write_text("id\n1\n", encoding="utf-8")
    return d


def test_is_oracle_parsing():
    assert is_oracle("oracle_k0") == 0
    assert is_oracle("oracle_k12") == 12
    for bad in ("R", "CF", "oracle", "oracle_k", "oracle_kx", ""):
        assert is_oracle(bad) is None, bad
    print("✓ oracle_k<N> 解析")


def test_expands_to_blocker_count():
    """档数 = 考点数 + 1（k 从 0 到 N）。硬约束不算考点。"""
    got = oracle_conditions(_case())
    assert got == ["oracle_k0", "oracle_k1", "oracle_k2", "oracle_k3"], got
    print("✓ 按考点数展开（3 个考点 → 4 档）")


def test_single_cli_expands_oracle_per_case():
    """单点 CLI 多 case 时，每个 case 都按自己的考点数展开。"""
    from harness.run_cli import _expand_conditions_by_case

    root = Path(tempfile.mkdtemp())
    first = root / "first"
    second = root / "second"
    first.mkdir()
    second.mkdir()
    for case, n in ((first, 1), (second, 3)):
        data = {f"澄清项·{i}": f"答案{i}" for i in range(n)}
        (case / "gt.json").write_text(json.dumps(data), encoding="utf-8")

    got = _expand_conditions_by_case(root, ["first", "second"], ["oracle"])
    assert got["first"] == ["oracle_k0", "oracle_k1"]
    assert got["second"] == ["oracle_k0", "oracle_k1", "oracle_k2", "oracle_k3"]
    print("✓ 多 case oracle 按各自考点数展开")


def test_injects_exactly_k_answers():
    """注入条数必须精确等于 k，且按 gt.json 的书写顺序。"""
    c = _case()
    ws = Path("/tmp/ws")
    for k, expect in [(0, []), (1, ["70 km/h"]), (2, ["70 km/h", "重量为空"]),
                      (3, ["70 km/h", "重量为空", "Haversine"])]:
        p = build_prompt(c, f"oracle_k{k}", ws)
        for frag in expect:
            assert frag in p, f"k={k} 缺少 {frag}"
        # k 之后的考点不能出现
        for frag in ["70 km/h", "重量为空", "Haversine"][k:]:
            assert frag not in p, f"k={k} 多注入了 {frag}"
    print("✓ 注入条数精确，顺序稳定")


def test_never_leaks_blocker_labels():
    """考点名是内部标签，泄漏等于告诉 agent 哪几点会被评分。"""
    c = _case()
    for cond in oracle_conditions(c):
        p = build_prompt(c, cond, Path("/tmp/ws"))
        assert "澄清项·" not in p, f"{cond} 泄漏了考点名"
        assert "硬约束" not in p, f"{cond} 泄漏了硬约束标签"
    print("✓ 不泄漏考点名 / 硬约束标签")


def test_k0_offers_no_clarify_channel():
    """oracle 全档都不给提问渠道——它测的是"已知 k 条"而非"能不能问出来"。"""
    p = build_prompt(_case(), "oracle_k0", Path("/tmp/ws"))
    assert "no one available to answer" in p.lower(), p[-300:]
    print("✓ 不提供澄清渠道")


def test_out_of_range_rejected():
    c = _case()
    try:
        build_prompt(c, "oracle_k9", Path("/tmp/ws"))
    except ValueError as e:
        assert "越界" in str(e), e
        print("✓ k 越界报错")
        return
    raise AssertionError("k=9 越界未报错")


def test_missing_blockers_rejected():
    """没标考点的 case 不能跑 oracle——静默跑成 R 会污染曲线。"""
    d = Path(tempfile.mkdtemp())
    (d / "gt.json").write_text(json.dumps({"目标": "x"}), encoding="utf-8")
    (d / "instruction.md").write_text("做点事", encoding="utf-8")
    (d / "data").mkdir()
    try:
        build_prompt(d, "oracle_k0", Path("/tmp/ws"))
    except FileNotFoundError as e:
        assert "考点" in str(e), e
        print("✓ 无考点的 case 显式报错")
        return
    raise AssertionError("无考点未报错")


# ── 条件命名（沿用 Ambig-SWE 谱系）─────────────────────────────────────────

def test_legacy_condition_names_still_work():
    """旧内部名（R/C/CF/F…）必须仍可用。

    历史结果目录名里带旧条件名（results/**/…__CF__run1），断了映射那些
    数据就读不出来、等于白跑。
    """
    from harness.conditions import canonical, CONDITIONS
    pairs = {"R": "Hidden", "C": "Interact", "CF": "Interact-Req",
             "F": "Full", "CF-confirm": "Interact-Conf",
             "F_base": "Full-Base", "F_data": "Full-Data", "F_rule": "Full-Rule"}
    for old, new in pairs.items():
        assert canonical(old) == new, f"{old} -> {canonical(old)}，应为 {new}"
        assert new in CONDITIONS, new
    # 新名与 oracle 原样返回，不被误改
    for keep in ("Hidden", "Full-Rule", "oracle_k3"):
        assert canonical(keep) == keep, keep
    print("✓ 旧条件名映射到新名，新名与 oracle 不变")


def test_legacy_name_builds_same_prompt():
    """用旧名生成的 prompt 必须与新名逐字相同——否则历史数据不可比。"""
    from harness.conditions import build_prompt
    c = _case()
    for old, new in [("CF", "Interact-Req"), ("F_base", "Full-Base")]:
        a = build_prompt(c, old, Path("/tmp/ws"))
        b = build_prompt(c, new, Path("/tmp/ws"))
        assert a == b, f"{old} 与 {new} 生成的 prompt 不同"
    print("✓ 旧名与新名生成同一份 prompt")


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
    print("\nall oracle tests passed")
