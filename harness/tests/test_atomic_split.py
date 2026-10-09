"""原子问句切分——precision 的分母就挂在这个函数上。

它是个纯工程细节，出错时**没有任何报错**，只是分数悄悄偏低：
实测历史数据里 78 条切分片段有 6 条是陈述句（7.7%），它们进了 precision
的分母却永远命中不了考点。三个 judge 的分歧（Ask-F1 0.50~0.64）主要
就来自这里，而不是考点定义不清（recall 三家完全一致）。

    python3 -m harness.tests.test_atomic_split
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness.clarify.score import split_atomic                    # noqa: E402


def test_drops_trailing_statement():
    """问号之后的陈述是理由/补充，不是提问——实测真实样本。"""
    got = split_atomic(
        "数据第一行编码为 0，是否就是配送仓库？该点纬度最高（最北），在所有网点上方。")
    assert got == ["数据第一行编码为 0，是否就是配送仓库？"], got
    print("✓ 剔除问号后的陈述句")


def test_drops_reason_clause():
    got = split_atomic("距离用什么方式计算？因为数据中只有经纬度坐标，没有道路网络数据。")
    assert len(got) == 1 and got[0].endswith("？"), got
    print("✓ 剔除理由从句")


def test_keeps_all_real_questions():
    """多个真问句必须全部保留——漏掉会让 recall 虚低。"""
    got = split_atomic("总重量的单位是什么？是公斤吗？每站装卸多久？")
    assert len(got) == 3, got
    assert all(q.endswith("？") for q in got), got
    print("✓ 多个真问句全部保留")


def test_keeps_statement_prefix_within_question():
    """问句内部的铺垫要跟着问句走，不能被切掉——它是判定所需的上下文。"""
    got = split_atomic("网点总重量范围是 4~130，而车辆载重是 7125 公斤。请问单位是什么？")
    assert len(got) == 1, got
    assert "7125" in got[0] and got[0].endswith("？"), got
    print("✓ 问句内的铺垫保留（judge 需要它做判定）")


def test_pure_statement_falls_back():
    """整条都没有问号时保留原文——可能是 agent 用陈述句表达疑问，
    交给 judge 判，总比丢掉信息强（宁可多判，不可漏判）。"""
    got = split_atomic("我需要确认一下速度参数的口径。")
    assert got == ["我需要确认一下速度参数的口径。"], got
    print("✓ 全无问号时回落保留（不静默丢信息）")


def test_english_question_marks():
    got = split_atomic("What speed should I use? The data has no road network.")
    assert got == ["What speed should I use?"], got
    print("✓ 英文问号同样处理")


def test_empty_input():
    assert split_atomic("") == []
    assert split_atomic("   ") == []
    print("✓ 空输入不产生幽灵条目")


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
    print("\nall atomic-split tests passed")
