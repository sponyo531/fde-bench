"""
harness/score.py — 两段式评分：extract → evaluate。

agent 自由交付（格式不限），extractor 负责发现产物并归一化成 evaluator
期望的 schema，evaluator 再独立重算分数。这一分工让「交付格式」不成为
考点噪声，同时保证评分不依赖 agent 的自述。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


class ScoreError(RuntimeError):
    pass


def _timeout() -> int:
    from ..config import get
    return int(get("score", "timeout_s", 900))


def _run(cmd: list[str], cwd: Path | None = None, timeout: int | None = None) -> subprocess.CompletedProcess:
    timeout = timeout or _timeout()
    env = dict(os.environ)
    # extractor 也是评分链路一环，模型应锁定版本（见 config.toml [extractor]）
    from ..config import get
    if _m := get("extractor", "model", ""):
        env["EXTRACTOR_MODEL"] = _m
    return subprocess.run(
        cmd, cwd=str(cwd) if cwd else None, env=env,
        capture_output=True, text=True, timeout=timeout,
    )


_PLAN_FILE = re.compile(r"^PLAN_FILE\s*=\s*[\"']([^\"']+)[\"']", re.M)


def _target_filename(case_dir: Path) -> str | None:
    """evaluator 期望的产物文件名（PLAN_FILE）。"""
    ev = case_dir / "tests" / "evaluator.py"
    if not ev.is_file():
        return None
    m = _PLAN_FILE.search(ev.read_text(encoding="utf-8", errors="replace"))
    return m.group(1) if m else None


def _try_passthrough(case_dir: Path, workspace: Path, out_dir: Path) -> dict | None:
    """agent 已按目标文件名与 schema 交付时，原样复制，不过 LLM。

    实测 extractor 在这种情形下依然会把内容重写一遍：四个条件里三个改了分组，
    notes 却写着 "already conforms to the required schema"（见 conservation.py
    的模块注释）。唯一可靠的办法是根本不给它改的机会——能直接用就直接用。

    判据刻意保守：文件名必须与 PLAN_FILE 完全一致，且 evaluator 能给出
    validity_score > 0。凡有一点不符就回落到 LLM extractor。
    """
    target = _target_filename(case_dir)
    if not target:
        return None
    src = workspace / target
    if not src.is_file():
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / target
    shutil.copy2(src, dst)
    try:
        got = evaluate(case_dir, out_dir)
    except Exception:
        dst.unlink(missing_ok=True)
        return None
    if not got.get("validity_score"):
        # 原文件不合格：可能是格式问题（该交给 extractor 修），
        # 也可能是解本身违约（extractor 会判 partial）。都不在此处下结论。
        dst.unlink(missing_ok=True)
        return None

    return {
        "status": "success",
        "notes": f"{target} 已符合目标 schema，直接复制（未经 LLM 改写）",
        "files_found": [target],
        "files_normalized": [target],
        "passthrough": True,
    }


def extract(case_dir: Path, workspace: Path, out_dir: Path, timeout: int | None = None) -> dict:
    """把 workspace 产物归一化到 out_dir。

    先试直通（源已是目标 schema），不行再调 case 自带的 extractor_agent。
    """
    if (direct := _try_passthrough(case_dir, workspace, out_dir)) is not None:
        return direct

    extractor = case_dir / "tests" / "extractor_agent.py"
    if not extractor.exists():
        raise ScoreError(f"缺少 extractor: {extractor}")

    out_dir.mkdir(parents=True, exist_ok=True)
    proc = _run([
        sys.executable, str(extractor),
        "--evaluator", str(case_dir / "tests" / "evaluator.py"),
        "--workspace", str(workspace),
        "--output", str(out_dir),
    ], timeout=timeout)

    # extractor 的 JSON 状态可能跨多行输出，取 stdout 中最后一个完整 JSON 对象
    text = proc.stdout.strip()
    start = text.find("{")
    while start != -1:
        try:
            payload = json.loads(text[start:])
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
        start = text.find("{", start + 1)
    return {"status": "failed", "stdout": text[-1500:], "stderr": proc.stderr[-1500:]}


def evaluate(case_dir: Path, normalized: Path, timeout: int | None = None) -> dict:
    """调 case 自带的 evaluator，对归一化产物独立重算分数。"""
    evaluator = case_dir / "tests" / "evaluator.py"
    baseline = case_dir / "tests" / "baseline"

    # 原实现硬写死 evaluate(data_dir=, baseline_dir=, submission_dir=)，
    # 仅适用于少数样例 case。不同 case 的签名可能不同（有的没有 baseline_dir，
    # 有的把提交侧参数叫 plan_path/file_path），
    # 按原写法一律 TypeError → ScoreError「evaluator 无输出」，根本跑不出分。
    # 现按 inspect.signature 反射，只传该 evaluator 真正接受的形参；
    # 带 **kwargs 的照旧全传。
    code = (
        "import json,sys,inspect,importlib.util as u\n"
        "from pathlib import Path as _P\n"
        f"s=u.spec_from_file_location('ev',r'{evaluator}')\n"
        "m=u.module_from_spec(s); sys.modules[s.name]=m; s.loader.exec_module(m)\n"
        "fn=getattr(m,'evaluate',None) or getattr(m,'compute_metrics')\n"
        f"avail={{'data_dir':_P(r'{case_dir / 'data'}'),"
        f" 'baseline_dir':_P(r'{baseline}'), 'submission_dir':_P(r'{normalized}')}}\n"
        "sig=inspect.signature(fn); params=sig.parameters\n"
        "has_var_kw=any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())\n"
        "# 提交侧形参的别名：不同 case 叫法不一(plan_path/file_path/submission)\n"
        "SUB_ALIASES=('submission_dir','submission','plan_path','file_path','plan_file','sub_dir')\n"
        "# ...其中一部分要的是解文件本身，不是它所在的目录。名字已经说清楚了\n"
        "# (plan_path / file_path / plan_file)，可 2026-08-25 那版一律传目录，\n"
        "# 于是这类 evaluator 在 open() 上抛 IsADirectoryError，分数记 0——看起来\n"
        "# 像 agent 交了废解，实际是 harness 递错了参数。\n"
        "FILE_ALIASES=('plan_path','file_path','plan_file')\n"
        "DATA_ALIASES=('data_dir','data','dataset_dir')\n"
        "BASE_ALIASES=('baseline_dir','baseline')\n"
        "def _sub_file(d):\n"
        "    # 先按 case 自己声明的产物名找；submission_schema.json 是权威契约。\n"
        "    try:\n"
        f"        sc=json.loads(_P(r'{case_dir / 'tests' / 'submission_schema.json'}')"
        ".read_text(encoding='utf-8'))\n"
        "        names=[s.get('name') for s in (sc.get('files') or []) if s.get('name')]\n"
        "        for n in names:\n"
        "            if (d/n).is_file(): return d/n\n"
        "    except Exception: names=[]\n"
        "    # 契约缺失/对不上时，目录里只有一个文件就是它。\n"
        "    fs=[p for p in sorted(d.iterdir()) if p.is_file()] if d.is_dir() else []\n"
        "    if len(fs)==1: return fs[0]\n"
        "    # 仍无从判断时，返回**契约声明的那个路径**（哪怕不存在），不要返回目录。\n"
        "    # 传目录会让 evaluator 在 open() 上抛 IsADirectoryError，被记成 exception；\n"
        "    # 传一个缺失的文件路径，它才会走自己的「文件不存在」分支，报出\n"
        "    # 「缺 solution.json」这种可读的 fatal——同样是 0 分，但 0 分的原因是真的。\n"
        "    return d/(names[0] if names else 'solution.json')\n"
        "kw={}\n"
        "for name in params:\n"
        "    if name in FILE_ALIASES: kw[name]=_sub_file(avail['submission_dir'])\n"
        "    elif name in SUB_ALIASES: kw[name]=avail['submission_dir']\n"
        "    elif name in DATA_ALIASES: kw[name]=avail['data_dir']\n"
        "    elif name in BASE_ALIASES: kw[name]=avail['baseline_dir']\n"
        "if has_var_kw: kw=dict(avail, **kw)\n"
        "# 必填形参没被覆盖到就报出来，别静默传错\n"
        "missing=[n for n,p in params.items()\n"
        "         if p.default is inspect.Parameter.empty\n"
        "         and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)\n"
        "         and n not in kw]\n"
        "if missing: raise TypeError('evaluate 有无法映射的必填形参: '+repr(missing)+' 签名='+str(sig))\n"
        "r=fn(**kw)\n"
        "print('__SCORE__'+json.dumps(r,ensure_ascii=False,default=str))\n"
    )
    proc = _run([sys.executable, "-c", code], timeout=timeout)
    for line in proc.stdout.splitlines():
        if line.startswith("__SCORE__"):
            return json.loads(line[len("__SCORE__"):])
    raise ScoreError(f"evaluator 无输出\nstdout: {proc.stdout[-800:]}\nstderr: {proc.stderr[-800:]}")


def score_run(case_dir: Path, run_dir: Path, timeout: int | None = None,
              *, reuse_completed: bool = True) -> dict:
    """对单次 run 打分，结果写入 run_dir/result.json。

    默认走三模型投票抽取（triple.score_run，opencode 三模型）——抗单抽取器盲区。
    可用 config [score].extractor 切换回单模型（extractor_agent / CC SDK）：
        单模型快、依赖少，用于快速回归或对照；三模型慢但稳。

    两种入口的内部都做守恒校验（单模型是 token 多重集，三模型是 role 守恒）。
    """
    from ..config import get
    mode = (get("score", "extractor", "") or "").strip().lower()
    if mode == "single":
        return _score_run_single(case_dir, run_dir, timeout)
    return _score_run_triple(
        case_dir, run_dir, timeout,
        reuse_completed=reuse_completed,
    )


def _score_run_single(case_dir: Path, run_dir: Path, timeout: int | None = None) -> dict:
    """单模型评分：extractor_agent（CC SDK）→ token 守恒 → evaluator。

    保留作对照/回归；默认评分走 _score_run_triple。
    """
    workspace = run_dir / "workspace"
    normalized = run_dir / "normalized"

    # 重跑前清空：残留的旧产物会被当成本次抽取结果。实测一次重跑读到了
    # 上一版 extractor（会重排解）留下的 routes.json，守恒校验因此判 tampered，
    # 而 extractor 这次其实压根没跑——排查时看到 extract=failed + notes=None
    # 完全对不上号。
    if normalized.exists():
        shutil.rmtree(normalized)

    result: dict = {"case": case_dir.name}
    try:
        ex = extract(case_dir, workspace, normalized, timeout)
        result["extract"] = ex

        # 硬校验：extractor 只能改表示、不能改事实。prompt 白名单是软约束，
        # 这里比对 normalized 与 workspace 的 token 多重集——凡 normalized 中
        # 出现而 agent 从未写过的带数字 token，即视为凭空造内容（补行/填值/重算）。
        from .conservation import check_conservation, verdict
        cons = check_conservation(workspace, normalized)
        result["conservation"] = cons
        status = verdict(ex.get("status", "failed"), cons)

        if status == "tampered":
            # 两类违规分开报：文案说错原因会把排查引向错误方向
            # （实测重排被报成"凭空出现 0 个 token"，自相矛盾）
            if cons.get("regrouped"):
                src = (cons.get("structure") or {}).get("sources_compared") or []
                result["error"] = (
                    f"extractor 重新安排了 agent 的解（分组与源不一致，"
                    f"对比源 {src}）"
                )
            else:
                result["error"] = (
                    f"extractor 凭空造了内容（{cons['invented_count']} 个 token "
                    f"在 agent 产物中从未出现）"
                )
            result["combined_score"] = 0.0
        elif status != "success":
            result["error"] = f"extract {status}"
            # An extractor failure is not a benchmark score.  Keep the score
            # null so downstream aggregation cannot mistake it for a real 0.
            result["combined_score"] = None
            result["scoring_status"] = "failed"
        else:
            result.update(evaluate(case_dir, normalized, timeout))
    except Exception as e:  # noqa: BLE001 — 评分失败不应中断批量运行
        result["error"] = f"{type(e).__name__}: {e}"
        result["combined_score"] = None
        result["scoring_status"] = "failed"

    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def _score_run_triple(case_dir: Path, run_dir: Path, timeout: int | None = None,
                      *, reuse_completed: bool = True) -> dict:
    """三模型投票评分：opencode 三模型抽取 → role 守恒 → evaluator → 投票。

    委托给 triple.score_run（同一 run 目录，写 score_{tag}.json / result.json）。
    """
    from .triple import score_run as triple_score_run
    try:
        return triple_score_run(
            case_dir, run_dir, timeout,
            reuse_completed=reuse_completed,
        )
    except Exception as e:   # noqa: BLE001 — 评分失败不中断批量
        # A scorer crash is not a benchmark score.  Persist a null-score ledger
        # when no result exists, while never overwriting an earlier valid result
        # (important for a failed re-score attempt).
        result = {
            "case": case_dir.name,
            "final_score": None,
            "overall_score": None,
            "combined_score": None,
            "scoring_status": "failed",
            "error_info": {
                "category": "scorer_exception",
                "message": f"{type(e).__name__}: {e}",
            },
        }
        path = run_dir / "result.json"
        if not path.exists():
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        return result


def summarize(result: dict) -> str:
    if result.get("error"):
        return f"评分失败: {result['error']}"
    v = result.get("validity_score", result.get("validity"))
    q = result.get("quality_score")
    c = result.get("combined_score", result.get("overall_score"))
    parts = [f"combined={c}"]
    if v is not None:
        parts.append(f"validity={v}")
    if q is not None:
        parts.append(f"quality={q}")
    return "  ".join(parts)
