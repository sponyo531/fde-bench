"""统一入口，对标 `harbor run -d <dataset> -a <agent> -m <model> -k <n>`。

    python -m harness -a codex -m gpt-6-astra -c Interact-Req
    python -m harness -a opencode -m direct/glm-5.2 -c Hidden,Interact,Interact-Req,Full -k 3
    python -m harness -e E1                       # 跑预定义实验
    python -m harness --list                      # 看可用脚手架 / 条件 / 实验

也可以直接调用 ``python -m harness.run_cli``；``harness.run`` 是单次 run 的
落盘实现模块，不是 CLI 入口。

设计意图：让「注册一个新 agent 并跑通」是一条命令的事。
新脚手架的接入步骤见 docs/SCAFFOLDS.md，只需两步：
  1. 写一个 backend（多数情况在自己的包里声明一个 _Spec 即可）
  2. 无需登记——registry.py 会自动发现
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _expand_conditions_by_case(case_root: Path, cases: list[str],
                               requested_conditions: list[str]) -> dict[str, list[str]]:
    """按 case 展开条件，尤其是考点数不同的 ``oracle`` 档位。"""
    from .conditions import oracle_conditions

    expanded: dict[str, list[str]] = {}
    for case_name in cases:
        values: list[str] = []
        for condition in requested_conditions:
            values.extend(
                oracle_conditions(case_root / case_name)
                if condition == "oracle" else [condition]
            )
        expanded[case_name] = values
    return expanded


def _list_all() -> None:
    from .backends import available, load_errors
    from .conditions import CONDITIONS
    import tomllib

    print("脚手架 (-a):")
    for s in available():
        print(f"  {s}")
    # 某个 agent 包依赖坏掉时（OpenHands 的独立 venv 最容易），
    # 不该静默消失——否则用户只会看到"unknown scaffold"而不知道真因
    for pkg, why in load_errors().items():
        print(f"  {pkg:22} ✗ 加载失败: {why}")
    print("\n条件 (-c):")
    print(f"  {', '.join(CONDITIONS)}")
    print("  oracle_k<N>            E5：注入前 N 个考点的答案")
    print("\n实验 (-e):")
    with open(_ROOT / "experiments.toml", "rb") as fh:
        cfg = tomllib.load(fh)
    for name, e in cfg["experiments"].items():
        cases = e.get("cases")
        n = len(cases) if isinstance(cases, list) else cases
        print(f"  {name:12} cases={n} conditions={e.get('conditions')}")


def main() -> None:
    p = argparse.ArgumentParser(
        prog="python -m harness",
        description="FDE-Bench：模糊需求 → 验证交付",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("设计意图")[0].strip(),
    )
    p.add_argument("-a", "--agent", help="脚手架，见 --list")
    p.add_argument("-m", "--model", help="模型（含 provider 前缀，如 direct/glm-5.2）")
    p.add_argument("-c", "--conditions", default="Interact-Req",
                   help="信息条件，逗号分隔（默认 Interact-Req）")
    p.add_argument("-d", "--cases", default=None,
                   help="case 名，逗号分隔；留空=全部")
    p.add_argument("-k", "--runs", type=int, default=1, help="每格重复次数")
    p.add_argument("-e", "--experiment", help="跑 experiments.toml 里的预定义实验")
    p.add_argument("-j", "--concurrency", type=int, default=None)
    p.add_argument("-o", "--out", default=None, help="结果目录")
    p.add_argument("--timeout", type=int, default=None, help="单 run 秒数上限")
    p.add_argument("--container", metavar="IMAGE", default=None)
    p.add_argument("--no-eval", action="store_true", help="只跑不评分")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--list", action="store_true", help="列出可用脚手架/条件/实验")
    args = p.parse_args()

    if args.list:
        _list_all()
        return

    if args.experiment:
        from .matrix import load_plan, run_plan
        plan = load_plan(args.experiment)
        if args.concurrency:
            plan.concurrency = args.concurrency
        if args.out:
            plan.out = Path(args.out).resolve()
        asyncio.run(run_plan(plan, container_image=args.container,
                             dry_run=args.dry_run))
        return

    if not args.agent:
        p.error("需要 -a/--agent，或用 -e/--experiment 跑预定义实验；--list 看选项")

    # 单点模式：临时组一个 Plan，与 -e 走同一条执行路径，避免两套逻辑漂移
    from .matrix import Job, Plan, run_plan
    from .config import get
    import tomllib
    with open(_ROOT / "experiments.toml", "rb") as fh:
        defaults = tomllib.load(fh)["defaults"]

    case_root = (_ROOT / defaults["case_root"]).resolve()
    if args.cases:
        cases = [c.strip() for c in args.cases.split(",")]
    else:
        cases = sorted(d.name for d in case_root.iterdir() if d.is_dir())

    model = args.model or get("agent", "model", "") or None
    requested_conditions = [x.strip() for x in args.conditions.split(",")]
    # oracle 档位按 case 的考点数动态展开；不能拿 cases[0] 的档位套到所有 case，
    # 否则多 case 单点命令会在考点数不同的 case 上越界或漏掉档位。
    conditions_by_case = _expand_conditions_by_case(
        case_root, cases, requested_conditions)
    conds = [condition for values in conditions_by_case.values() for condition in values]

    _mt = int(defaults.get("max_turns", 0)) or None
    plan = Plan(
        name=f"{args.agent}-{'-'.join(conds)[:20]}",
        jobs=[Job(case=c, model=model, condition=cond, run_index=i,
                  scaffold=args.agent, max_turns=_mt)
              for c in cases for cond in conditions_by_case[c]
              for i in range(1, args.runs + 1)],
        out=Path(args.out or defaults.get("out") or (_ROOT / "results" / args.agent)).resolve(),
        case_root=case_root,
        timeout_s=args.timeout or int(defaults.get("timeout_s", 43200)),
        concurrency=args.concurrency or int(defaults.get("concurrency", 8)),
        max_turns=_mt,
    )
    asyncio.run(run_plan(plan, container_image=args.container,
                         dry_run=args.dry_run))


if __name__ == "__main__":
    sys.exit(main())
