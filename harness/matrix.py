"""实验矩阵编排：把 (case × 模型 × 条件 × run) 展开成任务队列并并发执行。

设计要点
────────
1. **实验在配置里声明，不改代码**——experiments.toml 里每个 [experiments.X]
   段就是一次消融，加实验只需加一段。
2. **断点续跑**：已存在 usage.json 且 status 非 error/degraded 的 run 直接跳过。
   4700 个 run 跑 4 天，中途必然中断，不能从头再来。
3. **失败不阻塞**：单个 run 失败只记录，不中止整个矩阵。
4. **并发上限**：单 run 已占满一个容器，实测中位 25 分钟，靠并发摊平长尾
   （最长 85 分钟）。
"""

from __future__ import annotations

import asyncio
import json
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .run import CLARIFY_PROTOCOL, _code_fingerprint, _config_fingerprint
from .case_catalog import resolve_case_spec, validate_case

_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Job:
    case: str
    model: str
    condition: str
    run_index: int
    scaffold: str = "opencode"
    max_turns: int | None = None

    @property
    def run_id(self) -> str:
        m = (self.model or "default").replace("/", "-")
        return f"{self.case}__{self.scaffold}__{m}__{self.condition}__run{self.run_index}"


@dataclass
class Plan:
    name: str
    jobs: list[Job] = field(default_factory=list)
    out: Path = _ROOT / "results"
    case_root: Path = _ROOT / "case"
    timeout_s: int = 43200
    concurrency: int = 8
    max_turns: int | None = None  # 工具调用轮数上限；None=不限制（仅 native 后端生效）
    image: str | None = None      # 容器镜像模板；None 表示裸跑（无数据隔离）


def load_plan(name: str, cfg_path: Path | None = None) -> Plan:
    """从 experiments.toml 读取一个实验的完整任务列表。"""
    cfg_path = cfg_path or _ROOT / "experiments.toml"
    with cfg_path.open("rb") as fh:
        cfg = tomllib.load(fh)
    exps = cfg.get("experiments", {})
    if name not in exps:
        raise SystemExit(f"未知实验 {name!r}；可选：{sorted(exps)}")
    e = {**cfg.get("defaults", {}), **exps[name]}

    # case 根目录可配；发布包默认使用仓库内已去敏的 case/。
    case_root = Path(e.get("case_root") or (_ROOT / "case")).expanduser()
    if not case_root.is_absolute():
        case_root = (_ROOT / case_root).resolve()

    cases = e.get("cases")
    named_subsets = {k: v for k, v in cfg.get("defaults", {}).items()
                     if isinstance(v, list)}
    try:
        cases = resolve_case_spec(cases, case_root, named_subsets=named_subsets)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if not cases:
        raise SystemExit(f"实验 {name} 没有任何 case")
    missing = {c: validate_case(case_root, c) for c in cases
               if validate_case(case_root, c)}
    if missing:
        sample = list(missing.items())[:5]
        raise SystemExit(f"以下 clean case 文件不完整：{sample}"
                         + (f" 等 {len(missing)} 个" if len(missing) > 5 else ""))
    # oracle 档要按 case 展开：各 case 的考点数不同（3–13 个），
    # 写死 oracle_k0..k5 会在考点少的 case 上越界、在多的 case 上漏档。
    # 在 conditions 里写 "oracle" 即自动展开成该 case 的 k=0…N。
    from .conditions import oracle_conditions
    jobs = []
    for c in cases:
        conds: list[str] = []
        for cond in e["conditions"]:
            if cond == "oracle":
                conds += oracle_conditions(case_root / c)
            else:
                conds.append(cond)
        jobs += [
            Job(case=c, model=m, condition=cond, run_index=i,
                scaffold=e.get("scaffold", "opencode"),
                max_turns=int(e["max_turns"]) if e.get("max_turns") else None)
            for m in e["models"]
            for cond in conds
            for i in range(1, int(e.get("runs", 1)) + 1)
        ]
    return Plan(
        name=name, jobs=jobs, case_root=case_root,
        image=e.get("image"),
        out=Path(e.get("out") or (_ROOT / "results" / name)).resolve(),
        timeout_s=int(e.get("timeout_s", 43200)),
        concurrency=int(e.get("concurrency", 8)),
        max_turns=int(e["max_turns"]) if e.get("max_turns") else None,
    )


def _done(out: Path, job: Job, case_root: Path | None = None) -> bool:
    """已完成、未失败、且协议版本对得上的 run 才可跳过（断点续跑）。

    协议版本必须参与判断：澄清通路从自造的 <clarify> 换成脚手架原生 question
    tool 之后，旧 run 的 usage.json 一切正常（status=ok），续跑会当成"已完成"
    跳过，于是两套协议的结果混进同一张表——而 CF 条件下这两者测的根本不是
    同一件事。manifest 里只有 scaffold=opencode，区分不了。
    """
    run_dir = out / job.run_id
    u = run_dir / "usage.json"
    if not u.is_file():
        return False
    try:
        st = json.loads(u.read_text(encoding="utf-8")).get("status")
    except Exception:
        return False
    if st in (None, "error", "degraded"):
        return False

    mf = run_dir / "manifest.json"
    try:
        proto = json.loads(mf.read_text(encoding="utf-8")).get("clarify_protocol")
    except Exception:
        proto = None          # 字段是本次引入的，缺失即为协议 1 的历史产物
    if proto != CLARIFY_PROTOCOL:
        return False
    try:
        data = json.loads(mf.read_text(encoding="utf-8"))
        # matrix 的 case_root 不在这里可见；优先使用 manifest 中的 case fingerprint
        # 和当前仓库/配置指纹，case 内容变化由 run signature 在新提交时阻止跳过。
        if data.get("case") != job.case or data.get("condition") != job.condition \
                or data.get("model") != job.model or data.get("scaffold") != job.scaffold \
                or int(data.get("run_index", -1)) != int(job.run_index):
            return False
        if data.get("code_fingerprint") != _code_fingerprint(_ROOT):
            return False
        if data.get("config_fingerprint") != _config_fingerprint():
            return False
        if case_root is not None:
            from .run import _case_fingerprint
            if data.get("case_fingerprint") != _case_fingerprint(case_root / job.case):
                return False
        return bool(data.get("run_signature"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


async def run_plan(plan: Plan, container_image: str | None = None,
                   dry_run: bool = False) -> dict:
    """执行矩阵。返回统计。"""
    import os
    os.environ["DELIVER_EXPERIMENT"] = plan.name          # 进 manifest.experiment
    from .cli import _one
    from .conditions import ClarifyLimits
    from .run import RunSpec

    container_image = container_image or plan.image
    todo = [j for j in plan.jobs if not _done(plan.out, j, plan.case_root)]

    # 镜像 preflight：缺失就停，不静默裸跑（裸跑=agent 能读到 gt.json/tests/）
    if container_image and not dry_run and todo:
        from .isolation.container import image_exists
        need = {container_image.replace("{case}", j.case) for j in todo}
        missing = sorted(i for i in need if not image_exists(i))
        if missing:
            raise SystemExit(
                f"缺 {len(missing)}/{len(need)} 个 case 镜像，无法保证数据隔离：\n  "
                + "\n  ".join(missing[:10])
                + (f"\n  ... 另 {len(missing)-10} 个" if len(missing) > 10 else "")
                + "\n先运行 ./build_images.sh <case_name>，或使用 --container fde-bench-runner:v1。"
            )
    skipped = len(plan.jobs) - len(todo)
    print(f"[{plan.name}] 共 {len(plan.jobs)} 个 run，跳过已完成 {skipped}，待跑 {len(todo)}")
    if dry_run:
        for j in todo[:20]:
            print("   ", j.run_id)
        if len(todo) > 20:
            print(f"    ... 另 {len(todo)-20} 个")
        return {"total": len(plan.jobs), "skipped": skipped, "todo": len(todo)}

    sem = asyncio.Semaphore(plan.concurrency)
    stats = {"ok": 0, "failed": 0}
    t0 = time.time()

    async def worker(job: Job, idx: int) -> None:
        async with sem:
            spec = RunSpec(
                case=plan.case_root / job.case,
                condition=job.condition,
                model=job.model or None,
                scaffold=job.scaffold,
                run_index=job.run_index,
                limits=ClarifyLimits(),
                max_turns=job.max_turns,
            )
            ct = None
            if container_image:
                # 与 cli.py 共用 build_spec：bind、资源上限、**凭据透传**一处定义。
                # 之前这里自己拼 ContainerSpec 且没有 env_passthrough，容器里没有 key。
                from .isolation.container import build_spec
                ct = build_spec(container_image.replace("{case}", job.case), job.scaffold)
            try:
                await _one(spec, plan.out, plan.timeout_s, container=ct)
                stats["ok"] += 1
            except Exception as exc:                    # 单点失败不拖垮矩阵
                stats["failed"] += 1
                print(f"  [FAIL] {job.run_id}: {type(exc).__name__}: {exc}")
            done = stats["ok"] + stats["failed"]
            el = time.time() - t0
            eta = el / done * (len(todo) - done) if done else 0
            print(f"  [{done}/{len(todo)}] {job.run_id}  已用 {el/60:.0f}min  ETA {eta/60:.0f}min")

    await asyncio.gather(*(worker(j, i) for i, j in enumerate(todo)))
    print(f"[{plan.name}] 完成 ok={stats['ok']} failed={stats['failed']} "
          f"耗时 {(time.time()-t0)/3600:.1f}h")
    return {**stats, "total": len(plan.jobs), "skipped": skipped}


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="按 experiments.toml 跑实验矩阵")
    p.add_argument("experiment", help="experiments.toml 里的实验名")
    p.add_argument("--container", default=None,
                   help="容器镜像模板，{case} 会被替换，如 fde-bench:{case}")
    p.add_argument("--no-container", action="store_true",
                   help="不用容器（agent 可读到 gt.json/tests，仅调试用）")
    p.add_argument("--dry-run", action="store_true", help="只列任务不执行")
    p.add_argument("--concurrency", type=int, default=None)
    a = p.parse_args()
    plan = load_plan(a.experiment)
    if a.concurrency:
        plan.concurrency = a.concurrency
    img = None if getattr(a, "no_container", False) else a.container
    asyncio.run(run_plan(plan, container_image=img, dry_run=a.dry_run))


if __name__ == "__main__":
    main()
