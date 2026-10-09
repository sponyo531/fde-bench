"""单次运行：准备 workspace → 生成 prompt → 跑 agent → 落盘。

一次 run 的产物固定为一个目录，含可复现所需的全部信息（manifest）、
喂给 agent 的完整 prompt、agent 的原始产物、执行轨迹与用量。
没有 manifest，别人无法判断一个分数是在什么条件下得到的。
"""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from .conditions import ClarifyLimits, build_prompt

_ROOT = Path(__file__).resolve().parent.parent


def _hash_files(paths: list[Path], *, include_metadata: bool = False,
                base: Path | None = None) -> str:
    """对一组文件生成稳定指纹；不存在的文件也进入指纹。"""
    h = hashlib.sha256()
    for path in sorted(paths, key=lambda p: str(p)):
        # Staging can place identical configs under different absolute paths;
        # fingerprints therefore contain logical relative paths only.
        try:
            rel = str(path.relative_to(base)) if base is not None else path.name
        except ValueError:
            rel = path.name
        h.update(rel.encode("utf-8", "replace"))
        if not path.is_file():
            h.update(b"<missing>")
            continue
        try:
            st = path.stat()
            h.update(f"{st.st_size}:{st.st_mtime_ns}".encode()) if include_metadata else None
            h.update(path.read_bytes())
        except OSError:
            h.update(b"<unreadable>")
    return h.hexdigest()[:16]


@lru_cache(maxsize=512)
def _case_fingerprint(case: Path) -> str:
    """记录 case 版本；小文件按内容、大文件按路径/大小/mtime 参与哈希。"""
    h = hashlib.sha256()
    for path in sorted(p for p in case.rglob("*") if p.is_file()):
        try:
            st = path.stat()
            rel = str(path.relative_to(case))
            h.update(f"{rel}\0{st.st_size}\0{st.st_mtime_ns}".encode())
            if st.st_size <= 2 * 1024 * 1024:
                h.update(path.read_bytes())
        except OSError:
            h.update(str(path).encode() + b"<unreadable>")
    return h.hexdigest()[:16]


@lru_cache(maxsize=1)
def _config_fingerprint() -> str:
    return _hash_files([
        _ROOT / "opencode" / "opencode.jsonc",
        _ROOT / "config.toml",
        _ROOT / "pricing.toml",
    ], base=_ROOT)


def _run_signature(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]

# 澄清协议版本。**改动澄清通路时必须 +1**，否则续跑会把旧协议的 run 当成
# 「已完成」静默跳过，新旧结果混进同一张表——而两者根本不可比。
#
#   1  <clarify> 文本标记 + 回灌（自造协议，已废弃）
#   2  脚手架原生提问 tool（opencode 的 question + /question/{id}/reply）
#   3  文本提问检测与考点评分统一为冻结的五模型 judge 投票
#   4  修复协议 3 从未真正生效：detect 的 judge 模型 ID 带 provider 前缀致五票全
#      503、恒判"没在提问"（所有后端的文本澄清全死）；问句改取单份不取并集；
#      原生通路（opencode）拿到回复后回落文本判定，正文提问不再整轮丢弃；
#      judge 截断改保尾。协议 3 的 Interact* run 结果全部不可比，必须重跑。
#   5  max_questions 在文本、OpenCode 与 OpenHands 原生通路统一按问题条数生效；
#      嵌套交付物与最终 artifact 扫描采用同一递归口径，避免误催促额外轮次。
CLARIFY_PROTOCOL = 5


def assert_deepseek_route(model: str | None) -> None:
    """Require the published Chat route for DeepSeek experiment models."""
    m = (model or "").lower()
    if "deepseek" in m and not m.startswith("chat/"):
        raise ValueError(
            f"DeepSeek model must use chat/ route: {model!r}")


@dataclass
class RunSpec:
    case: Path
    condition: str
    model: str
    scaffold: str = "opencode"
    run_index: int = 1
    temperature: float | None = None
    limits: ClarifyLimits = field(default_factory=ClarifyLimits)
    max_turns: int | None = None    # 工具调用轮数上限；None = 不限制（仅 native 后端生效）

    def __post_init__(self) -> None:
        assert_deepseek_route(self.model)

    @property
    def run_id(self) -> str:
        # 模型名含 provider 前缀（direct/glm-5.2），斜杠会被当成路径分隔符
        # 把 run 目录劈成两层，故统一替换为连字符
        model = (self.model or "default").replace("/", "-")
        return (
            f"{self.case.name}__{self.scaffold}__{model}"
            f"__{self.condition}__run{self.run_index}"
        )


def _git_commit(path: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


@lru_cache(maxsize=1)
def _code_fingerprint(root: Path) -> str:
    """harness/ + agents/ 全部源文件的内容哈希。

    git commit 才是首选，但仓库不在 git 下时（本机 FDE-bench 就落在挂载
    边界外，git 发现不到）它恒为 None——manifest 于是完全没有"这个分数是哪份
    代码跑出来的"这一信息，而 run.py 的立意恰恰是"没有 manifest，别人无法判断
    一个分数是在什么条件下得到的"。故补一个不依赖 git 的指纹兜底。

    只收源文件本身，不含结果目录，因此同一份代码跑多少次都是同一个值。
    """
    import hashlib
    h = hashlib.sha256()
    for sub in ("harness", "agents"):
        base = root / sub
        if not base.is_dir():
            continue
        for f in sorted(base.rglob("*")):
            if f.suffix not in (".py", ".md") or "__pycache__" in f.parts:
                continue
            h.update(str(f.relative_to(root)).encode())
            h.update(f.read_bytes())
    for name in ("config.toml", "pricing.toml", "experiments.toml", "opencode/opencode.jsonc"):
        f = root / name
        if f.is_file():
            h.update(name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:12]


def prepare_workspace(spec: RunSpec, run_dir: Path) -> Path:
    """把 case 的 data/ 拷进 workspace。

    只拷 data/ —— instruction 通过 prompt 传入，information.md 与 gt.json
    绝不进 workspace，否则 agent 直接读到答案，三条件对照失去意义。
    """
    ws = run_dir / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    src = spec.case / "data"
    if src.is_dir():
        shutil.copytree(src, ws / "data", dirs_exist_ok=True)
    return ws


def workspace_snapshot(ws: Path) -> dict[str, str]:
    """记录启动前非输入文件的内容指纹，用于审计 Agent 真实产物。"""
    snapshot: dict[str, str] = {}
    if not ws.is_dir():
        return snapshot
    for path in ws.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ws)
        # data/ 是 harness 复制的输入；隐藏文件/目录（.opencode / .oh_clarify.* /
        # .gemini / .codex_last_message…）是脚手架与框架自留物；__pycache__ 与
        # node_modules 是运行副产物。实测 dsh 一次 run 把 __pycache__/*.pyc 记成
        # 交付物、OpenHands 的 .oh_clarify.N.ask 也会被算——artifact_status 误判
        # "有产物"，零产物的 run 被送进抽取器得 0 分，与真正"做出来了但错"混掉。
        if ("data" in rel.parts or "__pycache__" in rel.parts or "node_modules" in rel.parts
                or any(part.startswith(".") for part in rel.parts)):
            continue
        try:
            snapshot[str(rel)] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            continue
    return snapshot


def workspace_produced(ws: Path, before: dict[str, str]) -> list[str]:
    """返回运行后新增或内容发生变化的真实交付文件。"""
    after = workspace_snapshot(ws)
    return sorted(path for path, digest in after.items()
                  if before.get(path) != digest)


def write_manifest(spec: RunSpec, run_dir: Path, prompt: str, extra: dict) -> None:
    # case 标识取自目录名，而报表按它做 case 级聚合（scoring/report.py 的
    # 聚合键、case 分组、bootstrap 分层）。目录名一旦是个通用词，所有 case 的
    # run 会塌进同一个桶，run_id 也会跨 case 重名 —— 而**分数看着完全正常**，
    # 因为评分用的是另一条路径上的真实 case 目录。隔离运行时可能
    # 把 case 拷成名为 "case" 的瘦目录，三个 run 全记成 case="case"。
    # 这里只报警不中断：跑都跑完了，中断只会白扔结果；留痕让人能查。
    if spec.case.name.lower() in {"case", "cases", "workspace", "data", "tmp", "thin"}:
        print(f"  [警告] case 目录名是通用词 {spec.case.name!r} —— 报表按它做 case 级"
              f"聚合，所有 case 会塌成一个桶且 run_id 跨 case 重名。"
              f"请把 case 目录改成真实 case 名（如 003_city_delivery_route_planning）",
              flush=True)
    case_name = spec.case.name
    base = {
        "case": case_name,
        "source_case": case_name,
        "case_variant": "raw" if case_name.endswith("_raw") else "clean",
        "condition": spec.condition,
        "model": spec.model,
        "scaffold": spec.scaffold,
        "run_index": spec.run_index,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
        # 隔离运行时可由外部控制器传入原始 case 指纹；本地运行则直接对
        # spec.case 指纹化。
        "case_fingerprint": os.environ.get("DELIVER_CASE_FINGERPRINT")
        or _case_fingerprint(spec.case),
        "config_fingerprint": _config_fingerprint(),
        "clarify_protocol": CLARIFY_PROTOCOL,
    }
    from .oracle_coverage import load_payload, parse_condition
    if parse_condition(spec.condition) is not None:
        # IDs stay in the private host campaign plan, not this agent-readable file.
        base["oracle_coverage"] = load_payload(
            spec.case, spec.condition, spec.run_index)["metadata"]
    manifest = {
        "run_id": spec.run_id,
        **base,
        "temperature": spec.temperature,
        "max_turns": spec.max_turns,
        "clarify_limits": asdict(spec.limits),
        # 外部控制器可传入代码版本；本地运行时直接探测。
        "bench_commit": os.environ.get("DELIVER_BENCH_COMMIT") or _git_commit(_ROOT),
        # 这条 run 属于哪个实验（experiments.toml 的段名）。矩阵入口设
        # DELIVER_EXPERIMENT；单跑 cli 为 None。同一 (scaffold, model, case, condition)
        # 可能被多个实验复用（E1 与 E2 都有 glm-5.3×Interact-Req），有了它才分得开。
        "experiment": os.environ.get("DELIVER_EXPERIMENT") or None,
        "code_fingerprint": _code_fingerprint(_ROOT),
        "run_signature": _run_signature(base | {"temperature": spec.temperature,
                                                  "clarify_limits": asdict(spec.limits),
                                                  "max_turns": spec.max_turns}),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "prompt_chars": len(prompt),
        **extra,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def setup_run(spec: RunSpec, results_root: Path) -> tuple[Path, Path, str]:
    """建立 run 目录、workspace 与 prompt。返回 (run_dir, workspace, prompt)。"""
    # 外部调度器可以在提交时直接把 results_root 设为
    # ``model/case/condition``。开启此标志后，每个重复运行直接落在
    # ``tryN``，这样运行尚未结束时目录层级也已经是最终布局；默认行为
    # 保持旧的 ``results/<run_id>`` 布局，兼容本地运行和历史任务。
    # Some isolated runners may be restricted to one try directory. In that
    # layout the caller passes the try directory itself as ``--out`` and this
    # flag prevents setup_run from creating a nested ``tryN/tryN`` directory.
    if os.environ.get("DELIVER_FIXED_RUN_DIR") == "1":
        run_dir = results_root
    elif os.environ.get("DELIVER_DIRECT_RUN_LAYOUT") == "1":
        run_dir = results_root / f"try{spec.run_index}"
    else:
        run_dir = results_root / spec.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    ws = prepare_workspace(spec, run_dir)
    prompt = build_prompt(spec.case, spec.condition, ws, spec.limits,
                          run_index=spec.run_index)

    (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    write_manifest(spec, run_dir, prompt, {})
    return run_dir, ws, prompt


def update_manifest(run_dir: Path, **fields) -> None:
    """原子更新 manifest，供运行结束后写入实际生效模型等信息。"""
    path = run_dir / "manifest.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = {}
    data.update(fields)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def effective_models(run_dir: Path) -> list[str]:
    """回读 agent 实际调用的 provider/model。

    spec.model 只是请求值，未显式指定时为 None；scaffold 也可能静默回落到自带
    provider（实测 opencode 配置解析失败时会用 opencode/big-pickle 照常跑完并出分）。
    请求值不能作为"这行数据测的是哪个模型"的证据，故从 scaffold 自己的记录回读。
    """
    dbs = list(run_dir.glob(".agent_home/**/opencode.db"))
    if not dbs:
        return []
    seen: list[str] = []
    try:
        import sqlite3
        conn = sqlite3.connect(str(dbs[0]), timeout=10)
        rows = conn.execute(
            "SELECT DISTINCT json_extract(data,'$.providerID'),"
            " json_extract(data,'$.modelID') FROM message"
            " WHERE json_extract(data,'$.role')='assistant'"
        ).fetchall()
        conn.close()
        seen = [f"{p}/{m}" for p, m in rows if p or m]
    except Exception:
        return []
    return sorted(seen)


def finalize(run_dir: Path, response: str, elapsed_s: float, events: list | None = None,
             status: str | None = None, produced: list[str] | None = None,
             tool_calls_known: bool = True,
             agent_sends: int | None = None) -> None:
    (run_dir / "response.txt").write_text(response, encoding="utf-8")
    usage = {"elapsed_s": round(elapsed_s, 1)}
    models = effective_models(run_dir)
    if models:
        usage["effective_models"] = models
        update_manifest(run_dir, effective_models=models,
                        model_mismatch=bool(models and models != [
                            json.loads((run_dir / "manifest.json").read_text()).get("model")]))
    if status is not None:
        usage["status"] = status
    if produced is not None:
        usage["produced"] = produced
    if agent_sends is not None:
        # 所有 backend 共有的交互单位；不要与底层 LLM call（并非各 CLI 都暴露）混用。
        usage["agent_sends"] = agent_sends
    if events is not None:
        # 后端发不出结构化 tool 事件时记 None：那是"没采到"，不是"没用工具"
        usage["tool_calls"] = (sum(1 for e in events if getattr(e, "type", "") == "tool_call")
                               if tool_calls_known else None)
        # 跨脚手架可比的分类计数：shell / read / edit / search / other。原始 tool_calls
        # 因各家工具集不同只能同脚手架内比；shell 与 edit 两类各家都有明确对应。
        if tool_calls_known:
            from .backends.usage import count_tool_kinds
            usage["tool_calls_by_kind"] = count_tool_kinds(events)
        else:
            usage["tool_calls_by_kind"] = None
        usage["events"] = len(events)
    (run_dir / "usage.json").write_text(
        json.dumps(usage, indent=2, ensure_ascii=False), encoding="utf-8"
    )
