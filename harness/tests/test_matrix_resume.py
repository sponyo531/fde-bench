"""断点续跑的跳过判据。

这条逻辑没有报错路径——判错了就是静默把旧结果混进新表，跑完几百个 run
才会在数据里发现异常。故必须有测试盯着。

    python3 -m harness.tests.test_matrix_resume
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness.matrix import Job, _done                    # noqa: E402
from harness.run import CLARIFY_PROTOCOL, _code_fingerprint, _config_fingerprint  # noqa: E402


def _mk(root: Path, run_id: str, *, status="ok", protocol=CLARIFY_PROTOCOL,
        manifest=True) -> None:
    d = root / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "usage.json").write_text(json.dumps({"status": status}), encoding="utf-8")
    if manifest:
        case, scaffold, model, cond, run = run_id.split("__")
        body = {"run_id": run_id, "case": case, "condition": cond,
                "model": model.replace("-", "/", 1), "scaffold": scaffold,
                "run_index": int(run.removeprefix("run")),
                "code_fingerprint": _code_fingerprint(Path(__file__).resolve().parents[2]),
                "config_fingerprint": _config_fingerprint(), "run_signature": "fixture"}
        if protocol is not None:
            body["clarify_protocol"] = protocol
        (d / "manifest.json").write_text(json.dumps(body), encoding="utf-8")


def _job(run_id: str) -> Job:
    # Job.case 是 case **名**（str），不是路径——传 Path 会把 "case/" 也拼进
    # run_id，与 cli.py 的 RunSpec.run_id 对不上
    case, scaffold, model, cond, run = run_id.split("__")
    return Job(case=case, condition=cond,
               model=model.replace("-", "/", 1), scaffold=scaffold,
               run_index=int(run.removeprefix("run")))


RID = "002_city_delivery_route_planning__opencode__direct-glm-5.2__Interact-Req__run1"


def test_skips_completed_current_protocol():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _mk(root, RID)
        assert _done(root, _job(RID)) is True
    print("✓ 当前协议的完成 run → 跳过")


def test_reruns_old_protocol():
    """协议 1（<clarify> 自造标记）的 run 必须重跑，不能当成已完成。"""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _mk(root, RID, protocol=1)
        assert _done(root, _job(RID)) is False
    print("✓ 旧协议的完成 run → 重跑")


def test_reruns_manifest_without_protocol_field():
    """字段是后加的：老结果里没有这个键，同样视为旧协议。"""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _mk(root, RID, protocol=None)
        assert _done(root, _job(RID)) is False
    print("✓ 无 clarify_protocol 字段 → 视为旧协议，重跑")


def test_reruns_failed_statuses():
    for st in ("error", "degraded"):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk(root, RID, status=st)
            assert _done(root, _job(RID)) is False, st
    print("✓ error / degraded → 重跑")


def test_reruns_when_missing_files():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        assert _done(root, _job(RID)) is False            # 目录都没有
        _mk(root, RID, manifest=False)                    # 只有 usage.json
        assert _done(root, _job(RID)) is False
    print("✓ 缺 usage.json / manifest.json → 重跑")


def test_run_id_matches_between_matrix_and_cli():
    """matrix 的 Job 与 cli 的 RunSpec 必须算出同一个 run_id。

    两处各自拼串，一旦漂移，_done 会永远查不到已完成的目录 → 每次续跑都
    从头再跑一遍，且不报任何错。
    """
    from harness.run import RunSpec
    from harness.conditions import ClarifyLimits
    name = "002_city_delivery_route_planning"
    for cond in ("Hidden", "Interact", "Interact-Req", "Full",
                 "Interact-Conf", "Full-Base"):
        job = Job(case=name, model="direct/glm-5.2", condition=cond,
                  scaffold="opencode", run_index=2)
        spec = RunSpec(case=Path("cases") / name, condition=cond,
                       model="direct/glm-5.2", scaffold="opencode",
                       run_index=2, limits=ClarifyLimits())
        assert job.run_id == spec.run_id, f"{cond}: {job.run_id} != {spec.run_id}"
    print("✓ matrix 与 cli 的 run_id 一致")


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
    print("\nall matrix-resume tests passed")
