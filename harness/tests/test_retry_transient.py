"""agent send 层的瞬时故障重试：瞬时错误重试到成功、永久错误立刻放弃、留痕。"""

from __future__ import annotations

import pytest

from harness.backends import _retry as R


def test_transient_error_is_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    calls = {"n": 0}
    notes = []

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("claude exit=1: 502 Bad Gateway")
        return "ok"

    assert R.retry_transient(flaky, label="t", on_retry=notes.append) == "ok"
    assert calls["n"] == 3 and len(notes) == 2 and "[retry]" in notes[0]


def test_permanent_error_is_not_retried(monkeypatch):
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def bad():
        calls["n"] += 1
        raise RuntimeError("503 model_not_found: No available")

    with pytest.raises(RuntimeError):
        R.retry_transient(bad, label="t")
    assert calls["n"] == 1


def test_unknown_error_is_not_retried(monkeypatch):
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def bad():
        calls["n"] += 1
        raise RuntimeError("codex exit=2: unexpected argument")

    with pytest.raises(RuntimeError):
        R.retry_transient(bad, label="t")
    assert calls["n"] == 1


def test_gives_up_after_attempts(monkeypatch):
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def always():
        calls["n"] += 1
        raise RuntimeError("Connection reset")

    with pytest.raises(RuntimeError):
        R.retry_transient(always, label="t", attempts=3)
    assert calls["n"] == 3


def test_subprocess_timeout_is_never_retried(monkeypatch):
    """TimeoutExpired 的消息含 "timed out"，会命中瞬时签名——必须按类型拦住。"""
    import subprocess
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def slow():
        calls["n"] += 1
        raise subprocess.TimeoutExpired(cmd="claude", timeout=600)

    with pytest.raises(subprocess.TimeoutExpired):
        R.retry_transient(slow, label="t")
    assert calls["n"] == 1


def test_run_pg_kills_whole_process_group_on_timeout():
    """子进程再起一个握着 stdout 的孙进程：run_pg 超时必须整组杀掉，不能挂在 communicate。"""
    import subprocess, sys, time
    from harness.backends._proc import run_pg
    cmd = [sys.executable, "-c",
           "import subprocess,sys;print('partial',flush=True);"
           "subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
           "import time;time.sleep(60)"]
    t0 = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired) as ei:
        run_pg(cmd, timeout=1.5)
    assert time.monotonic() - t0 < 15, "被孙进程拖住了"
    assert "partial" in (ei.value.output or "")            # 超时前的 stdout 要带回来
