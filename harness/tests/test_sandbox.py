"""宿主机文件沙箱的仓库布局回归测试。"""

from pathlib import Path

from harness.isolation import sandbox


def test_real_vendored_sandbox_module_is_loadable():
    """适配器必须加载仓库真实 vendor，不能指向不存在的 isolation/vendor。"""
    expected = Path(sandbox.__file__).resolve().parents[1] / "vendor" / "sandbox.py"
    assert sandbox._VENDOR == expected
    assert expected.is_file()

    module = sandbox._mod()
    config = module.SandboxConfig(deny_read=["/private"], allow_read=["/public"])
    assert config.deny_read == ["/private"]
    assert config.allow_read == ["/public"]


def test_build_config_with_real_vendor_and_nested_run_paths(tmp_path):
    """真实结果布局下，case/results 被屏蔽而本 run 工作目录仍被放行。"""
    case_dir = tmp_path / "cases" / "001_demo_clean"
    results_root = tmp_path / "deliver-runs"
    run_dir = results_root / "001_demo__opencode__model__Hidden__run1"
    workspace = run_dir / "workspace"
    agent_home = run_dir / ".agent_home"
    for path in (case_dir, workspace, agent_home):
        path.mkdir(parents=True)

    config = sandbox.build_config(case_dir, workspace, results_root, agent_home)

    assert str(case_dir.resolve()) in config.deny_read
    assert str(results_root.resolve()) in config.deny_read
    assert str(workspace.resolve()) in config.allow_read
    assert str(agent_home.resolve()) in config.allow_read
