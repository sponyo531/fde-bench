"""容器规格：两处构造点共用，凭据必须透传，脚手架名要映射到真实二进制名。"""

from __future__ import annotations

from harness.isolation import container as C


def test_env_passthrough_includes_the_key_opencode_reads():
    spec = C.build_spec("img", "opencode")
    assert "DELIVER_AGENT_API_KEY" in spec.env_passthrough
    assert "DELIVER_RESPONSES_API_KEY" in spec.env_passthrough
    assert "DELIVER_DIRECT_API_KEY" in spec.env_passthrough
    assert "FDE_OPENCODE_CONFIG" in spec.env_passthrough
    assert "FDE_EXTRACTOR_OPENCODE_CONFIG" in spec.env_passthrough
    assert "DELIVER_ANSWERER_TOKEN" in spec.env_passthrough


def test_agent_binary_maps_scaffold_names(monkeypatch):
    seen = []
    monkeypatch.setattr(C.shutil, "which", lambda n: seen.append(n) or f"/bin/{n}")
    for scaffold, binary in [("claude-code", "claude"), ("deepseek-harness", "dsh"),
                             ("opencode-run", "opencode"), ("codex", "codex")]:
        p = C.agent_binary(scaffold)
        assert p is not None and p.name == binary, (scaffold, p)
    assert "claude-code" not in seen                       # 不再拿脚手架名去 which


def test_docker_cmd_forwards_only_present_env(tmp_path):
    spec = C.build_spec("img", "opencode")
    env = {"DELIVER_AGENT_API_KEY": "k", "UNRELATED": "x", "XDG_DATA_HOME": str(tmp_path)}
    cmd = C.wrap_command(["opencode", "serve"], spec, tmp_path, env)
    joined = " ".join(cmd)
    assert "-e DELIVER_AGENT_API_KEY=k" in joined and "UNRELATED" not in joined
