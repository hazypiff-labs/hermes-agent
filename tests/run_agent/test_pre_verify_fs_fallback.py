"""Opt-in filesystem fallback for the `pre_verify` firing condition.

`agent._turn_file_mutation_paths` only sees write_file/patch results that
were classified as landed, so a turn that edited files through
`execute_code` never fires the hook. `HERMES_PRE_VERIFY_FS_FALLBACK`
substitutes git's view of the dirty worktree; unset, the firing condition is
unchanged.
"""

import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.turn_stop_gates import _pre_verify_fs_fallback_paths
from run_agent import AIAgent


def _response(content="composed report"):
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        model="test/model",
        usage=None,
    )


@pytest.fixture
def dirty_repo(tmp_path, monkeypatch):
    """A git work tree with one untracked edit the tracker never saw."""
    repo = tmp_path / "slot"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "changed.py").write_text("x = 1\n")
    monkeypatch.chdir(repo)
    return repo


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        instance = AIAgent(
            session_id="pre-verify-fs-fallback-test",
            api_key="test-key",
            base_url="https://example.invalid/v1",
            provider="openai-compat",
            model="test/model",
            max_iterations=1,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    instance._cached_system_prompt = "stable test prompt"
    instance._session_db = None
    instance._session_json_enabled = False
    instance.save_trajectories = False
    instance.compression_enabled = False
    instance._cleanup_task_resources = lambda *_a, **_kw: None
    instance._save_trajectory = lambda *_a, **_kw: None
    return instance


def _run_turn(agent, monkeypatch, hook_message="run project tests"):
    """One turn that edits files without populating the mutation tracker."""
    agent._interruptible_api_call = lambda _kwargs: _response()
    agent._handle_max_iterations = MagicMock(return_value="replacement summary")
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")
    continue_message = MagicMock(return_value=hook_message)

    with (
        patch("hermes_cli.plugins.has_hook", side_effect=lambda name: name == "pre_verify"),
        patch("hermes_cli.plugins.get_pre_verify_continue_message", continue_message),
        patch("agent.verify_hooks.max_verify_nudges", return_value=2),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        result = agent.run_conversation("edit changed.py with execute_code")
    return result, continue_message


def test_fs_fallback_fires_hook_when_tracker_is_empty(agent, dirty_repo, monkeypatch):
    monkeypatch.setenv("HERMES_PRE_VERIFY_FS_FALLBACK", "1")

    result, continue_message = _run_turn(agent, monkeypatch)

    assert continue_message.call_count == 1
    assert continue_message.call_args.kwargs["changed_paths"] == ["changed.py"]
    assert agent._pre_verify_nudges == 1


def test_fs_fallback_off_by_default(agent, dirty_repo, monkeypatch):
    monkeypatch.delenv("HERMES_PRE_VERIFY_FS_FALLBACK", raising=False)

    result, continue_message = _run_turn(agent, monkeypatch)

    assert continue_message.call_count == 0
    assert getattr(agent, "_pre_verify_nudges", 0) == 0
    assert all("_pre_verify_synthetic" not in msg for msg in result["messages"])


@pytest.mark.parametrize("value", ["0", "false", "no", "", "off"])
def test_fs_fallback_ignores_falsey_values(dirty_repo, monkeypatch, value):
    monkeypatch.setenv("HERMES_PRE_VERIFY_FS_FALLBACK", value)
    assert _pre_verify_fs_fallback_paths() == []


@pytest.mark.parametrize("value", ["1", "true", "YES", " True "])
def test_fs_fallback_accepts_truthy_values(dirty_repo, monkeypatch, value):
    monkeypatch.setenv("HERMES_PRE_VERIFY_FS_FALLBACK", value)
    assert _pre_verify_fs_fallback_paths() == ["changed.py"]


def test_fs_fallback_outside_a_git_work_tree(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_PRE_VERIFY_FS_FALLBACK", "1")
    plain = tmp_path / "not-a-repo"
    plain.mkdir()
    (plain / "changed.py").write_text("x = 1\n")
    monkeypatch.chdir(plain)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))

    assert _pre_verify_fs_fallback_paths() == []


def test_fs_fallback_maps_renames_quotes_and_caps(dirty_repo, monkeypatch):
    monkeypatch.setenv("HERMES_PRE_VERIFY_FS_FALLBACK", "1")
    porcelain = (
        " M src/mod.py\n"
        "?? untracked.py\n"
        "R  src/old.py -> src/new.py\n"
        '?? "caf\\303\\251.py"\n'
        # sorted last, so the cap drops these and keeps the entries above
        + "".join(f"?? zbulk/f{i:04d}.py\n" for i in range(400))
    )
    completed = subprocess.CompletedProcess([], 0, stdout=porcelain, stderr="")

    with patch("subprocess.run", return_value=completed):
        paths = _pre_verify_fs_fallback_paths()

    assert len(paths) == 200  # capped
    assert "src/new.py" in paths and "src/old.py" not in paths
    assert "café.py" in paths
    assert "src/mod.py" in paths and "untracked.py" in paths


def test_fs_fallback_survives_git_timeout(dirty_repo, monkeypatch):
    monkeypatch.setenv("HERMES_PRE_VERIFY_FS_FALLBACK", "1")

    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("git", 5)):
        assert _pre_verify_fs_fallback_paths() == []
