"""Safety contract for the code-only local/3090 sync helper."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNC_SCRIPT = PROJECT_ROOT / "scripts" / "local" / "sync_code_only.sh"
SYNC_FILTER = PROJECT_ROOT / "scripts" / "local" / "sync_code_only.filter"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def test_sync_policy_whitelists_code_and_excludes_runtime_artifacts():
    policy = SYNC_FILTER.read_text(encoding="utf-8")

    for allowed in (
        "+ /backend/***",
        "+ /frontend/***",
        "+ /scripts/***",
        "+ /simulation/***",
        "+ /tests/***",
        "+ /config/***",
        "+ /docs/***",
    ):
        assert allowed in policy

    for blocked in (
        "- /data/***",
        "- /logs/***",
        "- /outputs/***",
        "- /runs/***",
        "- /models/***",
        "- /checkpoints/***",
        "- /frontend/node_modules/***",
        "- /frontend/dist/***",
        "- /simulation/third_party/***",
        "- *.log",
        "- *.sqlite*",
        "- *.pt",
        "- *.safetensors",
    ):
        assert blocked in policy

    assert policy.index("+ /.env.example") < policy.index("- /.env.*")
    assert policy.rstrip().endswith("- *")


def test_check_is_dry_run_and_uses_the_code_filter(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    rsync_log = tmp_path / "rsync.args"
    _write_executable(
        fake_bin / "rsync",
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$SYNC_TEST_RSYNC_LOG"\n',
    )

    local_root = tmp_path / "project"
    local_root.mkdir()
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "SYNC_TEST_RSYNC_LOG": str(rsync_log),
        "SPECTRUMCLAW_SYNC_LOCAL_ROOT": str(local_root),
        "SPECTRUMCLAW_SYNC_REMOTE_HOST": "fake-3090",
        "SPECTRUMCLAW_SYNC_REMOTE_ROOT": "/srv/SpectrumClaw",
    }

    result = subprocess.run(
        ["bash", str(SYNC_SCRIPT), "check", "pull"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    arguments = rsync_log.read_text(encoding="utf-8").splitlines()
    assert "--dry-run" in arguments
    assert "--itemize-changes" in arguments
    assert f"--filter=merge {SYNC_FILTER}" in arguments
    assert "fake-3090:/srv/SpectrumClaw/" in arguments
    assert str(local_root) + "/" in arguments


def test_apply_pull_refuses_a_dirty_local_target(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    rsync_log = tmp_path / "rsync.args"
    _write_executable(
        fake_bin / "rsync",
        '#!/usr/bin/env bash\nprintf "called\\n" > "$SYNC_TEST_RSYNC_LOG"\n',
    )

    local_root = tmp_path / "project"
    local_root.mkdir()
    subprocess.run(["git", "init", "-q", str(local_root)], check=True)
    (local_root / "local-change.py").write_text("dirty = True\n", encoding="utf-8")
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "SYNC_TEST_RSYNC_LOG": str(rsync_log),
        "SPECTRUMCLAW_SYNC_LOCAL_ROOT": str(local_root),
        "SPECTRUMCLAW_SYNC_REMOTE_HOST": "fake-3090",
        "SPECTRUMCLAW_SYNC_REMOTE_ROOT": "/srv/SpectrumClaw",
    }

    result = subprocess.run(
        ["bash", str(SYNC_SCRIPT), "apply", "pull"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "本地工作区存在未提交改动" in result.stderr
    assert not rsync_log.exists()
