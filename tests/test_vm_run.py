"""cloud/vm_run.sh: it parses, and `run` passes queue mode and the
self-delete hook through to the container. Nothing here runs Docker or
reaches a VM."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "cloud" / "vm_run.sh"
GIT_BASH = "C:/Program Files/Git/usr/bin/bash.exe"


def bash() -> str | None:
    return GIT_BASH if os.path.exists(GIT_BASH) else shutil.which("bash")


def run_block() -> str:
    text = SCRIPT.read_text(encoding="utf-8")
    return text[text.index("\nrun)"):text.index("\n    ;;", text.index("\nrun)"))]


def test_vm_run_parses():
    exe = bash()
    if exe is None:
        pytest.skip("no bash on this machine")
    r = subprocess.run([exe, "-n", SCRIPT.as_posix()], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_run_needs_shard_only_outside_queue_mode():
    block = run_block()
    assert 'if [ "${QUEUE:-0}" != "1" ]; then : "${SHARD:?' in block
    assert not re.search(r'^\s*: "\$\{SHARD:\?', block, re.M)


def test_run_passes_queue_and_self_delete_through():
    block = run_block()
    for name in ("QUEUE", "LEASE", "SELF_DELETE", "GN_SERVER_NAME", "SHARD"):
        assert f'-e {name}="${{{name}:-' in block, name
    assert "if [ -f /root/gn_api.py ]; then mounts=(-v /root/gn_api.py:/app/cloud/gn_api.py:ro)" in block
    assert '"${mounts[@]}"' in block
    assert "queue mode" in block and "share $SHARD" in block
