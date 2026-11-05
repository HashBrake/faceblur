"""The privacy image: cloud/Dockerfile's `privacy` stage, its entry script,
the weights' checksums and the healthcheck's selfcheck. Static and fast:
nothing here runs Docker."""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = ROOT / "cloud" / "Dockerfile"
ENTRY = ROOT / "cloud" / "privacy_entry.sh"
SUMS = ROOT / "models" / "SHA256SUMS"
GIT_BASH = "C:/Program Files/Git/usr/bin/bash.exe"


def stages() -> dict[str, str]:
    """Each named stage's text, from its FROM line to the next."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    parts = re.split(r"(?m)^(?=FROM )", text)[1:]
    return {re.match(r"FROM \S+ AS (\S+)", p).group(1): p for p in parts}


def test_privacy_is_the_last_stage_so_the_default_target():
    names = list(stages())
    assert names[-1] == "privacy"
    assert "notebook" in names and "base" in names


def test_privacy_stage_runs_as_its_own_user_offline_with_a_healthcheck():
    stage = stages()["privacy"]
    assert stage.startswith("FROM base AS privacy")
    assert "useradd --create-home --uid 10001 faceblur" in stage
    assert re.search(r"(?m)^USER faceblur$", stage)
    assert "HF_HUB_OFFLINE=1" in stage and "TRANSFORMERS_OFFLINE=1" in stage
    health = stage[stage.index("HEALTHCHECK"):]
    health = health[:health.index("\nCMD") if "\nCMD" in health else None]
    assert '"faceblur.h5", "selfcheck"' in health
    assert re.search(r'(?m)^CMD \["bash", "/app/cloud/privacy_entry.sh"\]$', stage)
    assert 'VOLUME ["/in", "/out"]' in stage


def test_privacy_image_carries_no_notebook_or_login_server():
    # The privacy image is base plus its own stage, so neither may carry them.
    for name in ("base", "privacy"):
        # Instructions only: the comments may name what the image leaves out.
        text = "\n".join(l for l in stages()[name].lower().splitlines()
                         if not l.lstrip().startswith("#"))
        for word in ("jupyter", "openssh", "sshd", "rclone", "expose"):
            assert word not in text, f"{word} in the {name} stage"


def test_image_carries_the_audit_schema_verify_needs():
    """The entry script verifies every audit, and verify reads this file."""
    from faceblur import audit

    assert audit.SCHEMA_PATH == ROOT / "docs" / "audit_schema.json"
    assert "COPY docs/audit_schema.json docs/audit_schema.json" in stages()["base"]


def test_notebook_stage_keeps_the_greennode_behaviour():
    stage = stages()["notebook"]
    assert stage.startswith("FROM base AS notebook")
    assert "jupyterlab==" in stage and "openssh-server" in stage
    assert "EXPOSE 8888 22" in stage
    assert 'CMD ["bash", "/app/cloud/entry.sh"]' in stage


def test_dockerignore_keeps_secrets_and_footage_out():
    lines = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    for allowed in ("!models/SHA256SUMS", "!models/*.txt", "!cloud", "!docs/audit_schema.json"):
        assert allowed in lines, allowed
    for excluded in ("**/*.env", "**/cloud/mano", "**/footage", "**/out", "**/eval/cache", "**/.venv*",
                     "**/*_blurred*"):
        assert excluded in lines, excluded
        # After every allow, or an allow later in the file would let it back in.
        assert lines.index(excluded) > max(i for i, l in enumerate(lines) if l.startswith("!"))


def test_entry_script_checks_the_weights_and_reads_in():
    text = ENTRY.read_text(encoding="utf-8")
    assert "set -euo pipefail" in text
    assert "sha256sum -c models/SHA256SUMS" in text
    # The check comes before any run.
    assert text.index("sha256sum -c models/SHA256SUMS") < text.index("faceblur.h5 run")
    assert "/in/*.h5" in text and "/in/*.hdf5" in text
    assert "exit 2" in text
    assert "faceblur.h5 verify" in text
    assert "${FACEBLUR_ARGS:-}" in text and '"${FACEBLUR_ARGS' not in text


def test_entry_script_and_sums_have_lf_line_endings():
    for path in (ENTRY, SUMS, DOCKERFILE):
        assert b"\r\n" not in path.read_bytes(), path


def test_entry_script_parses():
    exe = GIT_BASH if os.path.exists(GIT_BASH) else shutil.which("bash")
    if exe is None:
        pytest.skip("no bash on this machine")
    r = subprocess.run([exe, "-n", ENTRY.as_posix()], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_sums_check_like_sha256sum_c():
    """What `sha256sum -c models/SHA256SUMS` does in the image: names are read
    from the folder it runs in, /app, which is the repo root here."""
    lines = SUMS.read_bytes().decode("ascii").splitlines()
    assert lines
    for line in lines:
        m = re.fullmatch(r"([0-9a-f]{64})  (models/\S+)", line)
        assert m, f"not sha256sum format: {line!r}"
        digest, name = m.groups()
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name


def test_selfcheck_runs_every_model_on_cpu():
    r = subprocess.run([sys.executable, "-m", "faceblur.h5", "selfcheck", "--device", "cpu"],
                       cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    for name in ("yunet", "yolox", "ppocr", "lpd_yunet"):
        assert re.search(rf"(?m)^{name}: \w+ExecutionProvider$", r.stdout), r.stdout
    assert r.stdout.strip().endswith("ok")


def test_selfcheck_fails_with_the_reason(monkeypatch, capsys):
    from faceblur import h5, screens

    def broken(*args, **kwargs):
        raise RuntimeError("model would not load")

    monkeypatch.setattr(screens, "ScreenDetector", broken)
    assert h5.main(["selfcheck", "--device", "cpu"]) == 1
    assert "model would not load" in capsys.readouterr().err
