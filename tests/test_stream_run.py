"""cloud/stream_run.py: its configuration, its shard split, and a whole SHARD
run against a local directory used as the rclone remote.

Nothing here reaches the bucket: S3_ENDPOINT is kept out of every
environment, REMOTE points at a temporary directory, and a fake pipeline
stands in for cli.py, so no GPU and no real video are needed.
"""
from __future__ import annotations

import importlib.util
import json
import os
import random
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import rclone_exe

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "cloud" / "stream_run.py"

# Everything config_from_env reads, plus the S3 keys: none of it may leak in
# from the shell that runs the tests.
NAMES = ("BUCKET", "REMOTE", "IN_PREFIX", "OUT_PREFIX", "WORK", "WORKERS", "LANES",
         "RUN_ARGS", "GPUS", "SHARD", "RCLONE", "PIPELINE_CMD", "CUDA_CHECK", "QUEUE",
         "LEASE", "SELF_DELETE", "GN_SERVER_NAME", "S3_ENDPOINT", "S3_ACCESS_KEY",
         "S3_SECRET_KEY")


@pytest.fixture
def sr(monkeypatch):
    for name in NAMES:
        monkeypatch.delenv(name, raising=False)
    spec = importlib.util.spec_from_file_location("stream_run", SOURCE)
    mod = importlib.util.module_from_spec(spec)
    # dataclasses looks the module up by name while it builds Config.
    monkeypatch.setitem(sys.modules, "stream_run", mod)
    spec.loader.exec_module(mod)
    return mod


def test_config_needs_a_bucket_or_a_remote(sr):
    with pytest.raises(SystemExit, match="BUCKET.*REMOTE"):
        sr.config_from_env({})


def test_config_defaults_match_the_old_globals(sr):
    cfg = sr.config_from_env({"BUCKET": "b", "S3_ENDPOINT": "x"})
    assert cfg.remote == "vs:b"
    assert cfg.inp == "vs:b/in"
    assert cfg.out == "vs:b/out"
    assert (cfg.shard_k, cfg.shard_n) == (0, 1)
    assert cfg.tag == ""
    assert cfg.workers == "6" and cfg.lanes == 1
    assert cfg.run_args == ["--quarantine"]
    assert cfg.gpus is None
    assert cfg.rclone == "rclone"
    assert cfg.pipeline_cmd == [sys.executable, str(ROOT / "cli.py")]
    assert cfg.cuda_check is True
    assert (cfg.queue, cfg.lease, cfg.self_delete, cfg.gn_server_name) == (False, 1800, False, "")
    assert cfg.work == Path("/tmp/faceblur")
    assert cfg.log == Path("/tmp/faceblur") / "stream_run.log"


def test_config_parses_every_new_knob(sr):
    cfg = sr.config_from_env({"REMOTE": "D:/x", "SHARD": "1/2", "CUDA_CHECK": "0",
                              "PIPELINE_CMD": "python fake.py", "QUEUE": "1", "LEASE": "3",
                              "SELF_DELETE": "1", "GN_SERVER_NAME": "box", "RCLONE": "r.exe",
                              "GPUS": "2", "RUN_ARGS": "", "WORK": "D:/w"})
    assert cfg.remote == "D:/x"
    assert cfg.inp == "D:/x/in" and cfg.out == "D:/x/out"
    assert (cfg.shard_k, cfg.shard_n) == (1, 2)
    assert cfg.tag == "shard1of2_"
    assert cfg.cuda_check is False
    assert cfg.pipeline_cmd == ["python", "fake.py"]
    assert cfg.queue is True and cfg.lease == 3
    assert cfg.self_delete is True and cfg.gn_server_name == "box"
    assert cfg.rclone == "r.exe" and cfg.gpus == 2 and cfg.run_args == []
    assert cfg.log == Path("D:/w") / "stream_run.log"


def test_vs_remote_only_with_an_endpoint(sr):
    env = {"REMOTE": "D:/x"}
    sr.vs_remote(env)
    assert env == {"REMOTE": "D:/x"}
    env = {"S3_ENDPOINT": "e", "S3_ACCESS_KEY": "a", "S3_SECRET_KEY": "s"}
    sr.vs_remote(env)
    assert env["RCLONE_CONFIG_VS_ENDPOINT"] == "e"
    assert env["RCLONE_CONFIG_VS_ACCESS_KEY_ID"] == "a"


def test_share_is_balanced_and_deterministic(sr):
    # i=9 ... a=1. Largest first to the lightest share, lowest index on a tie:
    # i0 h1 g2 f2 e1 d0 c0 b1 a2, loads 16, 15, 14.
    sizes = {c: n for n, c in enumerate("abcdefghi", start=1)}
    shares = [sr.share(sizes, k, 3) for k in range(3)]
    assert shares == [{"i", "d", "c"}, {"h", "e", "b"}, {"g", "f", "a"}]
    assert set().union(*shares) == set(sizes)
    assert sum(len(s) for s in shares) == len(sizes)
    assert sr.share(dict(reversed(sizes.items())), 1, 3) == shares[1]


FAKE_CLI = '''
import json, os, sys, time
from pathlib import Path
src = Path(sys.argv[1])
out = Path(sys.argv[sys.argv.index("-o") + 1])
if os.environ.get("FAKE_SLEEP") == src.stem:
    time.sleep(float(os.environ.get("FAKE_SLEEP_S", "5")))
out.mkdir(parents=True, exist_ok=True)
(out / f"{src.stem}_blurred.mp4").write_bytes(b"copy")
(out / f"{src.stem}_blurred.mp4.json").write_text(json.dumps(
    {"status": "done", "compute": {"yunet": "CUDAExecutionProvider"}}))
with open(os.environ["FAKE_LOG"], "a", encoding="utf-8") as f:
    f.write(f"{os.getpid()} {src.name}\\n")
'''


def test_shard_run_on_a_local_remote(sr, tmp_path):
    exe = rclone_exe()
    if exe is None:
        pytest.skip("no rclone binary: set RCLONE or put one under tools/")
    assert sr.SUFFIX == "_blurred"
    bucket = tmp_path / "bucket"
    rng = random.Random(6)
    sizes = {"cam1/a.mp4": 4000, "cam1/b.mp4": 3000, "c.mp4": 2000, "cam2/d.mp4": 1500}
    for rel, size in sizes.items():
        f = bucket / "in" / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(rng.randbytes(size))
    fake = tmp_path / "fake_cli.py"
    fake.write_text(FAKE_CLI, encoding="utf-8")
    calls = tmp_path / "calls.log"
    base = {k: v for k, v in os.environ.items()
            if k not in NAMES and not k.startswith("RCLONE_CONFIG")}

    def run(shard: str, work: str) -> subprocess.CompletedProcess:
        env = dict(base, REMOTE=bucket.as_posix(), RCLONE=exe,
                   PIPELINE_CMD=shlex.join([Path(sys.executable).as_posix(), fake.as_posix()]),
                   CUDA_CHECK="0", GPUS="1", WORK=str(tmp_path / work), SHARD=shard,
                   FAKE_LOG=str(calls), WORKERS="1", RUN_ARGS="")
        return subprocess.run([sys.executable, str(SOURCE)], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=120)

    def names() -> list[str]:
        return [line.split()[1] for line in calls.read_text(encoding="utf-8").splitlines()]

    r0 = run("0/2", "work0")
    assert r0.returncode == 0, r0.stdout + r0.stderr
    shard0 = names()
    r1 = run("1/2", "work1")
    assert r1.returncode == 0, r1.stdout + r1.stderr

    assert sorted(names()) == sorted(Path(rel).name for rel in sizes)
    ours = sr.share(sizes, 0, 2)
    assert ours == {"cam1/a.mp4", "cam2/d.mp4"}
    assert sorted(shard0) == sorted(Path(rel).name for rel in ours)
    out = bucket / "out"
    for rel in sizes:
        delivered, held = sr.record_names(rel)
        assert (out / delivered).is_file(), rel
        assert not (out / held).exists()
        assert (out / delivered[:-len(".json")]).read_bytes() == b"copy"
    assert len(list(out.rglob("*_blurred.mp4.json"))) == len(sizes)
    for k, n in ((0, 2), (1, 2)):
        done = json.loads((out / "_logs" / f"shard{k}of2_DONE.json").read_text())
        assert done["tally"] == {"delivered": n} and done["todo"] == n
        assert (out / "_logs" / f"shard{k}of2_stream_run.log").is_file()
        assert (out / "_logs" / f"shard{k}of2_gpu0_cli.log").is_file()

    before = calls.read_text(encoding="utf-8")
    r2 = run("0/2", "work2")
    assert r2.returncode == 0, r2.stdout + r2.stderr
    assert calls.read_text(encoding="utf-8") == before
    done = json.loads((out / "_logs" / "shard0of2_DONE.json").read_text())
    assert done["todo"] == 0 and done["tally"] == {}
