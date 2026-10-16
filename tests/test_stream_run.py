"""cloud/stream_run.py: its configuration, its shard split, its claims, and whole
SHARD and QUEUE runs against a local directory used as the rclone remote.

Nothing here reaches the bucket: S3_ENDPOINT is kept out of every
environment, REMOTE points at a temporary directory, and a fake pipeline
stands in for cli.py, so no GPU and no real video are needed.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import queue
import random
import shlex
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from conftest import rclone_exe

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "cloud" / "stream_run.py"

# Everything config_from_env reads, plus the S3 keys: none of it may leak in
# from the shell that runs the tests.
NAMES = ("BUCKET", "REMOTE", "IN_PREFIX", "OUT_PREFIX", "WORK", "WORKERS", "LANES",
         "RUN_ARGS", "GPUS", "SHARD", "RCLONE", "PIPELINE_CMD", "CUDA_CHECK", "QUEUE",
         "LEASE", "SELF_DELETE", "GN_SERVER_NAME", "CLAIM_SETTLE", "S3_ENDPOINT",
         "S3_ACCESS_KEY", "S3_SECRET_KEY", "GN_TOKEN_URL", "GN_API_URL", "GN_CLIENT_ID",
         "GN_CLIENT_SECRET", "GN_PROJECT_ID", "GN_PORTAL_USER_ID", "GN_NAME_PREFIX")


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
if os.environ.get("FAKE_FAIL") == src.stem:
    with open(os.environ["FAKE_LOG"], "a", encoding="utf-8") as f:
        f.write(f"{os.getpid()} {src.name}\\n")
    sys.exit(1)
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


def need_rclone() -> str:
    exe = rclone_exe()
    if exe is None:
        pytest.skip("no rclone binary: set RCLONE or put one under tools/")
    return exe


def claim_file(bucket: Path, rel: str) -> Path:
    return bucket / "out" / "_claims" / f"{rel}.json"


def test_a_live_claim_is_never_taken_and_a_stale_one_is(sr, tmp_path):
    exe = need_rclone()
    bucket = tmp_path / "bucket"
    cfg = sr.config_from_env({"REMOTE": bucket.as_posix(), "RCLONE": exe, "LEASE": "30",
                              "CLAIM_SETTLE": "0.2"})
    marker = claim_file(bucket, "a.mp4")
    assert sr.claim(cfg, "a.mp4", "me") is True
    held = json.loads(marker.read_text())
    assert held["owner"] == "me" and held["lease"] == 30
    assert sr.claim(cfg, "a.mp4", "other") is False
    assert json.loads(marker.read_text())["owner"] == "me"
    held["time"] -= 60
    marker.write_text(json.dumps(held))
    assert sr.claim(cfg, "a.mp4", "other") is True
    assert json.loads(marker.read_text())["owner"] == "other"


def test_a_held_claim_is_renewed_and_deleted_when_done(sr, tmp_path):
    exe = need_rclone()
    bucket = tmp_path / "bucket"
    (bucket / "in").mkdir(parents=True)
    (bucket / "in" / "v.mp4").write_bytes(b"x" * 2000)
    cfg = sr.config_from_env({"REMOTE": bucket.as_posix(), "RCLONE": exe, "LEASE": "3",
                              "CLAIM_SETTLE": "0.2", "QUEUE": "1"})
    cfg = dataclasses.replace(cfg, owner="me")
    sizes: dict = {}
    jobs = sr.ClaimQueue(cfg, sizes)
    assert jobs.get_nowait() == "v.mp4"
    assert sizes == {"v.mp4": 2000}
    time.sleep(2.5)
    marker = claim_file(bucket, "v.mp4")
    assert time.time() - json.loads(marker.read_text())["time"] < 1.5
    jobs.done("v.mp4")
    assert not marker.exists()
    # Still unfinished and held by nobody, but a process hands a video out
    # once: a video that failed here is not taken again in a loop.
    with pytest.raises(queue.Empty):
        jobs.get_nowait()


def queue_env(exe: str, bucket: Path, fake: Path, work: Path, calls: Path, **extra) -> dict:
    base = {k: v for k, v in os.environ.items()
            if k not in NAMES and k not in ("FAKE_SLEEP", "FAKE_SLEEP_S", "FAKE_FAIL")
            and not k.startswith("RCLONE_CONFIG")}
    return dict(base, REMOTE=bucket.as_posix(), RCLONE=exe,
                PIPELINE_CMD=shlex.join([Path(sys.executable).as_posix(), fake.as_posix()]),
                CUDA_CHECK="0", GPUS="1", WORK=str(work), QUEUE="1", FAKE_LOG=str(calls),
                WORKERS="1", RUN_ARGS="", **extra)


def make_bucket(tmp_path: Path, sizes: dict) -> tuple[Path, Path]:
    bucket = tmp_path / "bucket"
    rng = random.Random(7)
    for rel, size in sizes.items():
        f = bucket / "in" / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(rng.randbytes(size))
    fake = tmp_path / "fake_cli.py"
    fake.write_text(FAKE_CLI, encoding="utf-8")
    return bucket, fake


def test_three_queue_workers_take_every_video_once(sr, tmp_path):
    exe = need_rclone()
    sizes = {"cam1/a.mp4": 9000, "cam1/b.mp4": 8000, "c.mp4": 7000, "cam2/d.mp4": 6000,
             "cam2/e.mp4": 5000, "f.mp4": 4000, "cam1/g.mp4": 3000, "h.mp4": 2000,
             "cam3/i.mp4": 1500}
    bucket, fake = make_bucket(tmp_path, sizes)
    calls = tmp_path / "calls.log"
    procs = [subprocess.Popen([sys.executable, str(SOURCE)], cwd=ROOT,
                              env=queue_env(exe, bucket, fake, tmp_path / f"work{i}", calls, LEASE="30"),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
             for i in range(3)]
    outs = [p.communicate(timeout=180)[0] for p in procs]
    for p, text in zip(procs, outs):
        assert p.returncode == 0, text

    names = [line.split()[1] for line in calls.read_text(encoding="utf-8").splitlines()]
    assert len(names) == 9, names
    assert sorted(names) == sorted(Path(rel).name for rel in sizes)
    out = bucket / "out"
    for rel in sizes:
        delivered, held = sr.record_names(rel)
        assert (out / delivered).is_file(), rel
        assert not (out / held).exists()
        assert (out / delivered[:-len(".json")]).read_bytes() == b"copy"
    assert len(list(out.rglob("*_blurred.mp4.json"))) == len(sizes)
    assert len(list(out.rglob("*_blurred.mp4"))) == len(sizes)
    dones = [json.loads(f.read_text()) for f in (out / "_logs").glob("q-*_DONE.json")]
    assert len(dones) == 3
    assert sum(d["tally"].get("delivered", 0) for d in dones) == 9
    assert all(set(d["tally"]) <= {"delivered"} for d in dones)
    assert [f for f in (out / "_claims").rglob("*") if f.is_file()] == []


def test_a_dead_workers_video_is_taken_once_its_lease_runs_out(sr, tmp_path):
    exe = need_rclone()
    bucket, fake = make_bucket(tmp_path, {"big.mp4": 5000, "small.mp4": 2000})
    marker = claim_file(bucket, "big.mp4")
    a = subprocess.Popen([sys.executable, str(SOURCE)], cwd=ROOT,
                         env=queue_env(exe, bucket, fake, tmp_path / "work_a", tmp_path / "calls_a.log",
                                       LEASE="3", FAKE_SLEEP="big", FAKE_SLEEP_S="120"),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 60
        while not marker.exists():
            assert a.poll() is None, "worker A ended before it claimed big.mp4"
            assert time.time() < deadline, "worker A never claimed big.mp4"
            time.sleep(0.2)
    finally:
        a.kill()
        a.wait()
    # Windows leaves A's fake pipeline sleeping; it writes only into A's own
    # work dir and log, never into the bucket.
    time.sleep(3 + 1)

    calls_b = tmp_path / "calls_b.log"
    b = subprocess.run([sys.executable, str(SOURCE)], cwd=ROOT,
                       env=queue_env(exe, bucket, fake, tmp_path / "work_b", calls_b, LEASE="3"),
                       capture_output=True, text=True, timeout=120)
    assert b.returncode == 0, b.stdout + b.stderr
    out = bucket / "out"
    for rel in ("big.mp4", "small.mp4"):
        assert (out / sr.record_names(rel)[0]).is_file(), b.stdout
    assert "big.mp4" in calls_b.read_text(encoding="utf-8").split()
    (done,) = (out / "_logs").glob("q-*_DONE.json")
    assert json.loads(done.read_text())["tally"].get("delivered", 0) >= 1
    assert not marker.exists()


def test_a_failing_video_is_tried_once_per_worker(sr, tmp_path):
    exe = need_rclone()
    bucket, fake = make_bucket(tmp_path, {"bad.mp4": 5000, "good.mp4": 2000})
    calls = tmp_path / "calls.log"
    r = subprocess.run([sys.executable, str(SOURCE)], cwd=ROOT,
                       env=queue_env(exe, bucket, fake, tmp_path / "work", calls, LEASE="30",
                                     FAKE_FAIL="bad"),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 1, r.stdout + r.stderr
    names = [line.split()[1] for line in calls.read_text(encoding="utf-8").splitlines()]
    assert sorted(names) == ["bad.mp4", "good.mp4"], r.stdout
    out = bucket / "out"
    assert (out / sr.record_names("good.mp4")[0]).is_file()
    assert not any((out / n).exists() for n in sr.record_names("bad.mp4"))
    assert [f for f in (out / "_claims").rglob("*") if f.is_file()] == []
    (done,) = (out / "_logs").glob("q-*_DONE.json")
    assert json.loads(done.read_text())["tally"] == {"failed": 1, "delivered": 1}


class FakeApi:
    """A GreenNode API on 127.0.0.1 that lists one server, faceblur-test, and
    records every request. On a DELETE it notes whether a DONE marker was
    already in `done_dir`, which is what the hook's ordering promises."""

    def __init__(self):
        self.requests: list[dict] = []
        self.delete_status = 200
        self.done_dir: Path | None = None


@pytest.fixture
def api():
    state = FakeApi()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def answer(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            path = self.path.split("?")[0]
            req = {"method": self.command, "path": path, "body": raw}
            status, reply = 200, {}
            if self.command == "POST" and path == "/token":
                reply = {"access_token": "tok"}
            elif self.command == "GET" and path == "/proj/servers":
                reply = {"listData": [{"name": "faceblur-test", "uuid": "s1"}]}
            elif self.command == "DELETE":
                req["done"] = bool(state.done_dir and list(state.done_dir.glob("*DONE.json")))
                status = state.delete_status
            state.requests.append(req)
            data = json.dumps(reply).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = do_POST = do_DELETE = answer

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]
    state.env = {"SELF_DELETE": "1", "GN_SERVER_NAME": "faceblur-test",
                 "GN_TOKEN_URL": f"http://127.0.0.1:{port}/token",
                 "GN_API_URL": f"http://127.0.0.1:{port}",
                 "GN_CLIENT_ID": "id", "GN_CLIENT_SECRET": "secret", "GN_PROJECT_ID": "proj"}
    yield state
    httpd.shutdown()
    httpd.server_close()


def self_delete_run(tmp_path: Path, env_extra: dict) -> tuple[subprocess.CompletedProcess, Path]:
    exe = need_rclone()
    bucket, fake = make_bucket(tmp_path, {"a.mp4": 3000, "b.mp4": 2000})
    work = tmp_path / "work"
    r = subprocess.run([sys.executable, str(SOURCE)], cwd=ROOT,
                       env=queue_env(exe, bucket, fake, work, tmp_path / "calls.log",
                                     LEASE="30", **env_extra),
                       capture_output=True, text=True, timeout=120)
    return r, work / "stream_run.log"


def test_self_delete_deletes_this_machine_after_the_done_marker(tmp_path, api):
    api.done_dir = tmp_path / "bucket" / "out" / "_logs"
    r, log = self_delete_run(tmp_path, api.env)
    assert r.returncode == 0, r.stdout + r.stderr
    dels = [q for q in api.requests if q["method"] == "DELETE"]
    assert len(dels) == 1, api.requests
    assert dels[0]["path"] == "/proj/servers/s1"
    assert json.loads(dels[0]["body"]) == {"deleteAllVolume": True}
    assert dels[0]["done"] is True
    text = log.read_text(encoding="utf-8")
    assert "self-delete: deleting this machine, faceblur-test" in text
    assert "self-delete: the API accepted" in text


def test_no_self_delete_sends_no_request(tmp_path, api):
    env = {k: v for k, v in api.env.items() if k != "SELF_DELETE"}
    r, log = self_delete_run(tmp_path, env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert api.requests == []
    assert "self-delete" not in log.read_text(encoding="utf-8")


def test_self_delete_refuses_a_name_outside_the_scope(tmp_path, api):
    r, log = self_delete_run(tmp_path, dict(api.env, GN_SERVER_NAME="playerone-x"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert api.requests == []
    assert "refusing" in log.read_text(encoding="utf-8")


def test_self_delete_without_credentials_sends_no_request(tmp_path, api):
    env = {k: v for k, v in api.env.items() if k != "GN_CLIENT_ID"}
    r, log = self_delete_run(tmp_path, env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert api.requests == []
    assert "GN_CLIENT_ID/GN_CLIENT_SECRET/GN_PROJECT_ID missing" in log.read_text(encoding="utf-8")


def test_self_delete_logs_an_api_error_and_does_not_raise(sr, tmp_path, api):
    api.delete_status = 500
    cfg = sr.config_from_env({"REMOTE": tmp_path.as_posix(), "WORK": str(tmp_path),
                              "SELF_DELETE": "1", "GN_SERVER_NAME": "faceblur-test"})
    sr.self_delete(cfg, api.env)
    assert [q["method"] for q in api.requests][-1] == "DELETE"
    text = cfg.log.read_text(encoding="utf-8")
    assert "HTTP 500" in text and "the machine stays" in text


def test_self_delete_without_a_server_name_sends_no_request(sr, tmp_path, api):
    cfg = sr.config_from_env({"REMOTE": tmp_path.as_posix(), "WORK": str(tmp_path)})
    sr.self_delete(cfg, api.env)
    assert api.requests == []
    assert "GN_SERVER_NAME is not set" in cfg.log.read_text(encoding="utf-8")
