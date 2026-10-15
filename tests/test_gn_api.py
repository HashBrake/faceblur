"""cloud/gn_api.py as a library, against a fake GreenNode API on 127.0.0.1.

No test reaches the real API or reads cloud/greennode.env: every call gets an
env dict that points both endpoints at the fake server, and the CLI tests
replace the module's env() function.
"""
from __future__ import annotations

import base64
import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parent.parent / "cloud" / "gn_api.py"


def load():
    spec = importlib.util.spec_from_file_location("gn_api_under_test", SOURCE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gn = load()


class Fake:
    """The scripted API: the servers it lists, the status it forces, and every request it saw."""

    def __init__(self):
        self.servers: list[dict] = []
        self.status = 200
        self.requests: list[dict] = []


@pytest.fixture
def fake():
    state = Fake()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def answer(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            path = self.path.split("?")[0]
            if raw and self.headers.get("Content-Type") == "application/json":
                body = json.loads(raw)
            else:
                body = raw.decode()
            state.requests.append({
                "method": self.command, "path": path, "body": body,
                "authorization": self.headers.get("Authorization"),
                "content_type": self.headers.get("Content-Type"),
                "portal_user_id": self.headers.get("portal-user-id"),
            })
            parts = path.strip("/").split("/")
            if state.status != 200:
                reply = {"error": "boom"}
            elif self.command == "POST" and path == "/token":
                reply = {"access_token": "tok123"}
            elif self.command == "GET" and len(parts) == 2 and parts[1] == "servers":
                reply = {"listData": state.servers}
            elif self.command == "GET" and len(parts) == 3:
                reply = {"data": next(s for s in state.servers if s["uuid"] == parts[2])}
            elif self.command == "DELETE" and len(parts) == 3:
                reply = {}
            else:
                reply = {}
            data = json.dumps(reply).encode()
            self.send_response(state.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = do_POST = do_DELETE = answer

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    state.env = {
        "GN_CLIENT_ID": "id", "GN_CLIENT_SECRET": "secret", "GN_PROJECT_ID": "proj",
        "GN_PORTAL_USER_ID": "u1",
        "GN_TOKEN_URL": f"http://127.0.0.1:{port}/token",
        "GN_API_URL": f"http://127.0.0.1:{port}",
    }
    yield state
    httpd.shutdown()
    httpd.server_close()


def deletes(fake):
    return [r for r in fake.requests if r["method"] == "DELETE"]


def test_import_ignores_odd_prefix_in_process_env(monkeypatch):
    monkeypatch.setenv("GN_NAME_PREFIX", "wrong-")
    load()  # must not raise, and must not look at the process environment
    with pytest.raises(ValueError):
        gn.scope({"GN_NAME_PREFIX": "wrong-"})
    assert gn.scope({}) == gn.OURS


def test_token_sends_basic_auth_and_form_body(fake):
    assert gn.token(fake.env) == "tok123"
    (r,) = fake.requests
    assert r["method"] == "POST" and r["path"] == "/token"
    assert r["authorization"] == "Basic " + base64.b64encode(b"id:secret").decode()
    assert r["content_type"] == "application/x-www-form-urlencoded"
    assert "grant_type=client_credentials" in r["body"]


def test_servers_sends_bearer_and_portal_user(fake):
    fake.servers = [{"name": "faceblur-5090-1", "uuid": "s1"}, {"name": "other", "uuid": "s2"}]
    assert gn.servers(fake.env) == fake.servers
    r = fake.requests[-1]
    assert r["method"] == "GET" and r["path"] == "/proj/servers"
    assert r["authorization"] == "Bearer tok123"
    assert r["portal_user_id"] == "u1"


def test_delete_server_deletes_the_one_match_with_volumes(fake):
    fake.servers = [{"name": "faceblur-5090-1", "uuid": "s1"}, {"name": "faceblur-5090-2", "uuid": "s2"}]
    assert gn.delete_server("faceblur-5090-1", fake.env) == "s1"
    (d,) = deletes(fake)
    assert d["path"] == "/proj/servers/s1"
    assert d["body"] == {"deleteAllVolume": True}
    assert d["authorization"] == "Bearer tok123"


def test_delete_server_refuses_foreign_name_before_any_request(fake):
    fake.servers = [{"name": "playerone-x", "uuid": "p1"}]
    with pytest.raises(ValueError):
        gn.delete_server("playerone-x", fake.env)
    assert fake.requests == []


def test_delete_server_needs_exactly_one_match(fake):
    with pytest.raises(LookupError):
        gn.delete_server("faceblur-5090-9", fake.env)
    assert deletes(fake) == []
    fake.servers = [{"name": "faceblur-5090-9", "uuid": "a"}, {"name": "faceblur-5090-9", "uuid": "b"}]
    with pytest.raises(LookupError):
        gn.delete_server("faceblur-5090-9", fake.env)
    assert deletes(fake) == []


def test_name_prefix_narrows_what_can_be_deleted(fake):
    env = dict(fake.env, GN_NAME_PREFIX="faceblur-dev-a-")
    fake.servers = [{"name": "faceblur-5090-1", "uuid": "s1"}, {"name": "faceblur-dev-a-1", "uuid": "d1"}]
    with pytest.raises(ValueError):
        gn.delete_server("faceblur-5090-1", env)
    assert fake.requests == []
    assert gn.delete_server("faceblur-dev-a-1", env) == "d1"
    assert [d["path"] for d in deletes(fake)] == ["/proj/servers/d1"]
    with pytest.raises(ValueError):
        gn.scope({"GN_NAME_PREFIX": "other-"})


def test_call_raises_api_error_on_http_500(fake):
    fake.status = 500
    with pytest.raises(gn.ApiError) as exc:
        gn.call("GET", fake.env["GN_API_URL"] + "/proj/servers?page=1", {})
    msg = str(exc.value)
    assert "GET" in msg and "/proj/servers" in msg and "HTTP 500" in msg


def test_cli_names_missing_credentials(monkeypatch):
    monkeypatch.setattr(gn, "env", lambda: {"GN_PROJECT_ID": "proj"})
    monkeypatch.setattr(sys, "argv", ["gn_api.py", "list"])
    with pytest.raises(SystemExit) as exc:
        gn.main()
    assert "GN_CLIENT_ID" in str(exc.value)


def test_cli_refuses_foreign_delete(fake, monkeypatch):
    fake.servers = [{"name": "playerone-x", "uuid": "p1"}]
    monkeypatch.setattr(gn, "env", lambda: dict(fake.env))
    monkeypatch.setattr(sys, "argv", ["gn_api.py", "delete", "playerone-x"])
    with pytest.raises(SystemExit) as exc:
        gn.main()
    assert "refusing" in str(exc.value)
    assert deletes(fake) == []
