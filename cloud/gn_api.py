"""Delete the run's GreenNode VMs through the vServer API, without the console.

The console session ends after about 12 hours whatever is done in it, and a
slice can finish at night. This deletes a VM, with its volumes, the moment its
slice is done.

    python cloud/gn_api.py list                      # read only: our servers
    python cloud/gn_api.py delete faceblur-5090-1    # one VM and its volumes
    python cloud/gn_api.py create faceblur-5090-d --like faceblur-5090-b-3          # dry run
    python cloud/gn_api.py create faceblur-5090-d --like faceblur-5090-b-3 --apply  # creates it

`create` copies a running VM's flavour, image, network, subnet, security
groups, SSH key, zone and boot disk, with a public IP and no password change
at first login, so a new VM matches the fleet exactly. Without `--apply` it
only prints the request. Then `cloud/vm_bootstrap.sh IP k/n setup run` puts
it to work.

Credentials come from cloud/greennode.env (or the environment): GN_CLIENT_ID
and GN_CLIENT_SECRET of an IAM service account allowed to manage vServer, and
GN_PROJECT_ID. GN_PORTAL_USER_ID is sent when set; the docs list the header
as required. Only servers named faceblur-* can be deleted, so a typo can never
reach anyone else's machine.

Docs: https://docs.api.greennode.ai/service-docs/vserver.html
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOKEN_URL = "https://iamapis.vngcloud.vn/accounts-api/v2/auth/token"
API = "https://hcm-3.api.vngcloud.vn/vserver/vserver-gateway/v2"
OURS = "faceblur-"


def env() -> dict[str, str]:
    values = dict(os.environ)
    path = HERE / "greennode.env"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                values.setdefault(k.strip(), v.strip())
    return values


def call(method: str, url: str, headers: dict, body: bytes | None = None) -> dict:
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            text = r.read().decode() or "{}"
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"{method} {url.split('?')[0]}: HTTP {exc.code} {exc.read().decode()[:300]}")
    return json.loads(text)


def token(e: dict) -> str:
    basic = base64.b64encode(f"{e['GN_CLIENT_ID']}:{e['GN_CLIENT_SECRET']}".encode()).decode()
    r = call("POST", TOKEN_URL, {"Authorization": f"Basic {basic}",
                                 "Content-Type": "application/x-www-form-urlencoded"},
             b"grant_type=client_credentials&scope=email")
    return r.get("access_token") or r["accessToken"]


def headers(e: dict) -> dict:
    h = {"Authorization": f"Bearer {token(e)}", "Content-Type": "application/json"}
    if e.get("GN_PORTAL_USER_ID"):
        h["portal-user-id"] = e["GN_PORTAL_USER_ID"]
    return h


def servers(e: dict, h: dict) -> list[dict]:
    r = call("GET", f"{API}/{e['GN_PROJECT_ID']}/servers?page=1&size=100", h)
    return r.get("listData") or r.get("data") or []


def server(e: dict, h: dict, name: str) -> dict:
    match = [s for s in servers(e, h) if s.get("name") == name]
    if len(match) != 1:
        raise SystemExit(f"{name}: {len(match)} servers by that name")
    sid = match[0].get("uuid") or match[0].get("id")
    r = call("GET", f"{API}/{e['GN_PROJECT_ID']}/servers/{sid}", h)
    return r.get("data", r)


def create_body(e: dict, h: dict, name: str, like: str) -> dict:
    """The request for a new server set up exactly like the running `like`."""
    t = server(e, h, like)
    nic = t["internalInterfaces"][0]
    boot = call("GET", f"{API}/{e['GN_PROJECT_ID']}/volumes/{t['bootVolumeId']}", h)
    boot = boot.get("data", boot)
    keys = call("GET", f"{API}/{e['GN_PROJECT_ID']}/sshKeys?page=1&size=100", h)
    key = [k for k in (keys.get("listData") or keys.get("data") or [])
           if k.get("name") == t["sshKeyName"]]
    if len(key) != 1:
        raise SystemExit(f"SSH key {t['sshKeyName']}: {len(key)} found")
    return {
        "name": name, "zoneId": t["zoneId"],
        "flavorId": t["flavor"]["flavorId"], "imageId": t["image"]["id"],
        "networkId": nic["networkUuid"], "subnetId": nic["subnetUuid"],
        "securityGroup": [g["uuid"] for g in t["secGroups"]],
        "sshKeyId": key[0].get("id") or key[0].get("uuid"),
        "rootDiskSize": boot.get("size"), "rootDiskTypeId": boot.get("volumeTypeId"),
        "encryptionVolume": False, "attachFloating": True,
        # The API's own wording: "Skip change password: false". Without it
        # the first login asks for a password nobody has.
        "expirePassword": False,
        "createdFrom": "NEW",
    }


def main() -> int:
    e = env()
    missing = [k for k in ("GN_CLIENT_ID", "GN_CLIENT_SECRET", "GN_PROJECT_ID") if not e.get(k)]
    if missing:
        raise SystemExit(f"missing in cloud/greennode.env: {', '.join(missing)}")
    h = headers(e)
    if sys.argv[1:2] == ["list"]:
        for s in servers(e, h):
            print(s.get("name"), s.get("uuid") or s.get("id"), s.get("status"))
        return 0
    if len(sys.argv) == 3 and sys.argv[1] == "delete":
        name = sys.argv[2]
        if not name.startswith(OURS):
            raise SystemExit(f"refusing: {name} is not one of ours ({OURS}*)")
        match = [s for s in servers(e, h) if s.get("name") == name]
        if len(match) != 1:
            raise SystemExit(f"{name}: {len(match)} servers by that name, nothing deleted")
        sid = match[0].get("uuid") or match[0].get("id")
        call("DELETE", f"{API}/{e['GN_PROJECT_ID']}/servers/{sid}", h,
             json.dumps({"deleteAllVolume": True}).encode())
        print(f"deleted {name} ({sid}) with its volumes")
        return 0
    if len(sys.argv) >= 5 and sys.argv[1] == "create" and sys.argv[3] == "--like":
        name, like = sys.argv[2], sys.argv[4]
        if not name.startswith(OURS):
            raise SystemExit(f"refusing: {name} is not one of ours ({OURS}*)")
        if any(s.get("name") == name for s in servers(e, h)):
            raise SystemExit(f"{name} already exists")
        body = create_body(e, h, name, like)
        if "--apply" not in sys.argv:
            print("dry run, nothing created:")
            print(json.dumps(body, indent=2))
            return 0
        r = call("POST", f"{API}/{e['GN_PROJECT_ID']}/servers", h, json.dumps(body).encode())
        print(json.dumps(r)[:500])
        return 0
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
