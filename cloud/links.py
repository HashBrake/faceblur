"""Make one download link for a whole tree of files in a bucket.

    python cloud/links.py REMOTE_TREE FOLDER_NAME (--public | --expire 168h) [--apply]

The link is a web page. It lists every file of the tree, each one a link,
and gives one command that downloads the whole tree into FOLDER_NAME with the
folder structure kept: curl, which Windows 10 and later, macOS and Linux all
carry, reading a list of files with many downloads at once, resuming a
broken one where it stopped.

`--public` is for a tree whose objects are already public-read (the blurred
tree): the page and the list are public too and never expire. `--expire` is
for a private tree (the raw one): every file is a signed link that stops
working after that long, and so are the page and the list. S3 signs for at
most seven days.

The page and the list are written beside the tree, under `_download/` in the
same bucket. Configuration as in stream_run.py.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from deliver import load_env, rclone  # noqa: E402

PAGE = "_download/index.html"
LIST = "_download/files.txt"


def public_url(bucket: str, key: str) -> str:
    return f"{os.environ['S3_ENDPOINT'].rstrip('/')}/{bucket}/{quote(key)}"


def signed_url(remote: str, expire: str) -> str:
    return rclone("link", remote, "--expire", expire).strip()


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000
    return ""


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tree", help="e.g. vs:faceblur-blurred or vs:faceblur/in")
    ap.add_argument("folder", help="the folder name the download lands in")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--public", action="store_true")
    mode.add_argument("--expire", help="e.g. 168h")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)
    load_env()

    tree = args.tree.rstrip("/")
    bucket, _, prefix = tree.split(":", 1)[1].partition("/")
    files = {e["Path"]: e["Size"] for e in json.loads(
        rclone("lsjson", tree, "--recursive", "--files-only"))
        if not e["Path"].startswith("_download/")}
    total = sum(files.values())
    print(f"{len(files)} files, {human(total)} in {tree}")

    def url(rel: str) -> str:
        key = f"{prefix}/{rel}" if prefix else rel
        return public_url(bucket, key) if args.public else signed_url(f"{tree}/{rel}", args.expire)

    with ThreadPoolExecutor(16) as pool:
        urls = dict(zip(sorted(files), pool.map(url, sorted(files))))

    # curl's config format: one url and one output per file, quoted.
    def q(s: str) -> str:
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
    lines = [f"url = {q(urls[rel])}\noutput = {q(f'{args.folder}/{rel}')}" for rel in sorted(files)]
    file_list = "\n".join(lines) + "\n"

    if not args.apply:
        print("dry run; --apply to write the page and the list")
        return 0

    acl = ["--s3-acl", "public-read"] if args.public else []
    base = f"{tree}"
    rclone("rcat", f"{base}/{LIST}", *acl, stdin=file_list)
    list_url = url(LIST)
    page_rel = PAGE

    flags = "--parallel --parallel-max 16 --create-dirs --retry 5 --continue-at - -K -"
    win = f'curl.exe -s "{list_url}" | curl.exe {flags}'
    nix = f"curl -s '{list_url}' | curl {flags}"
    expiry = ("These links do not expire." if args.public else
              f"These links stop working {args.expire} after they were made. "
              "Ask for new ones after that.")
    rows = "\n".join(
        f'<tr><td><a href="{html.escape(urls[rel])}">{html.escape(rel)}</a></td>'
        f'<td class="n">{human(files[rel])}</td></tr>' for rel in sorted(files))
    folders = sorted({str(PurePosixPath(r).parent) for r in files if "/" in r})
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(args.folder)}</title>
<style>
body{{font:15px/1.5 system-ui,sans-serif;margin:24px auto;max-width:1100px;padding:0 16px;color:#111;background:#fff}}
code,pre{{font:13px ui-monospace,monospace;background:#f3f3f3}} pre{{padding:10px;overflow-x:auto;white-space:pre-wrap;word-break:break-all}}
table{{border-collapse:collapse;width:100%}} td{{padding:2px 8px;border-bottom:1px solid #eee;word-break:break-all}} .n{{text-align:right;white-space:nowrap}}
@media (prefers-color-scheme:dark){{body{{background:#111;color:#eee}} code,pre{{background:#222}} td{{border-color:#333}} a{{color:#8ab4f8}}}}
</style></head><body>
<h1>{html.escape(args.folder)}</h1>
<p>{len(files)} files in {len(folders)} folders, {human(total)} in all. {expiry}</p>
<h2>Download everything</h2>
<p>Open a terminal in the place the folder should go, paste one line, press Enter.
It downloads 16 files at a time and keeps the folder structure. If it stops,
run the same line again: finished files are skipped and a broken one carries on.</p>
<p><b>Windows</b> (PowerShell or Command Prompt):</p><pre>{html.escape(win)}</pre>
<p><b>macOS or Linux</b>:</p><pre>{html.escape(nix)}</pre>
<h2>Every file</h2>
<table>{rows}</table>
</body></html>
"""
    rclone("rcat", f"{base}/{page_rel}", "--header-upload", "Content-Type: text/html; charset=utf-8",
           *acl, stdin=page)
    print("LINK:", url(page_rel))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
