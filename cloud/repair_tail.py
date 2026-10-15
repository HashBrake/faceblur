"""Copy each cut-off source to in_repaired/ without its damaged last frame(s).
The raw tree in/ is never written. A copy is kept only if its tail decodes clean."""
import subprocess, sys
from pathlib import Path
sys.path.insert(0, "/app/cloud")
import stream_run as s

def run(*a):
    return subprocess.run(a, capture_output=True, text=True)

def tail_ok(p):
    r = run("ffmpeg", "-nostdin", "-v", "error", "-sseof", "-3", "-i", str(p), "-f", "null", "-")
    return not r.stderr.strip()

work = Path("/work/repair"); work.mkdir(parents=True, exist_ok=True)
for rel in open("/tmp/broken.txt").read().split():
    src, dst = work / "src.mp4", work / Path(rel).name
    s.rclone("copyto", f"{s.IN}/{rel}", str(src))
    n = int(run("ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(src)).stdout.strip())
    kept = None
    for drop in range(1, 6):
        run("ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-map", "0:v:0", "-c", "copy",
            "-frames:v", str(n - drop), str(dst))
        if tail_ok(dst):
            kept = n - drop
            break
    if kept is None:
        print(f"GAVE UP {rel}", flush=True)
    else:
        s.rclone("copyto", str(dst), f"vs:{s.BUCKET}/in_repaired/{rel}")
        print(f"repaired {rel}: {n} packets, kept {kept}", flush=True)
    src.unlink(missing_ok=True); dst.unlink(missing_ok=True)
