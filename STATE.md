# STATE, handover notes

Last updated 2026-10-16 01:50. **All eight blur slices are done** (8 RTX 5090
VMs, 2026-10-14 18:20 to 2026-10-16 01:45) and all eight VMs are deleted,
the last three through the API. Left on the cloud: `faceblur-hands-pilot`,
re-running the 2 cut-off videos of `AZER76400FF_19700101_000137` with the
stitch fix (the other 7 cut-off videos are done). Then: check every video
has a record, delivery, the blurred link, cleanup. In parallel, the partner's
hand toolkit (EgoSmith, HaWoR + MANO) ran its pilot but is not good enough
to show yet (see "The hand toolkit"); four build sessions run locally
("Build sessions from the build plan"). "The production run"
below is the live work and is read first.
`BUILD_PLAN_PROCESSING_AND_STORAGE.md` (2026-10-16) is the plan for after this run:
better processing at lower compute, and HEVC/Parquet hot storage.

Before that, 2026-10-01: the gate made crash safe (report 24.4) and the gates
moved to the mask budget (report 25). 2026-09-26: a run that hung on a dead
worker, and "recreate the whole folder" in the window. 2026-09-14: a full run
of the pipeline over the four sample files (report 24), after the unattended
weekend build of `FACEBLUR_BUILD_PLAN_V2.md`. Packages R1, S1, 4.3, E1, R2,
H1, H2, R3, G1, F2, F1, T1, S2 and M1 are done. **T3, D and T4 are blocked**
on numbers rather than on time; report 21.5 says by what, and what is left
there needs footage with cards and words in it.

This file says where the project stands and what to do next. `README.md` says
how to use the tool. `docs/report.md` holds the measurements, one section per
pass, and is the place to look before changing anything measured.
`FACEBLUR_BUILD_PLAN_V2.md` is the spec being executed; where it and the code
disagree the code wins and this file says so.

Read "The production run", "The rule" and "Where things stand" first.
Everything else is reference.

## The production run

### What is asked

The owner, 2026-10-05 to 07: blur all of `D:\Player One - Phase 0 full` and
give two links, one to the raw folder and one to the blurred folder, each of
which lets anybody anywhere download the whole tree quickly. Nothing is
downloaded back to this PC; a person in another country downloads it.

- **The blurred tree is the source tree exactly**, same folders and names,
  every `.mp4` replaced by its blurred copy under the same name. `*_audio.wav`
  and `*_audio_pts.csv` are not delivered at all. The recorder's 24 byte stub
  videos go as they are (no frames, nothing to blur). A copy the gate held
  back is delivered blurred and listed in `faceblur-records/held_back.csv`. A
  real video with no finished copy is never delivered raw; it is listed in
  `faceblur-records/not_delivered.txt`. FaceBlur's own records go to
  `faceblur-records/`, beside the tree, not in it.
- **Settings**: faces, the zone, the check, the gate, one second chance round
  (`--quarantine --second-chance 1`). Confirmed by the owner after the cost
  was explained.
- **Links**: the blurred tree in its own public read bucket, `faceblur-blurred`,
  with a link that never expires; a separate bucket so a misconfiguration can
  never make raw footage public. The raw tree stays private and its link is
  signed, at most seven days, regenerated on request. Each link is a web page
  listing every file plus one `curl` line that downloads the whole tree with
  its folders, 16 files at a time, resuming. Still to build: one zip per
  device folder, on a cheap CPU instance.

### The footage

119 hours, left and right camera per session, about 1.1 TB with the extras:
1447 files in 150 folders, 351 `.mp4` (321 real, 30 empty or broken, most of
those the 24 byte stubs), 148 `.wav`, 650 `.csv`, 298 `.yaml`. **This is not
the four sample files** and it has never been measured as a whole; the speed
estimates below come from sample clips.

### Where everything is

GreenNode (VNG Cloud), account `alois@vng.com.vn`, a company account paying
as it goes with no credits. The console is driven through Chrome on this PC;
its session expires within hours, and deleting anything needs it.

| What | Where |
|---|---|
| Raw tree, complete | vStorage HCM04, bucket `faceblur`, prefix `in/`. Verified 2026-10-07: `rclone check` against `D:`, 0 differences |
| Raw link page and list | `faceblur/in/_download/` |
| Blurred copies and records, as the run writes them | `faceblur/out/`, logs in `out/_logs/` |
| Delivered blurred tree | bucket `faceblur-blurred`, made by `cloud/deliver.py` after the run |
| Image | `vcr.vngcloud.vn/110156-faceblur/faceblur-cuda:3` (CUDA 12.8, onnxruntime-gpu 1.24.4, ffmpeg with NVENC, rclone 1.75.1). Also on this PC's Docker |
| Secrets | `cloud/greennode.env`, gitignored: S3 keys and the registry robot. SSH key `~/.ssh/faceblur_greennode` |
| Also ours | AI Platform network volume `faceblur-test-nv` (empty, the notebook form requires one) |

Other resources on the account (`playerone-sandbox`, `playerone-demo`,
`cer-dev`, `unitree-g1`) are not ours. Never touch them.

rclone is not installed on this PC. The upload and the links ran in
containers of the `faceblur-cuda` image (`docker run --env-file
cloud/greennode.env ...`), which has it; `cloud/deliver.py` and
`cloud/links.py` call `rclone` and expect the same. Docker's disk lives on
`D:\DockerDesktopWSL`, because C: has about 19 GB free.

### The code for it

Commits `110288d` to `3558c42`, all pushed to
`HashBrake/faceblur`. Tests: 1285 passing.

| Path | What |
|---|---|
| `cloud/Dockerfile`, `requirements-cuda.txt`, `entry.sh` | the Linux CUDA image |
| `cloud/stream_run.py` | the runner: one file at a time per GPU, from the bucket and back; `SHARD=k/n` splits the inputs between machines balanced by size; a card that cannot run CUDA is stopped rather than left on the processor; a card waits for disk room (3.5 times the file plus 1 GB) |
| `cloud/notebook_run.sh` | starts the run on a notebook instance; returns at once, the run goes on under `nohup` |
| `cloud/deliver.py` | assembles the blurred tree in the public bucket by server side copies; refuses if any record says the copy kept audio; ends by checking that no raw video and no audio is in the tree |
| `cloud/links.py` | the download page and the `curl` list for a tree |
| `cloud/bench_detect.py`, `compare.py`, `run_shards.py` | measuring a card, comparing two copies, local shard runs |
| `cloud/instance_run.sh` | for a plain VM; superseded by `notebook_run.sh` |

Three fixes to the tool came out of it, all in `110288d` and `ef3a3a5`:

- **A recursive run never made the output subfolders**, so every video in a
  nested folder failed. `batch.py` makes the folder now.
- **A restart re-ran a video the gate had held back**, because it looked for
  the copy only in the output folder. It looks in `quarantine/` too.
- **The check found the hands a second time.** The detection pass now hands
  its hands to the check when `zone_stride` is 1. Output is byte for byte the
  same on `004100` and `004310`, records and copies; the check went 206 s to
  108 s and 264 s to 179 s.

### Running now (2026-10-14)

Eight GreenNode vServer VMs, `g7-standard-24x64-1rtx5090` (one RTX 5090, 24
vCPU, 64 GB, 300 GB NVMe), zone HCM03-1B, subnet `faceblur-1b` in VPC
`net192`, one `SHARD=k/8` each, 8 workers a card, started 18:21 to 19:14.
`bash cloud/vm_status.sh` prints one line per VM. SSH is port 234, user
`stackops`, key `~/.ssh/faceblur_greennode_rsa`; `cloud/vm_bootstrap.sh IP k/8
setup run` puts a fresh VM to work. Make each VM with "Skip password change at
first login" ticked. Billing is by the minute used (GreenNode, Teams,
2026-10-14), although the form shows a 30 day total. The vCPU quota was raised
to 250 on request.

| VM | IP | Slice |
|---|---|---|
| ~~faceblur-5090-1~~ | 116.118.92.106 | 0/8, done 2026-10-16 00:33 (its cut-off video is on the pilot VM), deleted with its volume |
| ~~faceblur-5090-3~~ | 116.118.92.82 | 1/8, done 2026-10-15 21:46 (its one cut-off video is on the pilot VM), deleted with its volume |
| ~~faceblur-5090-8~~ | 116.118.92.84 | 2/8, done 2026-10-15 23:42 (its cut-off videos are on the pilot VM), deleted with its volume 2026-10-16 |
| ~~faceblur-5090-b-1~~ | 116.118.92.112 | 3/8, done 2026-10-15 21:35, deleted with its volume |
| ~~faceblur-5090-b-2~~ | 116.118.92.109 | 4/8, done 2026-10-15 21:49 (its two cut-off videos are on the pilot VM), deleted with its volume |
| ~~faceblur-5090-b-3~~ | 116.118.92.113 | 5/8, done 2026-10-16 01:45 (its one failure is the cut-off FF_000137 right, on the pilot VM), deleted with its volume through the API |
| ~~faceblur-5090-b-4~~ | 116.118.92.111 | 6/8, done 2026-10-16 01:38 (38/38, no failures), deleted with its volume through the API |
| ~~faceblur-5090-c~~ | 116.118.92.114 | 7/8, done 2026-10-15 23:49, deleted with its volume 2026-10-16 |

Measured on a 5090: detection 17.4, 21.9, 24.8 frames a second at 4, 6, 8
processes; GPU 77 to 80 percent busy at 8 workers, load about 9 of 24 cores.
A whole video took 2.1 to 2.8 h on one lane, and the GPU sat idle for about
half of it: the tracker, the zone and the encode use one core, and the
second chance runs them twice. From 2026-10-15 00:28 every VM runs
`LANES=2 WORKERS=6` (two videos at once, restarted at a video boundary so no
work was lost); the GPU is 80 to 85 percent busy throughout.
One VM (`b-5`) lost its GPU from the bus four minutes after boot (Xid 154) and
again after a reboot; it was deleted and replaced, and GreenNode was told.

Cut-off sources (2026-10-15): 13 source videos end in a damaged last frame (the
camera stopped mid-write), so the writer refuses them ("decoded N frames,
expected N+1"). Each is copied without its damaged frame(s) to
`faceblur/in_repaired/` (same paths; `in/` is untouched) and blurred on the
pilot VM with `IN_PREFIX=in_repaired`, writing to the usual `out/` names. The
slices also try them and log a failure; those failures are expected. Delete
`in_repaired/` after delivery. Same day: two videos sat 12 h in the tracker's
stitch step (fixed in 1e62b27, deployed by docker cp into the running
containers; any new container from image `:3` needs the same `docker cp`, which the
pilot VM's first did not get, and lost 2 h on the same pair). The pair is in
the cut-off list too, so the pilot VM re-runs it.

Hand toolkit pilot (2026-10-15): VM `faceblur-hands-pilot`, 116.118.92.130, same
flavor, created 14:20; delete it with its volume when the pilot is done.

To remove when the run is delivered: the eight VMs with their volumes, subnet
`faceblur-1b`, security group `faceblur-ssh` (unused), SSH key
`faceblur_greennode` in the console.

### Build sessions from the build plan (2026-10-16)

Four autonomous Claude Code sessions, one per part of
`BUILD_PLAN_PROCESSING_AND_STORAGE.md`, each in its own git worktree and
branch, all from main 3eb0100, local PC only (no cloud):

| Session | Worktree | Branch | Kickoff prompt |
|---|---|---|---|
| A robustness (1.1) | `D:\Claude Projects\faceblur-robustness` | `build/robustness` | `D:\Claude Projects\kickoffs\A_robustness.txt` |
| B blur speed (1.2) | `D:\Claude Projects\faceblur-blur-speed` | `build/blur-speed` | `...\kickoffs\B_blur-speed.txt` |
| C stereo hands (1.3) | `D:\Claude Projects\faceblur-stereo-hands` | `build/stereo-hands` | `...\kickoffs\C_stereo-hands.txt` |
| D hot storage (2) | `D:\Claude Projects\faceblur-hot-storage` | `build/hot-storage` | `...\kickoffs\D_hot-storage.txt` |

Each worktree has the orchestration `CLAUDE.md` (Fable orchestrates, Opus
builders commit) and its own PLAN/STATUS/DECISIONS/BUILD_REPORT ledgers; these
are excluded through `.git/info/exclude`, so they never reach a commit. They
share one GPU through `D:\Claude Projects\gpu_lock.py`. Merge order
afterwards: A, B, D, C (A, B and D all touch `faceblur/` and `cli.py`; C is a
new package). Items 1.4 (one pass) and 1.5 (hardware) wait for these.

Cloud GPU for the sessions (owner, 2026-10-16): each may run one VM at a
time, named `faceblur-dev-<a|b|c|d>-N`, created and deleted only through
`GN_NAME_PREFIX=faceblur-dev-<x>- cloud/gn_api.py` (the prefix makes the
script refuse every other server), from `cloud/vm_template_5090.json`;
budget 24 VM-hours per session; this session's 10-minute check deletes any
faceblur-dev VM older than 24 h. The addenda pasted into the running sessions
are `D:\Claude Projects\kickoffs\cloud_addendum_<X>.txt`. `vm_bootstrap.sh`
no longer copies the `GN_*` API keys onto a VM.

### The hand toolkit (EgoSmith, 2026-10-15)

The partner's code, `egosmith_hand_mano_toolkit` (upload of 2026-10-15; not
in this repo, it is theirs), estimates both hands per frame from the left
camera: WiLoR's YOLO hand detector, then HaWoR, then MANO for 21 joints in
camera coordinates, plus a line-only skeleton video. It needs MANO; the
owner supplied `MANO_LEFT.pkl`/`MANO_RIGHT.pkl` 2026-10-16, kept in
`cloud/mano/` (gitignored, licensed) and on the pilot VM in `/root/mano/`.
HaWoR is CC BY-NC-ND and MANO non-commercial: the owner will settle the
licence later ("ignore the licensing for now").

- Its PyTorch 2.5.1/cu121 has no RTX 5090 kernels. `cloud/hands/Dockerfile`
  builds it with PyTorch 2.7.1/cu128 (build context: the Dockerfile next to
  the extracted toolkit folder). Built as `egosmith:cu128` on
  `faceblur-hands-pilot` (116.118.92.130); environment check and its 9 tests
  pass. Downloads from these VMs run at about 1.4 MB/s, so a build is ~40 min.
- It processes only the first `*_camera_left*.mp4` of a recording folder, so
  each part must be presented as its own folder, with the
  `*_calibration_camera.yaml` beside it. It unpacks every frame to JPEG
  (~50 GB per hour of video).
- Run: `docker run --gpus all --ipc=host -v /root/mano/MANO_RIGHT.pkl:/opt/egosmith/_DATA/data/mano/MANO_RIGHT.pkl:ro -v /root/mano/MANO_LEFT.pkl:/opt/egosmith/_DATA/data_left/mano_left/MANO_LEFT.pkl:ro -v /var/egosmith:/data egosmith:cu128 bash scripts/run_camera_hand_skeleton_pipeline.sh --input-root /data/recordings --output-root /data/results`
- Pilot: a 10 min cut of a blurred left video
  (`AZER76400EV_19700105_003108`, part 1), started 2026-10-16 00:13 while the
  VM also blurs. PyTorch 2.6+ refuses HaWoR's checkpoint by default
  (`weights_only`); the official file loads with
  `-e TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`. Result 2026-10-16 01:00: ran end to end
  (13.5 fps end to end, 21 fps HaWoR, GPU shared with blurring); a hand in
  67 % (left) and 82 % (right) of frames. **Not good enough to show**: drawn
  through the calibration, skeletons come out ~5x too small, and hands holding
  tools are often missed. Likely cause: the fisheye image goes into HaWoR,
  which assumes a pinhole camera, with only the focal length. Next: undistort
  to pinhole before the toolkit; ask EgoSmith how they run it on Orbbec
  footage. The showcase waits for a good result.
- The owner asked (2026-10-16): if positive, run 2 to 3 h of left videos from
  different devices, add a version with the skeleton over the blurred
  footage, and publish them on a separate link (signed, 7 days, unless the
  owner wants it permanent).

### Running it

On a GreenNode AI Platform **notebook instance** with the image above. The
platform replaces the image's command and runs only `sshd`, so the run is
started over SSH, on the forwarded port the console shows:

```
cat cloud/greennode.env | ssh -p PORT root@HOST 'cat > /root/faceblur.env && chmod 600 /root/faceblur.env'
cat cloud/stream_run.py   | ssh -p PORT root@HOST 'cat > /app/cloud/stream_run.py'
cat cloud/notebook_run.sh | ssh -p PORT root@HOST 'cat > /app/cloud/notebook_run.sh'
ssh -p PORT root@HOST 'SHARD=0/1 bash -l /app/cloud/notebook_run.sh'
```

The two script copies are needed because image `:3` predates the last two
commits; an image `:4` would make them unnecessary. Use `bash -l` (the PATH is
set in `/etc/profile.d`). `sftp` is disabled, so `scp` needs `-O`. Several
single card machines are one `SHARD=k/n` each, k counted from 0 (`0/8` to `7/8`). A restart skips what is
finished, held back copies included.

What went wrong on the platform:

- **Block storage above 20 GB fails to provision** on this account
  (2026-10-07), so the work folder is on the root disk, `/var/faceblur`.
- **No RTX 4090 capacity.** On 2026-10-06 one 1x4090 in zone HCM-03-1B came up
  and was measured; 1A failed. On 2026-10-07 every 4090 notebook, 8 card and
  single, failed with a local storage error after about 20 minutes, and
  GreenNode confirmed the 4090 is out of stock. All failed instances were
  deleted.

### Speed and cost

Measured on the one 4090 (driver 580, CUDA 13): five workers a card is best;
a 120 s clip took 586 to 634 s with the check and the second chance, GPU at 47
percent, before `ef3a3a5` cut the check. Estimates for the 119 hours:

| Where | Time | Cost |
|---|---|---|
| 8x RTX 4090 (`48x512-8rtx4090`, about 121,600 VND/h) | 55 to 60 h | about 7M VND |
| 8x A40 (`48x512-8A40`, about 175,000 VND/h) | 60 to 70 h | about 11 to 12M VND |
| This PC's RTX 3070 | 4 to 6 weeks without a stop | power only |

The A40 is not measured. Before committing to the 8 card machine, run
`16x64-1A40` for 20 minutes, check that it uses CUDA and set `WORKERS` from
what it does. The PC figure is 5.5 to 8 times real time from the sample clips.

Uploading from the office runs at about 45 Mbps to vStorage, 5.5 MiB/s, and
faster in the evening; more parallel streams did not help.

### The rules for instances

The owner, 2026-10-05: instances may be created as needed **and must be
deleted when done**. Deleting needs the console session, which expires, so
**the 8 card machine never runs unattended past the point where somebody can
sign in again**. The run itself does not delete the instance when it ends.

### Next, in order

1. **Delete finished VMs through the API** (working since 2026-10-16 01:30):
   `py -3.12 cloud/gn_api.py list` / `delete <name>` (deletes volumes too,
   refuses names not starting with `faceblur-`). The IAM service account
   `faceblur-automation` (the owner made it, with full access) has its
   `GN_CLIENT_ID`/`GN_CLIENT_SECRET` and `GN_PORTAL_USER_ID=110156` in
   `cloud/greennode.env`. The console (12 h sessions) is no longer needed for
   this. Creating: `py -3.12 cloud/gn_api.py create NAME --like RUNNING_VM [--apply]`
   copies a running VM's setup (dry run without `--apply`; not yet run for
   real), then `cloud/vm_bootstrap.sh IP k/n setup run`. The API showed the
   5090 flavour with `remainingVms: 1` on 2026-10-16: ask GreenNode first.
2. A slice's failures are fine to leave when every one is in the cut-off list
   (`faceblur/in_repaired/`, blurred by the pilot VM): stop its container
   with `docker update --restart=no faceblur && docker stop faceblur` so it
   does not retry them, then delete the VM.
3. When everything has run: read `out/_logs/` (DONE markers, held back,
   failed). Every real video must have a record; the pilot VM's run on
   `in_repaired/` writes to the same `out/` names.
4. `python cloud/deliver.py "D:\Player One - Phase 0 full"`, read the dry run,
   then `--apply`. It must end with 0 raw videos and 0 audio files found.
5. `python cloud/links.py vs:faceblur-blurred "Player One - Phase 0 full (blurred)" --public --apply`
   gives the blurred link.
6. Hand toolkit showcase, if the pilot is positive (see "The hand toolkit").
7. **Delete every instance** (incl. `faceblur-hands-pilot`), then subnet
   `faceblur-1b`, SG `faceblur-ssh`, console SSH key `faceblur_greennode`, AI
   Platform volume `faceblur-test-nv`, and `faceblur/in_repaired/`. Keep
   `faceblur/in/`. `faceblur/out/` can go once the delivered tree is verified.
8. Zips per device folder only if still wanted (video does not compress).
9. Push whatever is committed by then.

Open on the side:

- **The raw link expires 2026-10-22 04:16 UTC** (renewed 2026-10-15; the
  old one ended that day). A new one, from the repo root in the image, which
  has rclone and needs the repo on `PYTHONPATH`:
  `MSYS_NO_PATHCONV=1 docker run --rm --env-file cloud/greennode.env -v "$PWD:/w" -w /w -e PYTHONPATH=/w --entrypoint python vcr.vngcloud.vn/110156-faceblur/faceblur-cuda:3 cloud/links.py vs:faceblur/in "Player One - Phase 0 full" --expire 168h --apply`.
  Each renewal is a new URL; send it to whoever has the old one.
- **Outbound traffic.** A full download is 1.1 TB per link per person, and
  GreenNode publishes no price for international traffic. Ask before it is
  downloaded many times.
- **Storage** costs about 28 VND per GB per day, so about 31,000 VND a day
  for the raw tree, and the same again for the blurred one.
- **The registry secret** was rotated on 2026-10-07 and then appeared in a
  browser agent's screenshot once. Rotate it again.

## The rule

The owner, 2026-09-11: *remove all information not pertinent to the task the
collector is doing. Anything the collector is interacting with or handling is
not blurred at all. Everything else that carries information (text, screens,
faces) is blurred.*

This is the spine of the project now and it replaced three earlier rules: the
text policy by surface, "cards are left alone", and "hands untouched, minimum
region everywhere". Two consequences are easy to miss:

- **The precision constraint moved from hands to the handled zone.** It used
  to be that the wearer's hand must never be masked, and that constraint is
  what rejected every setting that would have found more faces. The zone is
  in and subtracted from every mask, so the hand cannot be masked by any
  setting, and those settings can be re-measured. That is package F2, and
  nothing may be loosened outside it.
- **Over masking outside the zone stopped being a failure.** A wall sign the
  collector reads is text, and text outside the zone is to be masked. YuNet
  scoring a sign at 0.80 was a false positive under the old rule and is a
  correct find under this one. What still constrains masking outside the zone
  is the per frame mask budget, so the environment stays usable, not privacy.

Inside the zone, precision is absolute: nothing is ever masked there. Outside
it, a miss is the leak and over masking of non informational things is the
cheaper error.

H1 built the zone and found that "inside the zone" needs two definitions, not
one. What the wearer **reaches over** is what they are holding, so text and
screens there are part of the task. A face inside somebody's reach is usually
**their own face**, so a face is protected only where a hand is actually on
it. Before that split the zone was protecting bystanders' faces from
redaction, measured on 20 of 27 labelled cases. Report section 17.5.

## What this is

FaceBlur: video in, video with the sensitive things destroyed. A Windows
window plus a command line, running on the owner's PC, no telemetry. The
production run of 2026-10 is the one exception to "video never leaves the
machine": the owner chose to upload the footage to GreenNode and run it there
on rented GPUs, with `cloud/` as the runner.

Three kinds, each a switch of its own (`faceblur/classes.py`): **faces**
(built and measured), **screens** as objects (built and measured), **personal
text** (not built). Plates and bodies are out of scope. Audio and container
metadata were offered on 2026-09-10 and not chosen.

Faces are on by default today. Once text is ready, all three go on by default:
that is package D and the owner's D3.

No labels and no human step anywhere in the pipeline, and none in any
measurement that gets re-run. A by eye audit is allowed once, only to check
that a rule which sets a detection aside is safe, and **its labels are
committed under `docs/audits/`**. Two earlier audits did not commit their
labels and cannot be re-scored; see "should not assume".

## The footage

`footage/` holds four Ego camera files, 1600x1300, about 30 fps, h264, no
audio, 264 s in total. They, every output folder and `eval/cache/` are
gitignored. Never commit them.

**They are not card collector footage.** They are a washroom being cleaned, a
corridor being mopped, a canteen and a table tennis hall, and the wearer
appears to be a facilities worker. Measured, not guessed: the OWLv2 oracle
asked about trading cards, graded slabs and binder pages over 1587 frames
spread across all four files reports zero at any score of 0.2 or above,
against 3213 televisions on one file alone (report section 16.2.1).

Everything else in this file is consistent with that once you look for it: the
wearer's hand on a mop, the face in a washroom mirror, the blue table tennis
table that reads as a laptop.

The card collector dataset is still what the tool is for. No number in this
repository has ever been measured on a card.

## Where things stand

Everything is committed. `HashBrake/faceblur` `main` is at `c0f0dc0`; the
eight cloud commits after it are local only. Tests: `tests/`, 1285 passing,
about six minutes (`.venv\Scripts\python.exe -m pytest tests`).

The weekend build ends at `7dc74a7`. Anything after that in the log is a later
session and this file will say so.

### After the weekend

- **2026-09-14, `629d156`.** The whole pipeline on the whole sample, report
  24, and the crash safety defect it found.
- **2026-09-26, `46dbddf` and `adcc469`.** A run from the window with 10
  workers saved waited forever: a worker the machine killed is replaced by
  `multiprocessing.Pool` and the job it held never completes. `PoolSubmit`
  now raises `WorkerLost`, the window and the CLI fail that one video and
  start a fresh pool, and the worker count is capped at `max_workers()`.
  The window also gained "Recreate the whole folder": the folder tree is
  copied as it is, every `.mp4` at any depth replaced by its blurred copy
  under the same name, no quarantine (it would change names), records in
  `<folder>_faceblur_report.json`. `tests/manual_ui.md` has the steps.
- **2026-10-01.** The gate is crash safe; item 0 below is done.
- **2026-10-01.** The off face gates moved to the mask budget and three
  detection thresholds moved with them; items 1 and 2 below are done.
- **2026-10-05 to 08, `110288d` to `3558c42`.** The production run: the cloud
  runner, delivery and links, and three fixes to the tool. "The production
  run" above.

| Package | What | State |
|---|---|---|
| R1 | Documents and the packaged app say what the code does | Done, report 9.1 |
| S1 | Screens in the output side check and the gate | Done, report 16.1 |
| 4.3 | Per kind mask accounting | Done, report 16.3 |
| E1 | Object oracle for evaluation | Done, report 16.2 |
| R2 | Fixes from the review of that pass | Done, report 16.6 |
| H1 | The handled zone | Done, report 17 |
| H2 | Zone follow ups | Done, report 17.3 and 17.7 |
| R3 | The orientation table made re-runnable; 17.7 narrowed to what was measured | Done, report 17.7 |
| G1 | The gate in the window | Done, report 18 |
| F2 | Recall outside the zone | Done, report 19. Nothing moved, and the gates are the finding |
| F1 | Second chance for missed faces | Done, report 20 |
| T1 | Text detector | Done, report 21 |
| T3 | Text policy, gate, ready | **Blocked by T1's numbers.** 90 percent of what the detector finds here is not text |
| S2 | Screens outside the zone | Done, report 22. The floor did not move |
| D | Defaults: all three kinds on | Blocked with T3 |
| M1 | Metadata line | Done, report 23 |
| T4 | Identifiers by pattern | Optional, and blocked with T3 |
| ~~F0~~, ~~T2~~ | Faces on cards, card veto | **Dropped by the rule, not deferred. Do not build them** |

### What happened while the owner was away

**Seven packages, all seven committed and pushed.** Every one asked for in the
order R3, G1, F2, F1, T1, T3, S2, D, T4, M1 was either finished or is recorded
below as blocked with the numbers that blocked it. Tests went from 1084 to
1195. The working tree is clean and `main` is pushed.

The run started on the evening of 2026-09-11 and finished at 04:57 on
2026-09-12. **Nothing has run since**, so the repository the owner comes back
to is exactly what the last commit below describes; the two days after it are
idle and not a second pass.

| Commit | Package | Report |
|---|---|---|
| `e785147` | R3, the orientation table made re-runnable | 17.7 |
| `c94dfa7` | G1, the gate in the window | 18 |
| `8b22836` | F2, recall outside the zone | 19 |
| `457b2a4` | F1, a second chance for missed faces | 20 |
| `8e92e45` | T1, the text detector | 21 |
| `a80aac8` | S2, screens outside the zone | 22 |
| `ba4ebc4` | M1, the metadata line | 23 |

**Read section 19.5 first.** It is the one finding that is bigger than the
package it came out of, and it is about the build rather than about a knob.

#### What the owner should look at first

**1. The off face gates had no feasible point.** Report 19.5. *Answered
2026-10-01: the gates moved to the mask budget, report 25.*

**2. Text is blocked, and the reason is measurable.** Report 21.3. 124 quads
from the text detector were rendered as crops and labelled by eye: five are
on writing, eight contain writing, and 111 have no words in them at all. The
labels are committed. Nothing separates the two groups, the highest scoring
quad in the set is a ceiling light, and raising the threshold throws the real
ones away first. T3 and D are blocked behind it and T4 with them.

The deciding argument is not the 90 percent, and it is worth knowing which it
is. Under the rule a mask that wrong outside the zone could be defended: over
masking there is the cheaper error and 1 percent of a frame is well inside
the budget. What refuses it is that **there is no recall number for text and
this footage cannot give one**, because the four files hold about a dozen
readable words between them. A switch in the window labelled "personal text"
cannot ship without one, because the person who ticks it will believe the
text is gone. Report 21.5.

**3. Two copies that would have shipped yesterday did not.** Report 18. The
window never set `check_output`, so no run started from it was ever checked.
It is on by default now, and the first two sample files run through it were
both held back for a face.

#### Package by package

**R3.** `eval.zone --orientation` prints report 17.7's tables from the
committed audit labels, so the numbers come from a command rather than from a
throwaway script. The refutation of the orientation rule was also narrowed to
what the evidence supports: it fails on bystanders standing beside the
wearer, and nobody in this footage faces them.

**G1.** The window has the output side check and the quarantine gate, on when
it opens, with a "Checking" status word and a summary line that names the
kinds a copy was held back for. Run over two sample files, both copies were
held back, 8 faces on 8 frames and 13 on 12. The check costs 2.0 to 2.1 times
the rest of the run, which the help text says.

**F2.** The answer is no and nothing moved. Every setting section 10 rejected
is still refused by the hand gate: without verification, or with the second
detector scanning the whole frame, masks land on 0.39 percent of the wearer's
hand pixels on `004100` and 1.41 percent on `005035` against a limit of 0.1.
The zone does not make verification optional, because it protects a face only
where a hand is on it and only knows the hands its own palm detector found.
The 2560 scan buys 0.5 to 2.9 points of pasted face recall, costs 125 ms a
frame against 59, and moves the exposure proxy the wrong way on the longest
file.

Three harness faults were fixed first, or the numbers would have described a
build that does not ship: the evaluation harness did not know the zone
existed, `hand_damage` had become two numbers under one name, and the pasted
face set was answering for a detector it was not built with.

**F1.** `--second-chance N`, off by default. The faces the check found in the
copy go back in as detections, the file is written again from the source and
checked again. On `004100` the faces still visible went 8 to 1; on `004310`,
13 to 9. Both copies are still held back. The interesting part: the tracker
threw most of the evidence away, because `min_track` is 2 and a seed that
stands alone on one frame makes a track of one. A seed no track picks up is
now masked on its own frame with the run's own tail, and that is the
difference between 8 to 6 and 8 to 1.

Worth knowing: every leak `eval/reid` finds, in all four copies measured, is
on a box the mask reached. The faces the check finds and the sightings the
recogniser matches are almost disjoint populations on this footage, so F1
moved identifiability barely at all, 46 leaks to 43 on one file. What is left
is the mask itself, which is section 12's question.

**T1.** The detector is built, measured and not wired to anything. See above.

**S2.** The screen floor was asked again at 0.5, 0.4, 0.3 and 0.25 with the
zone on and it did not move. Precision against the oracle clears 50 percent
on one file of four at any floor. The masked share was never the problem: 1.03
percent is the worst of any file at any floor. The exception is `005035`,
where the oracle finds 529 screens and 0.4 takes recall from 22 to 37 percent
at 74 percent precision. That gain is real and it is not a default, because
the setting would have to know which room it is in.

**M1.** The record says whether the source had sound and lists the names of
its metadata tags, never the values. A clip carrying a location tag now goes
through the pipeline by both paths in a test, and the value appears in
neither copy.

#### What is blocked

- **T3**, text policy and gate, by T1's numbers. What would unblock it:
  footage with words in it, and a second model to agree with. Report 21.5.
- **D**, all three kinds on by default, because it turns on every ready kind
  and text is not one.
- **T4**, identifiers by pattern, which needs T3.

#### What to do next, in this order

0. ~~**The gate is not crash safe, and this is a shipping defect.**~~ **Fixed
   2026-10-01**: a checked copy waits as `<name>.mp4.unchecked` and moves
   once, after the gate. Report 24.4 says what changed. Found on
   2026-09-14 by a run the machine killed. `run_video` moves the finished
   copy into the output folder, then checks it, then moves it to
   `quarantine` if the check holds it back. A process that dies in that
   window leaves a complete, playable, ungated copy sitting in the output
   folder as though it had been delivered, and the only sign is a missing
   `.json` that nothing looks for. The fix is to keep the copy at its part
   path until the gate has spoken and then move it once, to the output folder
   or to quarantine. It was not done on the day because it changes the path
   every delivered file takes and the owner was back. Report 24.4.
1. ~~**Settle the gate question in 19.5.**~~ **Done 2026-10-01**: the owner
   moved the off face gates to the mask budget, read on the 95th percentile
   frame. Report 25.
2. ~~**Re-run `eval.combine` against the new gates.**~~ **Done 2026-10-01**,
   without a new sweep: gates change verdicts, not measurements. 24 of 54
   pass on all three files, the budget never binds, and the defaults moved
   three thresholds (`conf` 0.5, `verify_conf_sure` 0.3, `third_conf` 0.1)
   for 588 to 554 exposed frames. Report 25.
3. **Footage with cards in it, which only the owner can supply.** Five
   measurements are waiting on it: D1, the orientation rule, text recall, the
   text detector's false positive rate on a scene that has words in it, and
   what the zone costs when the thing being handled is the thing that
   matters.
4. **Then F1 again, or the mask.** F1 took the faces left visible in a copy
   from 8 to 1 and moved identifiability by 3 leaks in 519, because every
   leak measured is on a box the mask already reached. If the next thing
   wanted is fewer identifiable faces rather than fewer visible ones, the
   subject is the mask itself and section 12 is where that argument is.

## What the tool actually produces today

A full run of all four sample files on 2026-09-14, faces only, the zone on,
the check and the gate on, one second chance round. Report 24.

| File | Faces left, first pass | After the second chance | Largest left |
|---|---|---|---|
| `003939` washroom | 6 | 2 | 130 px |
| `004100` corridor | 8 | 1 | 47 px |
| `004310` canteen | 13 | 9 | 232 px |
| `005035` sports hall | 63 | 7 | 41 px |

**90 faces still visible becomes 19, and all four copies are held back.**
Nothing passes its own gate on this footage, which is the honest state of the
tool. No frame of any file was over the zone gate, so the one promise that is
absolute held everywhere. It costs about ten times real time with the check
and the second chance on, and the check is 58 percent of that.

## Decisions taken by default

Judgement calls made without asking, each reversible, each recorded where it
lives:

- **H1: a face is protected only where a hand is on it** (`zone_face_needs_hand`,
  on), at 1.6 times the palm box (`zone_face_scale`). The audit found the zone
  protecting bystanders' faces from redaction in 20 of 27 labelled cases, and
  this is the fix. **It reverses the owner's D1**: a printed face on a card
  held in the hand is now masked, because the card sits in the reach and not
  under the hand. That cannot be tested here, because there are no cards in
  this footage, and the trade was taken deliberately: masking a mass produced
  photograph on a card is a smaller harm than leaving a stranger's real face
  in a copy. `zone_face_needs_hand=False` gives D1 back. Report section 17.5.
  **Ratified 2026-09-11**: the owner delegated the call and it stands, pending
  H2 item 2, which is the only thing that may reopen it.
- **H1: the precision gate tests the hand, not the reach.** Once a face may be
  masked inside the reach on purpose, a gate measuring the reach reports the
  rule working as the rule broken. The gate measures where the promise is
  absolute. What the reach prevented is reported by the pipeline instead, as
  `masked_in_zone_prevented`. Section 17.6.
- **H1: `zone_stride` stays 1.** Stride 2 saves 18 percent of the detect phase
  and halves the hand evidence. That is a bad trade for the one promise that
  is absolute.
- **H2: `zone_orientation` was measured and not shipped.** The build plan asked
  for the wrist to knuckle direction as a third "whose hands" condition. Scored
  against the 179 committed labels the two distributions lie on top of each
  other: the wearer's hands run 1 to 96 degrees from straight up and a
  bystander's 7 to 105. At the proposed 100 degrees it drops 1 bystander of 20
  and nothing else; at 60 it drops 6 and costs 18 of the wearer's own hands.
  That refutes the hypothesis rather than the threshold, so no setting was
  added: a knob whose only honest default is "no effect" is worse than a note
  saying it was tried. `zone_face_needs_hand` is unchanged and the re-sweep
  that was conditional on this table was not run. Report 17.7.
- **H2: `zone_top_frac` ships at 0.15, not the 0.2 the numbers allow.** Both
  leave hands per frame and zone share identical on all four files and both
  reproduce the check exactly; 0.2 saves 40 percent of the hand pass and 0.15
  saves 22. At 0.2 the topmost window starts at y=512, so no palm in the top 39
  percent of the frame can be found at all, against the top 20 percent at 0.15.
  Nothing on this footage holds anything up to look at it and the collector
  footage this tool is for is people doing exactly that, so the blind region
  was worth more than the extra 18 percent. Report 17.7 has the table.

- **T1: PP-OCRv3 from OpenCV's zoo, not PP-OCRv4 converted here.** The rule
  was "no pre converted file from an unknown repository" and the zoo is where
  `yunet.onnx` came from. A conversion done on this machine would carry a
  hash only this machine could reproduce; the zoo's has a URL and a sha256
  anybody can check. `models/README.md` and report 21.1.
- **T1: `text_sizes` ships at 1920 and 2560**, not the plan's 960 and 1920.
  Measured on all four files: 70 percent more lines covering half the area,
  and the worst single quad on a frame drops from a fifth of the picture to
  a fourteenth. Report 21.2.
- **T1: the text kind's one line summary was rewritten.** It described the
  owner's first scope, personal text masked and a card's own price left
  alone, which the rule of 2026-09-11 replaced. It is user facing, in
  `--help` and the window.
- **T1: text is recorded as blocked rather than shipped half on.** A detector
  that is 90 percent wrong could be defended under the rule, because over
  masking outside the zone is the cheaper error and 1 percent of a frame is
  well inside the budget. It is still not worth 200 ms a frame to remove
  information one time in ten, and two of its false quads are faces the face
  pass already covers. Report 21.5.
- **F1: a seed no track picks up is masked on its own frame, with the run's
  own tail.** The spec said the seeds go through a second `Tracker.run` and
  they do, but `min_track` 2 drops a seed that stands alone and that is most
  of them. The rule exists so one spurious detection cannot become a mask,
  and a seed is not one: it is a box a second detection pass found in the
  finished copy, on pixels the first pass never changed, after the hand rule
  and the zone have both had a say. Measured: 8 to 6 through the tracker
  alone, 8 to 1 with the put back and the tail. Report 20.2.
- **F1: `second_chance` is off by default** and `Settings` refuses it without
  `check_output` and with a `check_stride` above 1, rather than doing nothing
  quietly. A stride means one frame in N was looked at, and seeding a tracker
  from that would put a mask where nothing was looked at.
- **F2: the defaults do not move**, and the gates were not touched to make
  them. Verification is refused by the hand gate on two files of three and
  2560 costs more than it returns, so the verdict holds under any reading of
  the off face gate. Changing the measure in the same pass that needs it
  changed is the move a reader should trust least. Report 19.5.
- **F2: `hand_damage` was renamed `hand_damage_wearer`.** A reader who kept
  using the old name would have got the wearer's number and thought it was
  everybody's, and on `004310` that is the difference between refusing a
  setting and permitting it: the settings that leave the wearer's hands alone
  put masks on 8.4 percent of everybody else's.
- **F2: the raw cache and the pasted face set are named by the detection
  fingerprint.** Four caches per video, sitting beside each other instead of
  overwriting each other, so a grid that compares `engine both` against a
  2560 scan does not re-detect the video on every step of itself. The legacy
  `<stem>.raw.pkl` is adopted by rename when its own fingerprint matches, so
  nothing was re-detected to change a file name.
- **F2: the sweep keeps what it has measured.** Four caches over the long
  file is hours and this PC killed the run three times for memory; the part
  file means a kill costs one cache and not the run. Detector processes
  default to two and the last cache needs one.

- **R2: the oracle's weight hash is a constant in `eval/oracle_owl.py`, with a
  test tying it to `models/README.md`.** The spec said to verify against the
  value in the README. Parsing markdown at run time to decide whether to trust
  a model is worse than one constant plus a test that the two agree, so the
  check reads the constant and `test_the_pinned_weights_are_the_ones_the_documentation_names`
  fails if the README ever drifts from it.
- **R2: a screen over the size cap stays invisible to the check.**
  `residual_job` calls the same `screens.detect` the pipeline calls, so the 12
  percent cap applies on the copy side too. That is scope rather than a miss,
  the spec asked for the behaviour to be kept, and it is now written down in
  report section 16.4 and the README rather than left to be rediscovered.

Nothing else has been decided by default. The owner's D1, D2 and D3 are
stated in the spec and are not open.

## The numbers

Faces, on copies made with faces alone, shipped defaults, output folder
`footage_blurred_v8`, 507 s for 264 s of video:

| Measure | `003939` | `004100` | `004310` | `005035` |
|---|---|---|---|---|
| Faces two detectors agree on, covered | n/a | 100.0 % | 99.7 % | 99.7 % |
| Faces of known position, pasted in, covered | n/a | 87.8 % | 78.7 % | 86.2 % |
| Masked pixels where no detector sees a face | n/a | 0.18 % | 0.77 % | 0.43 % |
| Hand pixels touched | n/a | 0.00 % | 0.00 % | 0.09 % |
| Confirmed faces kept covered while visible | n/a | 99.0 % | 99.5 % | 93.5 % |
| Faces the copy still shows, on untouched pixels | 7 | 8 | 15 | 64 |
| ... and hands the rule set aside, not counted above | 1 | 0 | 2 | 11 |

(`003939` has no MediaPipe oracle, so it has no consensus or hand numbers.)

With screens on as well, measured 2026-09-10 to 11, output folders
`footage_blurred_r1` and `footage_blurred_43`:

| Measure | `003939` | `004100` | `004310` | `005035` |
|---|---|---|---|---|
| Frame destroyed by the face mask, mean | 0.85 % | 0.51 % | 2.31 % | 0.69 % |
| Frame destroyed by the screen mask, mean | 0.05 % | 0.43 % | 0.01 % | 0.31 % |
| Frames over the 5 percent budget: union | 0 | 6 | 24 | 40 |
| ... of which the face mask alone | 0 | 6 | 23 | 9 |
| Screens the copy still shows | 11 | 14 | 10 | 129 |
| Runs of those long enough to hold the copy back | 1 | 3 | 0 | 7 |
| Held back for | face, screen | face, screen | face | face, screen |
| Screen recall against the oracle | 0 % | 2 % | 0 % | 23 % |

With the handled zone on, from the H1 run of 2026-09-11:

| Measure | `003939` | `004100` | `004310` | `005035` |
|---|---|---|---|---|
| Frames with a zone | 2033/2033 | 982/984 | 938/938 | 3971/3971 |
| Zone share of the frame, mean | 8.98 % | 7.17 % | 10.12 % | 10.68 % |
| ... max | 36.41 % | 34.13 % | 17.66 % | 34.84 % |
| Hands per frame | 1.62 | 1.59 | 2.62 | 1.15 |
| Mask the zone took back | 0.036 % | 0.008 % | 0.015 % | 0.035 % |
| Frames over the zone gate | 0 | 0 | 0 | 0 |
| Worst zone frame | 0.10 % | 0.10 % | 0.50 % | 0.30 % |

Over all four files together, **"whose hands" is wrong 12 percent of the
time**: 20 of the 173 polygons that could be called, out of 179 on 100
uniformly sampled frames. Report 17.7 and
`docs/audits/zone_hands_*.csv`.

The zone takes almost nothing back because the track level hand rules were
already keeping masks off hands. What it changes is that the protection is
structural rather than three thresholds holding, which is what unblocks F2.

If someone asks how good it is in one percentage, there is not one, and the
honest answer is three numbers: **99.7 to 100 percent** of the faces the
detectors agree on are covered; **79 to 88 percent** of faces planted at known
positions are covered, which is the hard test and the one to quote, at 83
percent for 64 px and over and 65 percent under 32 px; and hands are touched
**0.00 to 0.09 percent**. Per face sighting, not per person. Screens have no
comparable figure: 0 to 23 percent recall against one model's opinion.

On the legal side nothing sets a percentage. Both GDPR and Vietnam's Decree 13
ask whether a person can be identified by means reasonably likely to be used,
and the standard is a **human** who may already know them, not a recogniser.
Faces alone will not make this footage anonymous, because clothing, tattoos,
gait, interiors and timestamps remain. The defensible posture is to treat the
output as pseudonymised personal data and let the measurements support
"appropriate technical measures", not "no longer personal data". The audit
records and `docs/report.md` are what an auditor actually asks for.

## What a next person should not assume

**About the screens.**

- **The screen class misses most screens and holds most copies back for the
  ones it finds.** Recall against the oracle is 0 to 23 percent; three of four
  files are held back for a screen. It is fit for best effort masking with the
  gate off, not for unattended use, until S2. Report sections 16.1 and 16.2.
- **The size cap and the persistence rule are not what costs the recall.** The
  cap loses nothing the oracle calls a screen and persistence loses nine
  sightings on one file. `screen_conf` 0.5 is what loses the rest, and section
  15.1 measured what lowering it costs. S2 is where that trade gets decided.
- **A screen over the cap is invisible to the check too**, by design.
- **`screen_max_area` does not bound what is destroyed.** The cap is checked
  on the detector's box and `screen_pad` grows it afterwards, so a box at the
  12 percent cap is masked at up to about 13.4 percent of the frame.
- **The screen rules are tuned on one venue.** The table tennis hall is what
  makes the cap necessary and what sets its value.

**About the evidence.**

- **The by eye labels behind report sections 14.1 and 15.2 were never
  committed.** Only the counts survive. So the oracle's screen precision of 56
  to 68 percent and the eye's 91 percent cannot be reconciled by anybody, now
  or later, and neither audit can be checked or extended. H1's audit must not
  repeat this: commit the labels under `docs/audits/`.
- **The two face count tables above differ, and that is the copies, not the
  check.** Report section 16.1 gives 7, 9, 16 and 67 where the first table
  gives 7, 8, 15 and 64. The first reads copies made with faces alone, the
  second copies made with faces and screens, so a face detector is not looking
  at the same pixels.
- **Report section 13.4 reports 94 boxes on `005035` where the current code
  reports 75.** The other three files reproduce exactly. Nothing explained it,
  the 2026-09-08 run cannot be repeated, and the finding does not change:
  that file still shows dozens of faces.
- **The caches went stale once and nothing said so** (2026-09-08), and every
  number for a week described a build that no longer existed. Every cache now
  carries a fingerprint of the code that wrote it. Changing
  `faceblur/detect.py` at all, comments included, invalidates the raw caches
  by design.
- **The recogniser bounds one thing only**: what SFace can do at these
  distances. It works on `003939`, where people are at conversational
  distance, and is near useless across a room.
- **The exposure proxy brackets pairs of different faces too.** Read
  `exposed_40` and `exposed_24_40` as a difference between settings, not as a
  count of exposed faces.

**About the zone.**

- **One of twenty audited bystander faces is still protected** after the fix,
  and **three of seven of the wearer's own hands are now masked** where the
  zone used to protect them. Both are the same hard case from opposite sides:
  a face and a hand close enough together that no region separates them.
- **"Whose hands" is wrong 12 percent of the time** on a uniform sample of 179
  polygons, 20 of the 173 that could be called. The 74 percent figure from H1
  came from the faces the zone protected, a set selected for being failures.
  Quote the 12. Report 17.7.
- **The 44 of 52 hand rule agreement in 17.4 is not from the copies that
  ship.** On the shipped copies the same comparison reads 13 of 21, because
  tightening the face region masks more of the wearer's hands and fewer
  survive in the copy to be compared at all. Both are in 17.4; do not quote
  one against the other.
- **29 of the 56 faces the zone set aside were never examined.** They are in
  `docs/audits/zone_faces_set_aside_2026-09-11.csv` marked as such.
- **The remembered zone drifts** under camera rotation or zoom, because
  `motion.estimate_shift` measures a translation only. The memory is three
  seconds for that reason and the drift is not separately measured.
- **The check cannot rebuild the remembered zone**, because memory crosses
  frame range boundaries and each range runs in its own worker. Only the live
  half is checked.
- **The zone costs about 45 ms a frame** since H2 cut the window grid from 30
  to 24, down from about 60. It still roughly doubles the detect phase. A
  further cut to 18 windows is measured and available at `zone_top_frac` 0.2,
  and was not taken; see decisions taken by default.
- **Do not re-tile the window grid to make it smaller.** Skip rows and leave
  the rest where they are. Laying a smaller grid over what is left moves every
  window, and what the palm model makes of a crop depends on where the hand
  sits in it, so the zone changes rather than shrinks: the first attempt moved
  hands per frame by 44 percent on one file while saving the same time.

**About what is not built.**

- **The gate holds all four sample files, and should.** They show faces.
  Package F1 is the work that would change that.
- **Seven of the twenty hands still read as faces to the output side check**,
  six on the table tennis file: a fist gripping a bat, side on, in motion. H1
  should make this moot for masking, but the check's own hand rule stays as
  belt and braces.
- **No installer, no code signing.** The cloud runner in `cloud/` is built
  for GreenNode notebook instances and S3 storage; elsewhere it would need
  its start script changed.

## How to run

```
.venv\Scripts\python.exe -m ui.app                              # window
.venv\Scripts\python.exe cli.py footage -o out                  # faces
.venv\Scripts\python.exe cli.py footage -o out --mask face,screen
.venv\Scripts\python.exe cli.py footage -o out --quarantine     # with the gate
.venv\Scripts\python.exe -m pytest tests
.venv\Scripts\python.exe -m faceblur.verify VIDEO BLURRED --workers 4 --mask face,screen
.venv\Scripts\python.exe -m eval.misses VIDEO --images DIR      # where faces may still be missed
.venv\Scripts\python.exe -m eval.reid VIDEO BLURRED             # is anybody still identifiable
.venv\Scripts\python.exe -m eval.sweep VIDEO                    # choose settings, write the report
.venv\Scripts\python.exe -m eval.screens VIDEO                  # the screen class against the oracle
.venv\Scripts\python.exe -m eval.zone VIDEO --frames 40         # zone share, hands per frame, cost
.venv\Scripts\python.exe -m eval.zone VIDEO --copy COPY         # masked pixels inside MediaPipe's hands
.venv\Scripts\python.exe -m eval.zone VIDEO --audit DIR         # render frames, write the label CSV
.venv\Scripts\python.exe -m eval.zone --orientation            # report 17.7's tables from the labels
.venv\Scripts\python.exe cli.py footage -o out --no-zone        # measure what the zone costs
.venv-oracle\Scripts\python.exe eval\oracle_owl.py VIDEO --stride 5
dist\FaceBlur\FaceBlur.exe footage -o out --mask face,screen    # packaged, no Python needed
```

Three environments. `.venv` runs everything that ships. `.venv-eval` has
MediaPipe, for the third face detector and the hand regions. `.venv-oracle`
has torch and OWLv2. Three rather than two because MediaPipe pins numpy below
2 and every face number here was measured through MediaPipe; loosening that
pin to save disk would put the face side's evidence at risk.

`faceblur.verify` exits 1 when the gate would hold the copy back and 0
otherwise, and prints everything it found either way.

The handled zone runs on every run and costs about 60 ms a frame, roughly
doubling the detect phase: 966 s against 556 s over the four sample files.
`--no-zone` is for measuring that, never for a copy that ships. `eval.zone
--audit` writes its frames wherever you point it, which must be outside the
repository, and its labels to `docs/audits/`, which is committed.

`eval.oracle_owl` takes about 4.2 s a frame on the processor, so the four
files at stride 5 are about two hours. It resumes after a kill, checks the
sha256 of its weights before it starts, and its cache carries a fingerprint of
its own source.

Detection costs about 110 to 130 ms per frame on the RTX 3070 through
DirectML. The output side check costs about the same again. Do not raise
`detect.GPU_WORKERS` above four: each worker holds about 850 MB of graphs and
six reached 7.3 GB of the 8 and DirectML paged.

The desktop shortcut `FaceBlur.lnk` on this PC launches
`.venv\Scripts\pythonw.exe -m ui.app`, so it always runs the current code.
`gh` is not installed; git credential manager handles the push.

## How it got here

One line per pass. The report has each one in full and is the place to read
before changing anything a pass measured.

| When | What | Report |
|---|---|---|
| 2026-09-05 to 06 | First build, then the precision rebuild that followed it destroying 33 percent of every frame and 82 percent of hand pixels | `docs/precision_audit.md`, 3 to 9 |
| 2026-09-07 | Missed faces: confirmation crops, three detector families, camera aware tracking. Found that raising sensitivity masks the wearer's hand on a mop | 10 |
| 2026-09-08 | Faces lost during a hit: track stitching across a smear. Then identity measured directly with a recogniser, which also found the caches were stale | 11, 12 |
| 2026-09-08 to 09 | `faceblur/verify.py`, the check that reads the copy instead of the source, and the `--quarantine` gate | 13 |
| 2026-09-09 | The hand rule: two MediaPipe models tell the wearer's hand from a face in the check. The eye pass over 108 boxes corrected 13.4 | 14 |
| 2026-09-10 | Scope to three kinds, the kind registry, and the screen class. The detector failed its first run and two rules fixed it | 15 |
| 2026-09-10 to 11 | R1, S1, 4.3, E1: documents and build caught up, screens in the gate, per kind accounting, the object oracle | 16.1 to 16.3 |
| 2026-09-11 | The owner restated the goal as one rule. The spec was rewritten around it; F0 and T2 dropped | spec section 3 |
| 2026-09-11 | R2: review fixes, exit code follows the gate, rows carry their box, imports checked by `ast` | 16.6 |
| 2026-09-11 | H1: the handled zone. Three defects found by measurement, including the zone protecting bystanders' faces | 17 |
| 2026-09-11 | H2: the gate threshold fixed, the uniform sample labelled, the orientation rule refuted, the window grid cut | 17.3, 17.7 |
| 2026-09-11 to 12 | R3, G1, F2, F1, T1, S2, M1, unattended | 17.7 to 23 |
| 2026-09-14 | The whole pipeline on the whole sample; the gate was not crash safe | 24 |
| 2026-10-01 | The gate made crash safe; the gates moved to the mask budget | 24.4, 25 |
| 2026-10-05 to 08 | The production run: cloud image and runner, the check made faster, delivery and links | this file |

## Open questions for the owner

- **Text, and whether it is worth a second model.** T1 measured the detector
  and 90 percent of what it finds on this footage is not text: lights,
  railings, a hand on a mop, twice a face. The labels are committed and
  report 21.3 has the tables. T3, D and T4 are blocked behind it. Two things
  would change the answer and both need the owner: footage with words in it,
  and a decision about whether a second text model is worth the time, since
  one detector at one threshold is what section 10 rejected for faces.
- **Card footage.** The rule no longer depends on cards, but nothing about
  card text or printed faces can be measured until there is footage with cards
  in it. One file of a collector at work would do.
- **A screen someone else holds up to the wearer**, a phone shown across a
  table. By the rule that is handled and protected; by intent it may not be.
  The default protects it.
- **A second venue.** Every screen and text setting will be tuned on the table
  tennis hall and the canteen otherwise.
- **Disk.** The machine sits near full. The owner frees space; this session
  does not delete `footage_blurred*` folders or environments.

## Files that matter

| Path | What |
|---|---|
| `FACEBLUR_BUILD_PLAN_V2.md` | the spec being executed; where it and the code disagree, the code wins |
| `faceblur/settings.py` | every setting and its default, with the reason |
| `faceblur/detect.py` | YuNet, CenterFace, UltraFace; crops, views, confirmation |
| `faceblur/motion.py` | camera shift between frames |
| `faceblur/track.py` | tracker: confirmation, camera aware prediction, continuation, tails |
| `faceblur/redact.py` | mask shapes, pixel destruction, per kind shares. H1 adds the zone here |
| `faceblur/segments.py` | keyframe cuts, copies, encodes, concat, verify |
| `faceblur/batch.py` | the phased runner `run_video`, the worker side model caches |
| `faceblur/pipeline.py` | the audit record, single process path |
| `faceblur/classes.py` | the kinds, and which are ready |
| `faceblur/screens.py` | YOLOX, the size cap, the persistence rule |
| `faceblur/verify.py` | the check that reads the copy, `held_for`, the gate |
| `faceblur/hands.py` | palm detector and landmark model, for the output side check |
| `faceblur/zone.py` | the handled zone: whose hands, how far, how long remembered |
| `faceblur/text.py` | the text detector, built and measured, wired to nothing |
| `eval/zone.py` | zone share, cost, hand damage on a copy, the audit renderer |
| `cli.py`, `ui/app.py` | the two front ends |
| `eval/common.py` | the raw cache and its fingerprint |
| `eval/measure.py`, `eval/sweep.py` | the label free harness and the gates |
| `eval/combine.py` | one default across every file, which is where F2's answer came from |
| `eval/text.py` | what the text detector finds, the size sweep, the audit tables |
| `eval/reid.py` | the recogniser: what the mask does, what is left in the copy |
| `eval/oracle_owl.py` | the object oracle: OWLv2, pinned, never ships |
| `eval/screens.py` | the screen class scored against the oracle |
| `models/README.md` | every model: source, sha256, input layout, output layout |
| `docs/report.md` | every measurement, one section per pass |
| `docs/audits/` | by eye labels, committed. H1's are the first that can be re-scored |
