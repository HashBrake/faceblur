# FaceBlur

FaceBlur blurs every face in a video. It runs on your PC. Your video never leaves
the machine.

This repo is under construction. See `FACEBLUR_BUILD_PLAN.md` for the full plan
and `docs/` for the phase records.

## Set up

You need Python 3.12 on Windows.

```
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the tests

```
.venv\Scripts\python.exe -m pytest tests/
```

## Audio

FaceBlur does not change audio. It copies the audio track from the source video
into the blurred copy, untouched.
