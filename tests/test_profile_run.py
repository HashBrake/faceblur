"""The profile driver's grouping and tables, on fake stats. No footage, no GPU."""
from __future__ import annotations

from eval import profile_run as prof

ORT = ("C:/venv/Lib/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py",
       278, "run")
READ = ("~", 0, "<method 'read' of '_io.BufferedReader' objects>")
WRITE = ("~", 0, "<method 'write' of '_io.BufferedWriter' objects>")
RESIZE = ("~", 0, "<resize>")
ASTYPE = ("~", 0, "<method 'astype' of 'numpy.ndarray' objects>")
DECODE = ("D:/x/faceblur/segments.py", 203, "decode_range")
ENC_WRITE = ("D:/x/faceblur/segments.py", 261, "write")
MATCH = ("D:/x/faceblur/track.py", 459, "_match")


def stat(tt, ct=None, nc=1, callers=None):
    return (nc, nc, tt, tt if ct is None else ct, callers or {})


def test_categories():
    assert prof.category(ORT) == "gpu"
    assert prof.category(RESIZE) == "cv2"
    assert prof.category(("~", 0, "<blobFromImage>")) == "cv2"
    assert prof.category(ASTYPE) == "numpy"
    assert prof.category(MATCH) == "python"
    assert prof.category(READ, DECODE) == "decode"
    assert prof.category(WRITE, ENC_WRITE) == "encode"
    assert prof.category(READ, MATCH) == "subprocess"
    assert prof.category(("~", 0, "<built-in method _winapi.WaitForSingleObject>")) == "subprocess"


def test_split_sums_to_the_total_and_splits_pipe_reads_by_caller():
    stats = {
        ORT: stat(5.0),
        RESIZE: stat(1.0),
        MATCH: stat(0.5),
        READ: stat(3.0, callers={DECODE: (2, 2, 2.0, 2.0),
                                 ENC_WRITE: (1, 1, 1.0, 1.0)}),
    }
    got = prof.split(stats)
    assert got["gpu"] == 5.0 and got["cv2"] == 1.0 and got["python"] == 0.5
    assert got["decode"] == 2.0 and got["encode"] == 1.0
    assert sum(got.values()) == prof.total(stats) == 9.5


def test_hot_spots_rank_by_own_time_with_share_of_phase():
    phases = {"detect": {ORT: stat(6.0), RESIZE: stat(2.0)},
              "check": {MATCH: stat(4.0), ASTYPE: stat(1.0)}}
    rows = prof.hot_spots(phases, {"detect": 10.0}, n=3)
    assert [r["tottime"] for r in rows] == [6.0, 4.0, 2.0]
    assert rows[0]["share"] == 0.6 and rows[0]["kind"] == "gpu"
    # No wall for the check: the share falls back to its profiled total.
    assert rows[1]["share"] == 0.8 and rows[1]["phase"] == "check"


def test_tables_name_every_phase_and_the_timers():
    phases = {"detect": {ORT: stat(6.0), RESIZE: stat(2.0)}, "other": {MATCH: stat(0.1)}}
    timers = {("detect", "(phase wall)"): [1, 10.0],
              ("detect", "InferenceSession.run"): [30, 6.5],
              ("detect", "decode_range"): [1, 1.5]}
    text = prof.tables(phases, timers, {**phases["detect"], **phases["other"]}, 12.0)
    assert "| detect | 10.0 | 6.5 | 65% |" in text
    assert "| other | 2.0 | 0.0 | 0% |" in text
    assert "| decode_range | 1 | 1.5 | 15% |" in text
    assert "Top 10 by own time" in text and "Top 40 by cumulative" in text


def test_timers_file_generators_under_the_phase_that_made_them():
    inst = prof.Instruments()

    def frames():
        yield from range(3)

    timed = inst.timed("decode_range", frames)
    job = inst.phase("encode", lambda: sum(timed()))
    assert job() == 3
    assert inst.timers[("encode", "decode_range")][0] == 1
    assert inst.timers[("encode", "(phase wall)")][0] == 1
    assert inst.stack == ["other"]


def test_the_profiled_run_is_the_production_configuration():
    s = prof.production_settings()
    assert s.check_output and s.quarantine and s.second_chance == 1 and s.check_stride == 1


def test_the_check_zone_crop_is_timed_not_the_old_alpha_for():
    # The check crops the zone with verify.zone_crop since T03.
    assert ("faceblur.verify", "zone_crop") in prof.PARTS
    assert ("faceblur.zone", "alpha_for") not in prof.PARTS
