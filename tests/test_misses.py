"""The audit record's unconfirmed stretches, and the camera shift it reports."""
from __future__ import annotations

from faceblur.batch import run_video, serial_submit, unconfirmed_runs
from faceblur.detect import Detection
from faceblur.pipeline import STATUS_DONE
from faceblur.settings import Settings

LM = ((10.0, 10.0), (30.0, 10.0), (20.0, 20.0), (12.0, 30.0), (28.0, 30.0))


def det(x, score=0.8):
    return Detection(float(x), 100.0, 40.0, 40.0, score, LM)


def test_stretches_of_strong_unmasked_boxes_are_listed():
    weak = [[det(100)] for _ in range(10)]
    weak[4] = []                              # breaks the stretch into 4 and 5
    per_frame = [[] for _ in range(10)]
    runs, flagged = unconfirmed_runs(weak, per_frame, conf=0.6)
    assert runs == [[0, 3], [5, 9]]
    assert flagged == 9


def test_a_masked_box_is_not_unconfirmed():
    weak = [[det(100)] for _ in range(5)]
    per_frame = [[det(102)] for _ in range(5)]        # the mask sits on it
    assert unconfirmed_runs(weak, per_frame, conf=0.6) == ([], 0)


def test_weak_boxes_below_the_threshold_do_not_count():
    weak = [[det(100, score=0.45)] for _ in range(5)]
    assert unconfirmed_runs(weak, [[] for _ in range(5)], conf=0.6) == ([], 0)


def test_short_stretches_are_dropped_but_still_counted():
    weak = [[det(100)], [det(100)], [], [], []]
    runs, flagged = unconfirmed_runs(weak, [[] for _ in range(5)], conf=0.6, min_run=3)
    assert runs == [] and flagged == 2


def test_the_record_carries_the_new_fields(video_silent, tmp_path):
    settings = Settings(det_sizes=(320,), verify=False, min_track=1, tail=1, chunk_seconds=0.5,
                        min_copy_seconds=0.3)
    record = run_video(video_silent, tmp_path / "out.mp4", settings, serial_submit)
    assert record.status == STATUS_DONE, record.error
    d = record.to_dict()
    assert "unconfirmed_runs" in d and "camera_shift_p95" in d
    assert d["unconfirmed_runs"] == []
    assert d["camera_shift_p95"] < 2.0          # the test pattern does not move


def test_default_workers_is_half_the_cores_capped_on_a_gpu():
    from faceblur.detect import GPU_WORKERS, default_workers, providers_for
    assert default_workers(cpu_count=4, device="cpu") == 2
    assert default_workers(cpu_count=40, device="cpu") == 20
    if providers_for("auto")[0] != "CPUExecutionProvider":
        assert default_workers(cpu_count=40) == GPU_WORKERS
