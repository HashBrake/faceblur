"""The window, driven through a real batch in real worker processes.

These tests start worker processes and encode video, so they are slower than the
rest of the suite. They cover the path that is easiest to break: the pool, the
progress queue, the cancel event and the deletion of unfinished files.
"""
from __future__ import annotations

import shutil
import time

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from ui import strings as S  # noqa: E402
from ui.app import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def app():
    existing = QApplication.instance()
    yield existing or QApplication([])


@pytest.fixture
def source_folder(tmp_path, video_silent):
    folder = tmp_path / "src"
    folder.mkdir()
    for name in ("clip1.mp4", "clip2.mp4", "clip3.mp4"):
        shutil.copy(video_silent, folder / name)
    return folder


def make_window(app, source, output, workers=2):
    window = MainWindow()
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.show()
    window.settings_store.clear()
    window._take_paths([source])
    window.output_dir = output
    window.output_value.setText(str(output))
    window.workers_spin.setValue(workers)
    window.stride_spin.setValue(3)
    window._update_start_enabled()
    return window


def run_to_end(app, window, seconds=240, stop_after=None):
    state = {"done": False, "stopped": None, "stages": set(), "records": 0}

    original_all = window._on_all_finished
    original_one = window._on_one_finished
    original_progress = window._on_progress

    def on_all(stopped):
        original_all(stopped)
        state["done"] = True
        state["stopped"] = stopped

    def on_one(index, record):
        original_one(index, record)
        state["records"] += 1
        if stop_after is not None and state["records"] >= stop_after and window.runner:
            window.runner.stop()

    def on_progress(index, stage, done, total):
        original_progress(index, stage, done, total)
        state["stages"].add(stage)

    window._start()
    window.runner.all_finished.disconnect()
    window.runner.one_finished.disconnect()
    window.runner.progress.disconnect()
    window.runner.all_finished.connect(on_all)
    window.runner.one_finished.connect(on_one)
    window.runner.progress.connect(on_progress)

    deadline = time.time() + seconds
    while time.time() < deadline and not state["done"]:
        app.processEvents()
        time.sleep(0.02)
    assert state["done"], "the run never finished"
    return state


def test_the_drop_zone_counts_what_it_was_given(app, source_folder, tmp_path):
    window = make_window(app, source_folder, tmp_path / "out")
    assert "3 videos" in window.drop_zone.title.text()
    assert window.start_button.isEnabled()


def test_start_stays_disabled_until_both_choices_exist(app, source_folder, tmp_path):
    window = MainWindow()
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.show()
    window.settings_store.clear()
    window.output_dir = None
    window.inputs = []
    window._update_start_enabled()
    assert not window.start_button.isEnabled()
    assert window.start_help.isVisible()


def test_a_full_run_writes_every_copy(app, source_folder, tmp_path):
    output = tmp_path / "out"
    window = make_window(app, source_folder, output)
    state = run_to_end(app, window)

    assert state["stopped"] is False
    assert {"detecting", "writing"} <= state["stages"]
    assert sorted(p.name for p in output.glob("*.mp4")) == [
        "clip1_blurred.mp4", "clip2_blurred.mp4", "clip3_blurred.mp4"]
    assert len(list(output.glob("*.mp4.json"))) == 3
    assert window.summary_label.text() == S.SUMMARY.format(done=3, failed=0)
    assert window.overall_label.text() == S.OVERALL_FINISHED.format(done=3, total=3)
    assert all(S.STATUS_DONE in row.status.text() for row in window.rows)


def test_a_second_run_marks_every_row_already_done(app, source_folder, tmp_path):
    output = tmp_path / "out"
    run_to_end(app, make_window(app, source_folder, output))

    window = make_window(app, source_folder, output)
    run_to_end(app, window)

    assert all(S.STATUS_SKIPPED in row.status.text() for row in window.rows)
    # Reporting zero done without saying why would read as a failure.
    assert "already done" in window.summary_label.text()


def test_stop_deletes_unfinished_files(app, source_folder, tmp_path, video_long):
    folder = tmp_path / "longer"
    folder.mkdir()
    for name in ("a.mp4", "b.mp4", "c.mp4", "d.mp4"):
        shutil.copy(video_long, folder / name)
    output = tmp_path / "out_stop"

    window = make_window(app, folder, output, workers=2)
    state = run_to_end(app, window, stop_after=1)

    assert state["stopped"] is True
    finished = [r for r in window.rows if S.STATUS_DONE in r.status.text()]
    assert len(list(output.glob("*.mp4"))) == len(finished), \
        "the output folder holds a file for a row that never finished"
    assert any(S.STATUS_STOPPED in r.status.text() for r in window.rows)
    # The overall count never claims a stopped row as done.
    assert window.overall_label.text() == S.OVERALL_FINISHED.format(
        done=len(finished), total=len(window.rows))


def test_the_window_keeps_its_settings(app, source_folder, tmp_path):
    window = make_window(app, source_folder, tmp_path / "out")
    window.engine_max.setChecked(True)
    window.mode_solid.setChecked(True)
    window.stride_spin.setValue(4)
    window.replace_check.setChecked(True)
    window._remember()

    again = MainWindow()
    again.setAttribute(Qt.WA_DontShowOnScreen, True)
    again.show()
    assert again.engine_max.isChecked()
    assert again.mode_solid.isChecked()
    assert again.stride_spin.value() == 4
    assert again.replace_check.isChecked()
    again.settings_store.clear()


# ------------------------------------------------- the switches for each kind

def test_the_window_lists_every_kind_and_enables_only_the_built_ones(app):
    """A person deciding whether to trust the output needs to see that text
    and screens are not covered. A list that grew later would hide that."""
    from faceblur.classes import KINDS

    window = MainWindow()
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    try:
        assert list(window.mask_checks) == [k.name for k in KINDS]
        for kind in KINDS:
            check = window.mask_checks[kind.name]
            assert check.isEnabled() is kind.ready, kind.name
            if not kind.ready:
                assert not check.isChecked(), f"{kind.name} must not start ticked"
    finally:
        window.close()


def test_faces_start_ticked_and_unticking_them_stops_the_run(app, source_folder,
                                                             tmp_path):
    """Nothing to mask is not a run. The button goes dead rather than writing
    a copy that is the same as the source."""
    window = make_window(app, source_folder, tmp_path / "out")
    try:
        assert window._mask() == ("face",)
        assert window.start_button.isEnabled()

        window.mask_checks["face"].setChecked(False)
        assert window._mask() == ()
        assert not window.start_button.isEnabled()
        assert window.mask_help.isVisible()

        window.mask_checks["face"].setChecked(True)
        assert window.start_button.isEnabled()
    finally:
        window.close()


def test_a_kind_with_no_detector_cannot_be_switched_on(app):
    """Not by clicking, and not by a stored setting from a later build."""
    window = MainWindow()
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    try:
        window.settings_store.setValue("mask_text", True)
        window._restore()
        assert not window.mask_checks["text"].isChecked()
        assert "text" not in window._mask()
    finally:
        window.settings_store.remove("mask_text")
        window.close()
