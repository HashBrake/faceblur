"""The raw cache rebuilds itself when the detector it came from has moved."""
from __future__ import annotations

import pickle

import pytest

from eval import common
from faceblur.settings import Settings


def test_the_fingerprint_follows_the_settings_that_decide_what_is_cached():
    base = Settings(verify=True)
    assert common.fingerprint(base) == common.fingerprint(Settings(verify=True))
    assert common.fingerprint(base) != common.fingerprint(base.with_changes(det_sizes=(1280,)))
    assert common.fingerprint(base) != common.fingerprint(base.with_changes(crop_scale=3.0))
    # A tracking setting is applied to the cache afterwards, so it must not
    # throw the evidence away.
    assert common.fingerprint(base) == common.fingerprint(base.with_changes(min_track=5))


def test_the_fingerprint_follows_the_detection_code(tmp_path, monkeypatch):
    src = tmp_path / "detect.py"
    src.write_text("# one", encoding="utf-8")
    monkeypatch.setattr(common, "DETECT_SRC", src)
    before = common.fingerprint(Settings())
    src.write_text("# two", encoding="utf-8")
    assert common.fingerprint(Settings()) != before


class _Capture:
    def __init__(self, path):
        pass

    def get(self, prop):
        return 4.0

    def release(self):
        pass


class _Pool:
    def __init__(self, *a, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def imap_unordered(self, fn, args):
        yield {0: {"yunet": []}}, {0: (0.0, 0.0)}


class _Context:
    Pool = _Pool


@pytest.fixture
def offline(tmp_path, monkeypatch):
    """build_cache without a video, a GPU or a pool."""
    path = tmp_path / "v.raw.pkl"
    monkeypatch.setattr(common, "cache_path", lambda video, settings=None: path)
    monkeypatch.setattr(common.cv2, "VideoCapture", _Capture)
    monkeypatch.setattr(common.multiprocessing, "get_context", lambda name: _Context())
    return path


def test_a_cache_from_a_detector_that_has_moved_is_rebuilt(tmp_path, offline):
    """The failure this guards: on 2026-09-08 the caches were built halfway
    through a pass that changed the crop rule, and every number after that
    described a build that no longer existed."""
    offline.write_bytes(pickle.dumps({"version": common.CACHE_VERSION, "fingerprint": "stale",
                                      "video": "v.mp4", "shape": (4, 4),
                                      "frames": {0: {"yunet": ["ghost"]}}, "shifts": {}}))
    cache = common.build_cache(tmp_path / "v.mp4", workers=1)
    assert cache["fingerprint"] == common.fingerprint(Settings(verify=True))
    assert cache["frames"][0] == {"yunet": []}


def test_a_cache_that_matches_is_kept(tmp_path, offline):
    stamp = common.fingerprint(Settings(verify=True))
    offline.write_bytes(pickle.dumps({"version": common.CACHE_VERSION, "fingerprint": stamp,
                                      "video": "v.mp4", "shape": (4, 4),
                                      "frames": {0: {"yunet": ["kept"]}}, "shifts": {}}))
    cache = common.build_cache(tmp_path / "v.mp4", workers=1)
    assert cache["frames"][0] == {"yunet": ["kept"]}


def test_a_synthetic_set_cut_from_labels_that_moved_is_rebuilt(tmp_path, monkeypatch):
    """Same guard as the raw cache, one step further down the chain."""
    from eval import synthetic

    path = tmp_path / "v.synthetic.pkl"
    monkeypatch.setattr(synthetic, "data_path", lambda video, settings=None: path)
    monkeypatch.setattr(synthetic, "face_bank", lambda video, consensus: ["a face"])
    monkeypatch.setattr(synthetic.multiprocessing, "get_context", lambda name: _Context())
    consensus = {"frames": {"0": [], "10": []}}
    path.write_bytes(pickle.dumps({"version": synthetic.VERSION, "stamp": "stale",
                                   "data": ["ghost"]}))
    assert synthetic.generate(tmp_path / "v.mp4", consensus, workers=1) != ["ghost"]
    held = pickle.loads(path.read_bytes())
    assert held["stamp"] == synthetic.stamp_of(consensus)
    assert synthetic.load(tmp_path / "v.mp4") == held["data"]


def test_a_synthetic_set_that_matches_its_labels_is_kept(tmp_path, monkeypatch):
    from eval import synthetic

    path = tmp_path / "v.synthetic.pkl"
    monkeypatch.setattr(synthetic, "data_path", lambda video, settings=None: path)
    consensus = {"frames": {"0": []}}
    path.write_bytes(pickle.dumps({"version": synthetic.VERSION,
                                   "stamp": synthetic.stamp_of(consensus),
                                   "data": ["kept"]}))
    assert synthetic.generate(tmp_path / "v.mp4", consensus, workers=1) == ["kept"]


def test_two_detection_settings_get_two_caches(tmp_path, monkeypatch):
    """Package F2 compares settings that change what is detected.

    `engine both` scans the whole frame with the second detector and a 2560
    scan adds a size, and neither can be filtered out of a cache built
    without it. One file per video would make the sweep re-detect the video
    on every step of its own grid.
    """
    monkeypatch.setattr(common, "CACHE_DIR", tmp_path)
    video = tmp_path / "v.mp4"
    plain = common.cache_path(video, Settings())
    wider = common.cache_path(video, Settings(det_sizes=(1280, 1920, 2560)))
    both = common.cache_path(video, Settings(engine="both"))
    assert len({plain, wider, both}) == 3
    # `verify` rides on the cache built with confirmations rather than
    # needing its own, so the sweep does not build one for it.
    assert common.cache_path(video, Settings()) == plain


def test_the_pasted_face_set_is_named_by_the_detector_that_built_it(tmp_path, monkeypatch):
    """It holds the detector's answers on every pasted frame, so a set built
    at 1280 and 1920 cannot score a 2560 scan. It would answer, and the
    answer would be the old detector's."""
    from eval import synthetic

    monkeypatch.setattr(synthetic, "CACHE_DIR", tmp_path)
    video = tmp_path / "v.mp4"
    plain = synthetic.data_path(video, Settings())
    wider = synthetic.data_path(video, Settings(det_sizes=(1280, 1920, 2560)))
    assert plain != wider
    assert synthetic.data_path(video) == plain


def test_a_cache_from_the_old_single_name_is_adopted_and_not_rebuilt(tmp_path, monkeypatch):
    """Every raw cache before 2026-09-12 was `<stem>.raw.pkl` whatever it
    held. Re-detecting four videos to change a file name would be an hour of
    GPU time for nothing."""
    monkeypatch.setattr(common, "CACHE_DIR", tmp_path)
    settings = Settings(verify=True)
    video = tmp_path / "v.mp4"
    legacy = tmp_path / "v.raw.pkl"
    legacy.write_bytes(pickle.dumps({"version": common.CACHE_VERSION,
                                     "fingerprint": common.fingerprint(settings),
                                     "video": "v.mp4", "shape": (4, 4),
                                     "frames": {0: {"yunet": ["kept"]}}, "shifts": {}}))
    common.adopt_legacy_cache(video, settings)
    assert not legacy.exists()
    held = pickle.loads(common.cache_path(video, settings).read_bytes())
    assert held["frames"][0] == {"yunet": ["kept"]}


def test_a_legacy_cache_that_does_not_match_is_left_where_it_is(tmp_path, monkeypatch):
    """Deleting a file it was not asked about is not this function's business,
    and `build_cache` will not read it."""
    monkeypatch.setattr(common, "CACHE_DIR", tmp_path)
    settings = Settings(verify=True)
    video = tmp_path / "v.mp4"
    legacy = tmp_path / "v.raw.pkl"
    legacy.write_bytes(pickle.dumps({"version": common.CACHE_VERSION,
                                     "fingerprint": "from a detector that has moved",
                                     "video": "v.mp4", "shape": (4, 4),
                                     "frames": {}, "shifts": {}}))
    common.adopt_legacy_cache(video, settings)
    assert legacy.exists()
    assert not common.cache_path(video, settings).exists()
