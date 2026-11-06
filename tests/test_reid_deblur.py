"""Does the face mask survive an attempt to undo it?

`eval.reid.mask_curve` asks whether a recogniser can still name a masked face.
That is the easy question: nobody attacking a published video looks at the
mask as it is. They try to sharpen it back first. Gaussian blur is a linear
filter, and a deconvolution that knows (or guesses) the kernel recovers part
of what it took, so the honest test is the recogniser after a deblur attempt.

Two attacks, both on the box the mask was drawn in (the region bounds plus
the feather, as `redact._cover_region` cuts it):

- **deconvolve**: upsample x2 (cubic), unsharp mask, ten rounds of
  Richardson-Lucy against a Gaussian PSF of half the block size the mask used,
  downsample back. The attacker is handed the exact block size, the strongest
  guess there is.
- **sharpen**: upsample x2, edge-preserving bilateral smoothing, unsharp mask,
  downsample back. What a casual "enhance" button does.

Decision D-C hangs on the result: the face fill stays the user's mode only if
the shipped mode leaves no face matching itself after either attack;
otherwise faces are filled solid (`Settings.face_fill`). The last test
enforces that the shipped default passes, so the decision cannot drift.

    .venv\\Scripts\\python.exe -m pytest tests/test_reid_deblur.py -s
"""
from __future__ import annotations

import json
import math
from collections import defaultdict

import cv2
import numpy as np
import pytest

from eval.common import CACHE_DIR, REPO, build_cache, cache_path
from eval.reid import (COSINE_THRESHOLD, PAIR_SAMPLE, Recogniser, best_match,
                       cosine, sightings, walk)
from faceblur.detect import Detection
from faceblur.redact import fill_mode, redact, region_for
from faceblur.settings import Settings, SettingsError

SAMPLE = REPO / "footage" / "ego_AZER76400FE_19700101_004100_camera_left_part0001.mp4"
REID_JSON = CACHE_DIR / f"{SAMPLE.stem}.reid.json"

# The masks compared. face_fill="mode" so each row is the mode it names,
# whatever the shipped default of face_fill is.
MASKS = {
    "blur 6": Settings(mode="blur", strength=6, face_fill="mode"),
    "pixelate 6": Settings(mode="pixelate", strength=6, face_fill="mode"),
    "solid": Settings(mode="solid", face_fill="mode"),
}

# Pass criterion: at most this share of the faces still matches after an attack.
MAX_MATCHED_SHARE = 0.01


# ---------------------------------------------------------------- the attacks

def _attack_box(frame: np.ndarray, det: Detection, settings: Settings):
    """The box the mask was drawn in, and the block size it used."""
    e = region_for(det, settings)
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = e.bounds(settings.feather * 2)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    k = max(2, int(round(2 * e.block_scale() / settings.strength)))
    return (x0, y0, x1, y1), k


def _gaussian_psf(sigma: float) -> np.ndarray:
    half = max(1, int(math.ceil(3 * sigma)))
    g = cv2.getGaussianKernel(2 * half + 1, sigma, cv2.CV_32F)
    return (g @ g.T).astype(np.float32)


def _unsharp(img: np.ndarray, amount: float = 1.5, sigma: float = 3.0) -> np.ndarray:
    """img + amount * (img - blur(img)), float32."""
    return img + amount * (img - cv2.GaussianBlur(img, (0, 0), sigma))


def _richardson_lucy(img: np.ndarray, psf: np.ndarray, iterations: int = 10,
                     eps: float = 1e-6) -> np.ndarray:
    """Classic RL in float32 on [0, 1]; the PSF is symmetric so it is its own flip."""
    est = img.copy()
    flipped = psf[::-1, ::-1].copy()
    for _ in range(iterations):
        conv = cv2.filter2D(est, -1, psf, borderType=cv2.BORDER_REFLECT)
        ratio = img / (conv + eps)
        est = np.clip(est * cv2.filter2D(ratio, -1, flipped, borderType=cv2.BORDER_REFLECT),
                      0.0, 1.0)
    return est


def deblur(masked: np.ndarray, det: Detection, settings: Settings) -> np.ndarray:
    """Upsample, unsharp, Richardson-Lucy at the mask's own scale, downsample."""
    (x0, y0, x1, y1), k = _attack_box(masked, det, settings)
    out = masked.copy()
    if x1 <= x0 or y1 <= y0:
        return out
    roi = masked[y0:y1, x0:x1]
    rh, rw = roi.shape[:2]
    up = cv2.resize(roi, (2 * rw, 2 * rh), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    up = np.clip(_unsharp(up), 0, 255) / 255.0
    # The mask blurred at sigma k/2 in frame pixels, which is k on the x2 grid.
    up = _richardson_lucy(up, _gaussian_psf(k))
    back = cv2.resize(up * 255.0, (rw, rh), interpolation=cv2.INTER_AREA)
    out[y0:y1, x0:x1] = np.clip(back + 0.5, 0, 255).astype(np.uint8)
    return out


def sharpen(masked: np.ndarray, det: Detection, settings: Settings) -> np.ndarray:
    """Upsample, bilateral, unsharp, downsample: no knowledge of the mask."""
    (x0, y0, x1, y1), _ = _attack_box(masked, det, settings)
    out = masked.copy()
    if x1 <= x0 or y1 <= y0:
        return out
    roi = masked[y0:y1, x0:x1]
    rh, rw = roi.shape[:2]
    up = cv2.resize(roi, (2 * rw, 2 * rh), interpolation=cv2.INTER_CUBIC)
    up = cv2.bilateralFilter(up, 9, 40, 5).astype(np.float32)
    up = np.clip(_unsharp(up), 0, 255)
    back = cv2.resize(up, (rw, rh), interpolation=cv2.INTER_AREA)
    out[y0:y1, x0:x1] = np.clip(back + 0.5, 0, 255).astype(np.uint8)
    return out


ATTACKS = {"deconvolve": deblur, "sharpen": sharpen}


# ---------------------------------------------------------------- cheap checks

def _synthetic_face():
    frame = np.full((200, 200, 3), 120, np.uint8)
    cv2.circle(frame, (100, 100), 30, (40, 200, 90), -1)
    det = Detection(70, 70, 60, 60, 0.9,
                    landmarks=((88, 92), (112, 92), (100, 104), (90, 116), (110, 116)))
    return frame, det


@pytest.mark.parametrize("attack", list(ATTACKS.values()), ids=list(ATTACKS))
def test_an_attack_only_touches_the_masked_box(attack):
    frame, det = _synthetic_face()
    masked, _ = redact(frame, [det], MASKS["blur 6"])
    out = attack(masked, det, MASKS["blur 6"])
    (x0, y0, x1, y1), _ = _attack_box(masked, det, MASKS["blur 6"])
    assert out.shape == masked.shape and out.dtype == np.uint8
    outside = np.ones(masked.shape[:2], bool)
    outside[y0:y1, x0:x1] = False
    assert np.array_equal(out[outside], masked[outside])
    assert not np.array_equal(out[y0:y1, x0:x1], masked[y0:y1, x0:x1])


def test_richardson_lucy_undoes_part_of_a_known_blur():
    """The attack is a real one: on a blurred edge it gets closer to the truth."""
    truth = np.zeros((64, 64), np.float32)
    truth[:, 32:] = 1.0
    psf = _gaussian_psf(2.0)
    blurred = cv2.filter2D(truth, -1, psf, borderType=cv2.BORDER_REFLECT)
    restored = _richardson_lucy(blurred, psf)
    assert np.abs(restored - truth).mean() < np.abs(blurred - truth).mean()


def test_face_fill_is_validated_and_recorded():
    with pytest.raises(SettingsError):
        Settings(face_fill="blur")
    assert Settings(face_fill="mode").to_dict()["face_fill"] == "mode"
    assert "face_fill" in Settings().to_dict()


def test_face_fill_solid_fills_every_face_solid_and_mode_follows_the_mode():
    assert fill_mode("face", Settings(mode="blur", face_fill="solid")) == "solid"
    assert fill_mode("face", Settings(mode="pixelate", face_fill="solid")) == "solid"
    assert fill_mode("face", Settings(mode="pixelate", face_fill="mode")) == "pixelate"
    # Only faces: a screen still follows the run's mode.
    assert fill_mode("screen", Settings(mode="pixelate", face_fill="solid")) == "pixelate"
    assert fill_mode("text", Settings(mode="blur", face_fill="mode")) == "solid"


# ---------------------------------------------------------------- the measurement

def _threshold() -> float:
    if REID_JSON.is_file():
        calib = json.loads(REID_JSON.read_text()).get("calibration", {})
        if "threshold_fmr1" in calib:
            return float(calib["threshold_fmr1"])
    return COSINE_THRESHOLD


def _select(seen: dict, gallery_px: int, limit: int):
    """`mask_curve`'s selection: the largest sightings, with their track."""
    rows = [(tid, i, d) for tid, rows in seen.items() for i, d in rows
            if d.long_side >= gallery_px]
    return sorted(rows, key=lambda r: -r[2].long_side)[:limit]


@pytest.fixture(scope="module")
def measurement():
    if not SAMPLE.is_file():
        pytest.skip(f"needs the 004100 sample at {SAMPLE}")
    if not cache_path(SAMPLE).is_file():
        pytest.skip(f"needs the raw detection cache {cache_path(SAMPLE).name}; "
                    f"build it with eval.common.build_cache (about a minute of GPU for the 004100 sample)")
    cache = build_cache(SAMPLE)
    rec = Recogniser()
    settings = Settings()
    threshold = _threshold()
    seen, _, _ = sightings(cache, settings)
    limit, gallery_px = 60, 48
    rows = _select(seen, gallery_px, limit)
    if len(rows) < limit:
        gallery_px = 32
        rows = _select(seen, gallery_px, limit)

    # Other sightings of each selected track, sampled as `eval.reid.pairs`
    # samples them, for the self-match: can the recogniser identify this face
    # in the source at all.
    others: dict[int, list] = {}
    for tid in {tid for tid, _, _ in rows}:
        track = seen[tid]
        if len(track) > PAIR_SAMPLE:
            step = len(track) / PAIR_SAMPLE
            track = [track[int(k * step)] for k in range(PAIR_SAMPLE)]
        others[tid] = track
    wanted = defaultdict(list)
    for tid, i, d in rows:
        wanted[i].append(("probe", tid, d))
    for tid, track in others.items():
        for i, d in track:
            wanted[i].append(("other", tid, d))

    other_emb = defaultdict(list)     # track -> (frame, embedding)
    probes = []                       # (tid, frame, source embedding, {(mask, attack): score})
    for i, frame in walk(SAMPLE, set(wanted)):
        for role, tid, d in wanted[i]:
            base = rec.embed(frame, d)
            if base is None:
                continue
            if role == "other":
                other_emb[tid].append((i, base))
                continue
            scores = {}
            for name, s in MASKS.items():
                masked, _ = redact(frame, [d], s)
                for attack_name, attack in ATTACKS.items():
                    after = rec.embed(attack(masked, d, s), d)
                    if after is not None:
                        scores[(name, attack_name)] = cosine(base, after)
            probes.append((tid, i, base, scores))

    table = {}
    for name in MASKS:
        for attack_name in ATTACKS:
            got = [p[3][(name, attack_name)] for p in probes if (name, attack_name) in p[3]]
            table[(name, attack_name)] = {
                "matched": sum(s >= threshold for s in got), "of": len(got),
                "worst": max(got) if got else None}
    identifiable = sum(
        best_match(base, [e for j, e in other_emb[tid] if j != i]) >= threshold
        for tid, i, base, _ in probes)

    print(f"\nre-ID after a deblur attempt, {SAMPLE.name}")
    print(f"  {len(probes)} largest faces (>= {gallery_px} px), threshold {threshold:+.3f}")
    print(f"  identifiable in the source (self-match in another frame >= threshold): "
          f"{identifiable}/{len(probes)}")
    print(f"  {'mask':12s} {'attack':11s} matched   worst")
    for (name, attack_name), row in table.items():
        worst = "n/a" if row["worst"] is None else f"{row['worst']:+.3f}"
        print(f"  {name:12s} {attack_name:11s} {row['matched']:3d}/{row['of']:<3d}  {worst}")
    return {"table": table, "probes": len(probes), "gallery_px": gallery_px,
            "threshold": threshold, "identifiable": identifiable}


def _shipped_key() -> str:
    s = Settings()
    mode = fill_mode("face", s)
    return "solid" if mode == "solid" else f"{mode} {s.strength}"


def test_the_shipped_face_fill_survives_both_deblur_attacks(measurement):
    """Decision D-C, enforced: the default face mask leaves no face matching itself."""
    key = _shipped_key()
    assert key in MASKS, f"the shipped face fill {key!r} is not one this test measures"
    assert measurement["probes"] >= 50, "too few faces could be embedded to decide anything"
    for attack_name in ATTACKS:
        row = measurement["table"][(key, attack_name)]
        assert row["of"] == measurement["probes"]
        assert row["matched"] / row["of"] <= MAX_MATCHED_SHARE, (key, attack_name, row)
