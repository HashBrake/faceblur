"""Every knob the pipeline has.

The defaults changed after the precision audit (docs/precision_report.md). The
first build was tuned to never miss a face and accepted over blurring. This
build destroys pixels only where several independent checks agree that a face
is there, and it masks the smallest region that hides identity.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

ENGINES = ("yunet", "centerface", "both")
DEVICES = ("auto", "gpu", "cpu")
ENCODERS = ("auto", "nvenc", "x264")
MODES = ("blur", "pixelate", "solid")

# Video extensions the batch walker treats as input.
VIDEO_EXT = frozenset(
    {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg", ".wmv"}
)


class SettingsError(ValueError):
    """A setting is outside the range the pipeline accepts."""


@dataclass(frozen=True)
class Settings:
    # --- what to mask --------------------------------------------------------
    # Which kinds of sensitive thing to hide, from faceblur/classes.py. Each
    # one is a switch of its own: masking faces does not mask text. A kind
    # with no detector behind it cannot be named here, because a switch that
    # masks nothing is worse than no switch.
    mask: tuple[str, ...] = ("face",)

    # --- screens -------------------------------------------------------------
    # Which COCO things count as a screen. The model knows eighty; these are
    # the ones that are glass with something on them.
    screen_labels: tuple[str, ...] = ("tv", "laptop", "phone")
    # One detector, no confirmation, because there is no identity to weigh:
    # a screen is masked as an object whatever is on it. 0.35 was the first
    # try and it is too low: section 15.1 has it destroying 78 percent of a
    # frame. At 0.5 the same footage reads 0.15 to 0.30 percent per frame.
    screen_conf: float = 0.5
    screen_nms: float = 0.45
    # Largest plausible screen, as a share of the frame's area. A COCO
    # detector calls any large flat rectangle a television and this footage
    # is full of them: a blue table tennis table reads as a laptop at 0.88
    # over a third of the frame, a washroom mirror as a television.
    #
    # 12 rather than the 4 that a cap on its own needs, because it does not
    # work on its own: with screen_min_run beside it, 4 and 12 hold the same
    # 91 percent precision, and 12 keeps a real monitor at 10.6 percent of
    # the frame that 4 threw away. That monitor is the kind most likely to be
    # showing something readable, so the looser cap is the safer one once
    # persistence is carrying the precision. Section 15.3.
    screen_max_area: float = 0.12
    # Frames a screen has to be seen in before any of it is masked. This is
    # what the cap cannot do: 14 of the 29 false runs on the sample footage
    # last a single frame and no real screen does, because a table only looks
    # like a laptop from some angles while a monitor on a wall stays a
    # monitor. It is the same idea as established_after on the face side, and
    # it costs one real screen of 54.
    screen_min_run: int = 3
    # Grow the box by this share of its own size. A detector's box stops at
    # the glass, and a box that stops at the glass leaves a rim of picture
    # once the camera has moved between the frame it was found in and the
    # frame it is masked in.
    screen_pad: float = 0.06
    # Frames a screen stays masked after its last sighting, and the most
    # frames that may be missing between two sightings and still be filled
    # rather than left open. A screen does not leave the room between one
    # frame and the next; a detector that drops it for two frames has not
    # been told that. Two frames side by side have a gap of zero, so a gap
    # of zero still joins them into a run.
    screen_tail: int = 6
    screen_gap: int = 12
    # The smallest screen the output-side check will hold a copy back for, as
    # the box's long side. 48 rather than the 24 px faces use, because the
    # thing being hidden is what is on the glass rather than who the glass
    # belongs to, and a 24 px screen shows nothing anybody could read. First
    # guess; section 16.1 measures what it lets through.
    screen_gate_min_px: int = 48
    # Checked frames a screen has to be found in on the copy before it holds
    # that copy back. The same rule as screen_min_run and for the same reason:
    # 14 of the 29 false runs on the sample footage last a single frame and no
    # real screen does, so a gate without it would hold every copy back for a
    # table that looked like a laptop once. The length counts *checked*
    # frames, so at check_stride 2 a run of 3 spans six source frames.
    screen_gate_min_run: int = 3

    # --- the handled zone ----------------------------------------------------
    # The region nothing is ever masked in: the wearer's hands and what they
    # are holding. This is the owner's rule of 2026-09-11 and it is the one
    # precision constraint that is absolute, so it is on by default and
    # --no-zone exists for measurement only. See faceblur/zone.py and report
    # section 17.
    zone: bool = True
    # Sliding window the palm model sees, before it is downscaled to its own
    # 192. The wearer's hands run 150 to 500 px on this camera, and the whole
    # frame at 192 puts a hand at 20 px where the model sees nothing. Same
    # reasoning as hand_windows, measured in section 14.2.
    zone_window: int = 512
    # Run the hand pass on every Nth frame. The zone has memory, so the frames
    # between are covered; 1 until section 17 says what 2 costs.
    zone_stride: int = 1
    # Suppression over palm boxes pooled from overlapping windows. One hand
    # lands in four windows and has to come out as one hand.
    zone_nms: float = 0.5
    # A hand qualifies as the wearer's at this palm size or when it reaches
    # the bottom zone_edge_frac of the frame. In egocentric footage the
    # wearer's hands are the closest thing of their kind and enter from below.
    # Both are first guesses; section 17 measures how often a bystander's hand
    # qualifies.
    zone_min_hand_px: int = 120
    zone_edge_frac: float = 0.25
    # How far the protected polygon reaches past MediaPipe's own hand
    # rectangle, which is already 2.6 times the palm box. A card or a phone
    # held in the hand extends about one hand's width past the fingers. First
    # guess, section 17 measures it.
    zone_scale: float = 1.5
    # Seconds a handled thing stays protected after the hand leaves it. Only
    # text and screens are protected by memory: a person who walks into the
    # space where a card was put down is still a person.
    zone_memory: float = 3.0
    # The precision gate. A copy in which more than this share of the zone's
    # pixels moved is held back, whatever else the check found. First guess,
    # section 17 measures what a clean run reads.
    zone_gate_changed: float = 0.005
    # A face is protected by the zone only where a hand is actually on it,
    # not merely inside the grown reach of one. The audit of 2026-09-11 found
    # the zone covering a canteen worker's face at 144 px: their own hand
    # rested near the bottom of the frame, qualified on the edge rule at 56 px,
    # and the grown polygon reached 40 px up and over their face. Protecting a
    # stranger's face from redaction is the worst failure this tool has, and
    # it is worse than masking a printed face on a card in somebody's hand,
    # which is what this costs. Set False to go back to the reach.
    # Report section 17, docs/audits/zone_hands_004310.csv.
    zone_face_needs_hand: bool = True
    # How far the face protecting region reaches, as a multiple of the palm
    # box. MediaPipe's own hand rectangle is 2.6 palm boxes, which is the crop
    # its landmark model wants rather than the hand's silhouette, and at 2.6 a
    # 56 px palm makes a 146 px quad that covers a 144 px face standing beside
    # it. Swept against the 27 labelled cases in
    # docs/audits/zone_faces_set_aside_2026-09-11.csv: 1.6 keeps all 7 of the
    # wearer's own hands and cuts the bystander faces protected from 14 to 3,
    # where 2.0 gives 5 and 1.3 starts losing real hands. Report section 17.
    zone_face_scale: float = 1.6
    # Share of a box the zone has to cover before the check sets that find
    # aside. A face inside the handled zone is not a miss, it is the rule
    # working, so it stays in the record marked with the coverage that decided
    # it and stays out of every count the gate reads, exactly as a hand does.
    zone_cover: float = 0.5
    # Where the two hand models run for the zone, like hand_device.
    zone_device: str = "auto"

    # --- detection -----------------------------------------------------------
    # yunet: YuNet finds faces, CenterFace confirms them (see verify).
    # both: union of the two, no confirmation. centerface: CenterFace alone.
    engine: str = "yunet"
    # YuNet score threshold. Chosen by eval/sweep.py: the highest recall that
    # keeps off-face masking under 0.3 percent and hands untouched.
    conf: float = 0.6
    # Absolute long side lengths to scan at. 640 was dropped: on a 1600 px frame
    # it shrinks a 35 px face to 14 px while a sink becomes a face.
    det_sizes: tuple[int, ...] = (1280, 1920)
    # A YuNet box survives only if CenterFace also fired on it.
    verify: bool = True
    verify_conf: float = 0.3
    verify_iou: float = 0.3
    # A borderline plain confirmation, below verify_conf_sure, counts only if
    # the mirror view also sees a face at mirror_agree or more. Faces are
    # symmetric and clear this every time measured (43 of 45 borderline
    # consensus faces on three files); the wearer's hand did not (0 of 2).
    verify_conf_sure: float = 0.5
    mirror_agree: float = 0.2
    # Confirmation runs on a crop around each candidate rather than on the
    # whole frame at every size. The crop is crop_scale times the box, scaled
    # so its long side is crop_size pixels. Same confirmations, a fraction of
    # the cost, and one fixed shape for the GPU.
    confirm_on_crops: bool = True
    crop_size: int = 224
    crop_scale: float = 4.0
    # Confirmation looks at more than one view of each crop: 0 the crop only,
    # 1 also its mirror image, 2 also a tighter crop where the face is larger.
    # The best view counts. A face turned down or cut by the frame edge is
    # confirmed in one of them where the plain crop fails.
    confirm_tta: int = 2
    # A mirror or tight view has to be more sure than the plain view needs to
    # be. Every extra view is an extra chance for a hand to cross the line;
    # a face the plain view nearly confirmed clears this in another view.
    verify_conf_view: float = 0.4
    # Third detector, a different family (UltraFace), asked only when the
    # second one is unsure: it scored between verify_conf_low and verify_conf.
    # Two of three then decide. Never asked about a box the second detector
    # scored below verify_conf_low, which is where hands and signs land.
    third_opinion: bool = True
    third_conf: float = 0.5
    verify_conf_low: float = 0.25
    # Two detectors agreeing is as good as one being sure: a YuNet box from
    # conf_agree up that CenterFace scores at verify_conf_agree or more is a
    # face, though YuNet alone is below conf. Mirror reflections and far
    # faces land here. Measured on three files: no hand box passes it.
    # Only for boxes up to agree_max_px: mirror reflections and far faces are
    # under 40 px, the wearer's hand is 90 px and more, and at 0.4 and 0.5
    # with no size limit the rule admitted a hand box on the busiest sample.
    # 0 for no limit.
    conf_agree: float = 0.35
    verify_conf_agree: float = 0.35
    agree_max_px: int = 48
    # Once a track is confirmed, YuNet alone may keep it alive at this lower
    # threshold, if the box overlaps where the track predicts the face to be.
    # A hand can never start a track, so this costs no precision.
    conf_weak: float = 0.4
    # An established track (established_after confirmed detections) may
    # continue on boxes down to this: a face smeared by a fast camera move
    # scores low, and the track already knows where it is.
    conf_weak_long: float = 0.3
    # Confirmed detections that make a track established: only then does it
    # reach backwards over weak boxes, get the entry tail, the long exit
    # tail, the lowest continuation floor and the gap filling. A face is
    # confirmed in every frame once it is in the picture, so with hindsight
    # five costs a face nothing; the wearer's hand is confirmed two or three
    # frames in a row at most, and used to get thirty frames of mask before
    # those from the backward reach alone.
    established_after: int = 5
    # Continuation on weak boxes starts only once a track holds this many
    # confirmed detections, and runs at most weak_run frames past the last
    # confirmed one. The wearer's own hand gets confirmed now and then (once
    # in 200 frames on the table tennis file) and YuNet scores it 0.4-0.6 in
    # every frame after; without these two limits that one confirmation
    # became a track that masked the hand for fifty frames.
    continue_after: int = 3
    weak_run: int = 30
    # A track earns continuation, stitching, gap filling and the long tails
    # only once one of its confirmations scored this much from CenterFace.
    # Faces get there in almost every track (391 of 401 consensus boxes on
    # the busiest file score 0.5 or more); the wearer's hand, confirmed now
    # and then at 0.3 to 0.4, never does.
    track_sure_conf: float = 0.5
    # Largest plausible face, as a share of the frame's long side. The largest
    # real face in the Ego footage was 150 px of 1600. False positives ran to 800.
    # 0.15 of 1600 is 240 px.
    max_face_frac: float = 0.15
    # A confirmed track may follow a face that grows past max_face_frac, up to
    # this share, as the person walks up to the camera. A box that large can
    # continue a track but never start one. Never below max_face_frac.
    grow_face_frac: float = 0.35
    # Faces are roughly square. Anything wider or taller than this is not one.
    min_aspect: float = 0.4
    max_aspect: float = 2.5
    stride: int = 1
    # auto uses the GPU when onnxruntime can see one (DirectML on Windows,
    # CUDA elsewhere) and falls back to the CPU.
    device: str = "auto"

    # --- tracking -----------------------------------------------------------
    # A detection counts only when its track holds at least min_track detections.
    min_track: int = 2
    # Frames a track may go undetected before it closes. Gaps are interpolated.
    # The sweeps of 2026-09-07 split 3 against 5 with nothing between them;
    # 5 keeps a face covered over more missed frames. A longer gap still is
    # allowed while the camera moves fast (max_gap_fast).
    max_gap: int = 5
    # Frames the mask extends past the last sighting of a track; an
    # established track gets tail_long.
    tail: int = 2
    tail_long: int = 4
    # Frames the mask extends before the first sighting. Longer than tail:
    # a face entering the picture is visible for a few frames before the
    # detectors lock on, and those frames are the ones that show a face.
    tail_before: int = 6
    # Tails follow the track's motion and grow by this share per frame, so an
    # entering or leaving face stays under the mask as it moves. 0.03 since
    # the tails follow the camera; 0.05 before.
    tail_grow: float = 0.03
    track_iou: float = 0.3
    # The camera's own movement between frames, measured by phase correlation,
    # is taken out of the tracker's prediction. A pan no longer breaks a track.
    camera_comp: bool = True
    # Besides overlap, a detection may join a track when its centre lies within
    # this share of the box size of the prediction and the sizes agree. Overlap
    # alone is brittle for a 30 px face that moves 20 px a frame.
    link_dist: float = 0.75
    # While the camera moves faster than this (source pixels per frame), a
    # track may survive a longer gap, max_gap_fast, since that is when the
    # detectors miss.
    fast_shift: float = 10.0
    max_gap_fast: int = 10
    # A moving face gets a longer entry tail than tail_before, up to this,
    # until the extrapolated box has left the picture.
    tail_before_max: int = 12
    # Two tracks of one face separated by up to this many frames, where the
    # detectors lost it (a hit at table tennis smears the picture for a
    # second), are joined and the gap interpolated. Offline hindsight: the
    # face came back where it was expected.
    stitch_gap: int = 45
    # How far (source pixels per frame of gap) the second piece may sit from
    # where the first one's motion and the camera put it. During a hit at
    # table tennis the picture smears and the camera shift cannot be measured,
    # so the prediction is off by the whole move; the gate widens with the gap.
    stitch_slack: float = 8.0
    # Boxes interpolated across a gap grow by this share of their size per
    # frame away from the nearest sighting, most in the middle of the gap,
    # where the face is least certain.
    gap_grow: float = 0.04
    # While the camera moves faster than this (source pixels per frame) every
    # mask grows by the shift, since the face is smeared by that much.
    blur_shift: float = 8.0
    # Tails follow the size trend of the track's end: a face that was
    # shrinking is drawn larger further back. Off: measured to add masking
    # on the worst frame and nothing to recall.
    tail_trend: bool = False

    # --- mask geometry -------------------------------------------------------
    # Ellipse axes as a share of the detector box, rotated to the eye line.
    ellipse_w: float = 1.10
    ellipse_h: float = 1.15
    # Fallback pad for a box that carries no landmarks.
    pad: float = 0.10
    # Soft edge width in pixels.
    feather: float = 6.0

    # --- pixel destruction ---------------------------------------------------
    mode: str = "blur"
    # Blocks across a face after downsampling. 6 leaves nothing to recognise.
    strength: int = 6

    # --- output --------------------------------------------------------------
    # Near lossless. Untouched pixels are training data.
    crf: int = 12
    preset: str = "fast"
    # auto uses NVENC when the GPU and the bundled ffmpeg both support it.
    encoder: str = "auto"
    # NVENC constant quality, the counterpart of crf. 16 is near lossless.
    nvenc_cq: int = 16
    suffix: str = "_blurred"
    replace_existing: bool = False
    # A frame masked beyond this share is flagged in the audit record.
    mask_budget: float = 0.05

    # --- segments ------------------------------------------------------------
    # Detection runs in frame ranges of this many seconds, in parallel. 0 means
    # one range for the whole video.
    chunk_seconds: float = 15.0
    # Stretches with no mask are copied from the source byte for byte. If the
    # joined file does not verify, the video is encoded end to end instead.
    copy_clean: bool = True
    # A clean stretch shorter than this is encoded with its neighbours.
    min_copy_seconds: float = 2.0
    # Masked stretches are cut at keyframes into pieces of about this length,
    # so several workers encode them at once.
    encode_seconds: float = 5.0
    # Concurrent NVENC encodes. GeForce drivers from 2024 allow 8; 6 is safe.
    nvenc_sessions: int = 6
    # ffmpeg hardware decode: none or cuda. Measured equal to the CPU on a
    # 1600x1300 file, so it is off by default.
    hwaccel: str = "none"

    nms_detect: float = 0.35
    nms_yunet: float = 0.30

    # --- checking the finished copy ------------------------------------------
    # Detect on the output as well: a face found there, on pixels nothing
    # changed, is one this run missed, and nothing that reads the source can
    # see it. Costs a second detection pass over the video, so it is off by
    # default and worth turning on for anything that leaves the machine.
    check_output: bool = False
    check_stride: int = 1
    # With this on, an output that still shows a face of quarantine_min_px or
    # more is moved into a quarantine folder beside it instead of shipping.
    # Implies check_output.
    quarantine: bool = False
    quarantine_min_px: int = 24
    # The check runs the same face detectors that produced the copy, and they
    # call the wearer's hand a face. The pipeline keeps hands out of the mask
    # with track-level rules; a check that reads one frame at a time cannot,
    # so it asks MediaPipe's hand models instead. A flagged box that a
    # confirmed hand covers is recorded as a hand and does not hold the copy
    # back. See faceblur/hands.py and section 14 of docs/report.md.
    hand_rule: bool = True
    # Windows, in pixels, centred on the flagged box, each downscaled to the
    # palm model's own 192. Fixed pixels rather than a multiple of the box:
    # what the palm detector makes of a crop depends on how large a hand is
    # inside it, and a multiple of the box held nothing constant. 512 alone
    # sets aside 12 of 20 hands, 384 and 512 together 13, a third at 768 none.
    hand_windows: tuple[int, ...] = (384, 512)
    # The palm detector's own floor. The rule reads the same from 0.3 to 0.5
    # once the landmark model has confirmed, so this is not where the decision
    # is made and it sits in the middle of the flat part.
    hand_conf: float = 0.4
    # How sure the landmark model has to be that the palm detector's rectangle
    # holds a hand. This is the setting that decides. At 0.6 the rule loses a
    # real face on the sample footage and at 0.65 it stops; 0.7 is one step
    # inside, and from there down to a palm floor of 0.3 and out to any
    # coverage between 0.3 and 0.7 it loses none.
    hand_presence: float = 0.7
    # Share of the flagged box a confirmed hand has to cover. Flat from 0.3 to
    # 0.7 on the sample footage: by the time the hands are confirmed, a box
    # they touch at all is one they mostly cover.
    hand_cover: float = 0.5
    # Where the two hand models run. Kept apart from `device` because the
    # worry was that they would not fit: four workers already hold about
    # 850 MB of face-detector graphs each on an 8 GB card. Measured, they cost
    # nothing there — 338 s against 341 s with the rule off on the busiest
    # sample file — and 13 percent on the CPU, so the worry was wrong and the
    # setting is only here for a machine where it is not. Section 14.4.
    hand_device: str = "auto"

    def wants(self, kind: str) -> bool:
        """Is this kind of thing to be masked on this run."""
        return kind in self.mask

    def __post_init__(self) -> None:
        object.__setattr__(self, "mask", tuple(self.mask))
        object.__setattr__(self, "det_sizes", tuple(int(s) for s in self.det_sizes))
        object.__setattr__(self, "hand_windows", tuple(int(w) for w in self.hand_windows))
        object.__setattr__(self, "screen_labels", tuple(self.screen_labels))
        self.validate()

    def validate(self) -> None:
        from .classes import BY_NAME, KINDS
        if not self.mask:
            raise SettingsError(
                "mask must name at least one kind of thing to mask; "
                f"this build knows {', '.join(k.name for k in KINDS)}")
        for name in self.mask:
            if name not in BY_NAME:
                raise SettingsError(
                    f"mask names {name!r}, which is not something this build knows; "
                    f"it knows {', '.join(k.name for k in KINDS)}")
            if not BY_NAME[name].ready:
                raise SettingsError(
                    f"this build cannot mask {name} yet")
        if self.engine not in ENGINES:
            raise SettingsError(f"engine must be one of {ENGINES}, got {self.engine!r}")
        if self.mode not in MODES:
            raise SettingsError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.device not in DEVICES:
            raise SettingsError(f"device must be one of {DEVICES}, got {self.device!r}")
        if self.encoder not in ENCODERS:
            raise SettingsError(f"encoder must be one of {ENCODERS}, got {self.encoder!r}")
        if not 0.0 < self.conf <= 1.0:
            raise SettingsError(f"conf must be above 0 and at most 1, got {self.conf}")
        if self.chunk_seconds < 0 or self.min_copy_seconds < 0:
            raise SettingsError("chunk_seconds and min_copy_seconds must be 0 or more")
        if self.check_stride < 1:
            raise SettingsError(f"check_stride must be at least 1, got {self.check_stride}")
        if self.quarantine_min_px < 0:
            raise SettingsError("quarantine_min_px must be 0 or more")
        if not self.hand_windows or any(w < 64 for w in self.hand_windows):
            raise SettingsError("hand_windows must hold at least one window of 64 px or more")
        if not 0.0 < self.hand_conf <= 1.0:
            raise SettingsError(f"hand_conf must be above 0 and at most 1, got {self.hand_conf}")
        if not 0.0 < self.hand_presence <= 1.0:
            raise SettingsError(
                f"hand_presence must be above 0 and at most 1, got {self.hand_presence}")
        if not 0.0 < self.hand_cover <= 1.0:
            raise SettingsError(
                f"hand_cover must be above 0 and at most 1, got {self.hand_cover}")
        if self.hand_device not in DEVICES:
            raise SettingsError(
                f"hand_device must be one of {DEVICES}, got {self.hand_device!r}")
        from .screens import SCREEN_LABELS
        known = set(SCREEN_LABELS.values())
        if not self.screen_labels:
            raise SettingsError("screen_labels must name at least one kind of screen")
        for name in self.screen_labels:
            if name not in known:
                raise SettingsError(
                    f"screen_labels names {name!r}; the model knows "
                    f"{', '.join(sorted(known))}")
        if not 0.0 < self.screen_conf <= 1.0:
            raise SettingsError(
                f"screen_conf must be above 0 and at most 1, got {self.screen_conf}")
        if not 0.0 < self.screen_nms <= 1.0:
            raise SettingsError(
                f"screen_nms must be above 0 and at most 1, got {self.screen_nms}")
        if self.screen_pad < 0:
            raise SettingsError("screen_pad must be 0 or more")
        if not 0.0 < self.screen_max_area <= 1.0:
            raise SettingsError(
                f"screen_max_area must be above 0 and at most 1, "
                f"got {self.screen_max_area}")
        if self.screen_min_run < 1:
            raise SettingsError(
                f"screen_min_run must be at least 1, got {self.screen_min_run}")
        if self.screen_tail < 0 or self.screen_gap < 0:
            raise SettingsError("screen_tail and screen_gap must be 0 or more")
        if self.zone_window < 64:
            raise SettingsError(
                f"zone_window must be at least 64, got {self.zone_window}")
        if self.zone_stride < 1:
            raise SettingsError(
                f"zone_stride must be at least 1, got {self.zone_stride}")
        if not 0.0 < self.zone_nms <= 1.0:
            raise SettingsError(
                f"zone_nms must be above 0 and at most 1, got {self.zone_nms}")
        if self.zone_min_hand_px < 0:
            raise SettingsError("zone_min_hand_px must be 0 or more")
        if not 0.0 <= self.zone_edge_frac <= 1.0:
            raise SettingsError(
                f"zone_edge_frac must be between 0 and 1, got {self.zone_edge_frac}")
        if self.zone_scale <= 0:
            raise SettingsError(f"zone_scale must be above 0, got {self.zone_scale}")
        if self.zone_memory < 0:
            raise SettingsError("zone_memory must be 0 or more seconds")
        if not 0.0 <= self.zone_gate_changed <= 1.0:
            raise SettingsError(
                f"zone_gate_changed must be between 0 and 1, got {self.zone_gate_changed}")
        if self.zone_face_scale <= 0:
            raise SettingsError(
                f"zone_face_scale must be above 0, got {self.zone_face_scale}")
        if not 0.0 < self.zone_cover <= 1.0:
            raise SettingsError(
                f"zone_cover must be above 0 and at most 1, got {self.zone_cover}")
        if self.zone_device not in DEVICES:
            raise SettingsError(
                f"zone_device must be one of {DEVICES}, got {self.zone_device!r}")
        if self.screen_gate_min_px < 0:
            raise SettingsError(
                f"screen_gate_min_px must be 0 or more, got {self.screen_gate_min_px}")
        if self.screen_gate_min_run < 1:
            raise SettingsError(
                f"screen_gate_min_run must be at least 1, got {self.screen_gate_min_run}")
        if self.crop_size < 64 or self.crop_scale < 1.0:
            raise SettingsError("crop_size must be at least 64 and crop_scale at least 1")
        if not 0.0 < self.conf_weak <= self.conf:
            raise SettingsError(f"conf_weak must be above 0 and at most conf, got {self.conf_weak}")
        # conf_agree above conf, or conf_weak_long above conf_weak, simply
        # means that rule has nothing to act on; a caller lowering conf for a
        # sweep should not have to lower these too.
        if not 0.0 < self.conf_agree <= 1.0 or not 0.0 < self.verify_conf_agree <= 1.0:
            raise SettingsError("conf_agree and verify_conf_agree must be in (0, 1]")
        if self.agree_max_px < 0:
            raise SettingsError("agree_max_px must be 0 or more")
        if not 0.0 < self.conf_weak_long <= 1.0:
            raise SettingsError("conf_weak_long must be in (0, 1]")
        if not 0.0 <= self.track_sure_conf <= 1.0:
            raise SettingsError("track_sure_conf must be between 0 and 1")
        if self.continue_after < 1 or self.weak_run < 0:
            raise SettingsError("continue_after must be at least 1 and weak_run 0 or more")
        if self.stitch_slack < 0 or self.gap_grow < 0:
            raise SettingsError("stitch_slack and gap_grow must be 0 or more")
        if not self.verify_conf <= self.verify_conf_sure <= 1.0 or not 0 <= self.mirror_agree <= 1:
            raise SettingsError("verify_conf_sure must be between verify_conf and 1, "
                                "mirror_agree between 0 and 1")
        if self.established_after < 1 or self.stitch_gap < 0 or self.blur_shift < 0:
            raise SettingsError("established_after must be at least 1; stitch_gap and "
                                "blur_shift 0 or more")
        if self.tail_long < self.tail:
            raise SettingsError("tail_long must be at least tail")
        if self.confirm_tta not in (0, 1, 2):
            raise SettingsError(f"confirm_tta must be 0, 1 or 2, got {self.confirm_tta}")
        if not 0.0 < self.third_conf <= 1.0:
            raise SettingsError(f"third_conf must be above 0 and at most 1, got {self.third_conf}")
        if not 0.0 <= self.verify_conf_low <= self.verify_conf:
            raise SettingsError("verify_conf_low must be between 0 and verify_conf")
        if not 0.0 < self.grow_face_frac <= 1.0:
            raise SettingsError(f"grow_face_frac must be in (0, 1], got {self.grow_face_frac}")
        if self.link_dist < 0 or self.fast_shift < 0:
            raise SettingsError("link_dist and fast_shift must be 0 or more")
        if self.max_gap_fast < self.max_gap:
            raise SettingsError("max_gap_fast must be at least max_gap")
        if self.tail_before_max < self.tail_before:
            raise SettingsError("tail_before_max must be at least tail_before")
        if not self.det_sizes:
            raise SettingsError("det_sizes must name at least one size")
        if any(s < 64 for s in self.det_sizes):
            raise SettingsError(f"every det_size must be at least 64, got {self.det_sizes}")
        if not 0.0 < self.max_face_frac <= 1.0:
            raise SettingsError(f"max_face_frac must be in (0, 1], got {self.max_face_frac}")
        if self.stride < 1:
            raise SettingsError(f"stride must be at least 1, got {self.stride}")
        if self.min_track < 1:
            raise SettingsError(f"min_track must be at least 1, got {self.min_track}")
        if self.max_gap < 0 or self.tail < 0 or self.tail_before < 0 or self.tail_grow < 0:
            raise SettingsError("max_gap, tail, tail_before and tail_grow must be 0 or more")
        if self.hwaccel not in ("none", "cuda"):
            raise SettingsError(f"hwaccel must be none or cuda, got {self.hwaccel!r}")
        if self.encode_seconds < 0 or self.nvenc_sessions < 1:
            raise SettingsError("encode_seconds must be 0 or more and nvenc_sessions at least 1")
        if self.ellipse_w <= 0 or self.ellipse_h <= 0:
            raise SettingsError("ellipse_w and ellipse_h must be above 0")
        if self.pad < 0:
            raise SettingsError(f"pad must be 0 or more, got {self.pad}")
        if self.feather < 0:
            raise SettingsError(f"feather must be 0 or more, got {self.feather}")
        if self.strength < 1:
            raise SettingsError(f"strength must be at least 1, got {self.strength}")
        if not 0 <= self.crf <= 51:
            raise SettingsError(f"crf must be between 0 and 51, got {self.crf}")

    def with_changes(self, **changes) -> "Settings":
        return replace(self, **changes)

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in (
            "mask",
            "engine", "conf", "conf_weak", "det_sizes", "verify", "verify_conf", "verify_iou",
            "confirm_on_crops", "crop_size", "crop_scale",
            "confirm_tta", "verify_conf_view", "third_opinion", "third_conf", "verify_conf_low",
            "conf_agree", "verify_conf_agree", "agree_max_px", "conf_weak_long", "established_after",
            "continue_after", "weak_run", "track_sure_conf",
            "tail_long", "stitch_gap", "stitch_slack", "gap_grow", "blur_shift",
            "verify_conf_sure", "mirror_agree",
            "max_face_frac", "grow_face_frac", "min_aspect", "max_aspect", "stride", "device",
            "min_track", "max_gap", "tail", "tail_before", "tail_grow", "track_iou",
            "camera_comp", "link_dist", "fast_shift", "max_gap_fast", "tail_before_max",
            "tail_trend",
            "ellipse_w", "ellipse_h", "pad", "feather",
            "mode", "strength", "crf", "preset", "encoder", "nvenc_cq", "device", "mask_budget",
            "chunk_seconds", "copy_clean", "min_copy_seconds", "encode_seconds",
            "nvenc_sessions", "hwaccel",
            "check_output", "check_stride", "quarantine", "quarantine_min_px",
            "hand_rule", "hand_windows", "hand_conf", "hand_presence", "hand_cover",
            "hand_device",
            "screen_labels", "screen_conf", "screen_nms", "screen_pad",
            "screen_max_area", "screen_min_run", "screen_tail", "screen_gap",
            "screen_gate_min_px", "screen_gate_min_run",
            "zone", "zone_window", "zone_stride", "zone_nms", "zone_min_hand_px",
            "zone_edge_frac", "zone_scale", "zone_memory", "zone_gate_changed",
            "zone_cover", "zone_face_needs_hand", "zone_face_scale", "zone_device",
            "nms_detect", "nms_yunet")}
        # Tuples so that a record read back from JSON compares equal to the
        # one that wrote it.
        d["mask"] = list(self.mask)
        d["screen_labels"] = list(self.screen_labels)
        d["det_sizes"] = list(self.det_sizes)
        d["hand_windows"] = list(self.hand_windows)
        return d


def parse_det_sizes(text: str) -> tuple[int, ...]:
    """Read a det_sizes string such as '1280,1920' into a tuple."""
    sizes = tuple(int(part) for part in text.split(",") if part.strip())
    if not sizes:
        raise SettingsError(f"could not read any det_size from {text!r}")
    return sizes
