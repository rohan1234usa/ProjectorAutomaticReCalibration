"""Pictures and video: deterministic, in range, and each with the property its scenario relies on."""

from fractions import Fraction

import numpy as np
import pytest

from sim import pictures
from sim.pictures import ContentGeometry
from sim.planar import points_in_convex, rect_polygon
from sim.sequence import from_config
from sim.textures import blank_inside, letterbox, photo, stripes
from sim.video import Clip, VideoFrames

SIZE = (400, 120)
GEOMETRY = ContentGeometry(px_per_mm=0.96, overlap_px=rect_polygon(170.0, -0.5, 230.0, 119.5), gamma=2.2)


def _rng(k: int = 0) -> np.random.Generator:
    return np.random.default_rng(k)


def test_photos_are_deterministic_and_in_range():
    a, b, c = photo(SIZE, _rng(1)), photo(SIZE, _rng(1)), photo(SIZE, _rng(2))
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert a.shape == (120, 400, 3) and a.dtype == np.float32 and 0.0 <= a.min() and a.max() <= 1.0
    assert a.std() > 0.05  # textured, not flat


def test_dark_stills_are_dark():
    item = pictures.parse({"type": "dark", "hold_s": 5}, "d", Fraction(5), GEOMETRY)
    img = pictures.draw(item, 0, _rng(3), SIZE, GEOMETRY)
    assert np.mean(img**2.2) < 0.02  # under 2% of white in light, on average


def test_stripes_have_the_requested_period():
    img = stripes((400, 40), period_px=12.5)
    spectrum = np.abs(np.fft.rfft(img[20, :, 0] - img[20, :, 0].mean()))
    assert np.argmax(spectrum) == round(400 / 12.5)  # 32 cycles across 400 px
    horizontal = stripes((40, 400), period_px=12.5, angle_deg=90.0)
    assert np.ptp(horizontal[:, 20, 0]) > 0.6 and np.ptp(horizontal[20, :, 0]) < 1e-6


def test_stripe_period_is_given_in_screen_mm():
    item = pictures.parse({"type": "stripes", "period_mm": 8.0, "hold_s": 1}, "s", Fraction(1), GEOMETRY)
    assert item.stripes[0] == pytest.approx(8.0 * 0.96)


def test_letterbox_and_blank_overlap():
    img = letterbox(photo(SIZE, _rng(4)), 0.1)
    assert img[:12].max() == 0.0 and img[-12:].max() == 0.0 and img[12:-12].std() > 0.05
    blank = blank_inside(photo(SIZE, _rng(5)), GEOMETRY.overlap_px, 0.5)
    yy, xx = np.mgrid[0:120, 0:400]
    inside = points_in_convex(GEOMETRY.overlap_px, np.stack([xx, yy], axis=-1).astype(float), margin=1.0)
    assert np.allclose(blank[inside], 0.5) and blank[~inside].std() > 0.05


def test_video_frames_move_cut_and_fade():
    clip = Clip(fps=Fraction(30), cut_s=Fraction(2), pan_px_per_s=(60.0, 0.0), objects=4,
                speed_px_per_s=200.0, fade_s=Fraction(1, 2))
    video = VideoFrames(clip, SIZE, seed=9, key=(0,))
    again = VideoFrames(clip, SIZE, seed=9, key=(0,))
    assert np.array_equal(video.frame(31), again.frame(31))  # a pure function of the frame index
    assert video.frame(0).max() == 0.0  # each scene fades in from black
    middle = video.frame(30)  # 1 s into the first scene: full brightness
    motion = np.abs(video.frame(31) - middle).mean()
    cut = np.abs(video.frame(75) - middle).mean()  # 0.5 s into the second scene
    assert middle.mean() > 0.3 and 0 < motion < 0.5 * cut  # frames move a little; a cut changes everything


def test_a_video_exposure_spans_two_frames_in_exact_shares():
    seq = from_config({"items": [{"type": "video", "duration_s": 10, "fps": 30}]}, 1, SIZE, Fraction(10), GEOMETRY)
    segments = seq.segments(Fraction(1, 100), Fraction(1, 30))  # starts 10 ms into frame 0
    assert [k for k, _ in segments] == [(0, 0, 0), (0, 0, 1)]
    assert [w for _, w in segments] == [Fraction(7, 10), Fraction(3, 10)]
    assert seq.tag((0, 0, 0)) == "video_photo"


def test_reference_feed_lists_what_was_sent_newest_first():
    seq = from_config({"items": [{"type": "video", "duration_s": 10, "fps": 30}]}, 1, SIZE, Fraction(10), GEOMETRY)
    sent = seq.sent_before(Fraction(1), 4)
    assert [key for _, key in sent] == [(0, 0, 30), (0, 0, 29), (0, 0, 28), (0, 0, 27)]
    assert [t for t, _ in sent] == [Fraction(30, 30), Fraction(29, 30), Fraction(28, 30), Fraction(27, 30)]
    assert len(seq.sent_before(Fraction(1, 20), 10)) == 2  # only frames 1 and 0 were sent by 50 ms


@pytest.mark.parametrize(
    "cfg, message",
    [
        ({"type": "video", "duration_s": 1.01, "fps": 30}, "whole number of frames"),
        ({"type": "video", "duration_s": 1, "fps": 0}, "fps: must be positive"),
        ({"type": "video", "duration_s": 1, "fps": "thirty"}, "frame rate"),
        ({"type": "video", "duration_s": 1, "cut_s": None, "fade_s": 0.5}, "fade at their cuts"),
        ({"type": "video", "duration_s": 1, "cut_s": 0}, "cut_s: must be positive"),
        ({"type": "photo", "hold_s": 1, "border_frac": 0.5}, "border_frac"),
        ({"type": "slide", "hold_s": 1}, "must be one of"),
        ({"type": "stripes", "period_mm": 1.0, "hold_s": 1}, "two content pixels"),
        ({"type": "photo", "hold_s": 1, "letterbox": 0.6}, "letterbox"),
        ({"type": "video", "duration_s": 1, "style": "sepia"}, "must be one of"),
        ({"type": "photo", "hold_s": 1, "density": "high"}, "unknown keys"),
    ],
)
def test_bad_pictures_fail_clearly(cfg, message):
    with pytest.raises(ValueError, match=message):
        pictures.parse(cfg, "content.items[0]", Fraction(1), GEOMETRY)


def test_stripes_half_a_period_apart_cancel_in_light():
    """The grating is sinusoidal in light, so two copies half a period apart sum to a flat field."""
    a = stripes((400, 8), period_px=16.0) ** 2.2
    b = stripes((400, 8), period_px=16.0, phase=np.pi) ** 2.2
    total = 0.5 * a + 0.5 * b
    assert np.ptp(total) < 1e-5 * total.mean()


def test_stripe_pictures_take_random_phases():
    seq = from_config({"loop": True, "items": [{"type": "stripes", "period_mm": 20, "pictures": 3, "hold_s": 1}]},
                      1, SIZE, Fraction(10), GEOMETRY)
    images = [seq.image(key) for key in ((0, 0, 0), (0, 0, 1), (0, 1, 0))]
    assert not np.array_equal(images[0], images[1]) and not np.array_equal(images[0], images[2])


def test_blank_overlap_leaves_the_border_and_bars_black():
    item = pictures.parse({"type": "photo", "hold_s": 1, "border_frac": 0.05, "letterbox": 0.1, "blank_overlap": 0.5},
                          "p", Fraction(1), GEOMETRY)
    img = pictures.draw(item, 0, _rng(6), SIZE, GEOMETRY)
    assert img[:12, 170:230].max() == 0.0 and img[-12:, 170:230].max() == 0.0  # the bars stay black
    assert np.allclose(img[20:100, 175:225], 0.5)  # flat inside the overlap, within the picture


def test_reference_feed_runs_ahead_of_the_display_by_lag_s():
    """The projectors show what was sent lag_s earlier; the feed is timestamped on the sending clock."""
    from sim.frames import FrameSource
    from sim.scenario import scenario_from_dict
    from tests.scenes import TINY_SCENARIO

    cfg = {**TINY_SCENARIO, "duration_s": 4, "camera": {**TINY_SCENARIO["camera"], "phase_s": 0.01},
           "reference": {"available": True, "lag_s": 0.1},
           "content": {"items": [{"type": "video", "duration_s": 4, "fps": 30, "cut_s": 2}]}}
    source = FrameSource(scenario_from_dict(cfg))
    # Frame 2 is exposed over [1.01, 1.0433) s and shows what was sent over [0.91, 0.9433) s.
    shown = [key for key, _ in source.state(2).segments]
    assert [k[2] for k in shown] == [27, 28]  # video frames sent at 0.9 s and 0.933 s
    feed = source.source(2)
    sent = {key: t for t, key, _ in feed}
    assert [sent[k] for k in shown] == pytest.approx([27 / 30, 28 / 30])
    assert feed[0][0] == pytest.approx(31 / 30)  # the newest was sent during the exposure, not yet shown


def test_reference_feed_holds_every_picture_the_exposure_shows():
    from sim.frames import FrameSource
    from sim.scenario import scenario_from_dict
    from tests.scenes import TINY_SCENARIO

    cfg = {**TINY_SCENARIO, "duration_s": 4, "camera": {**TINY_SCENARIO["camera"], "phase_s": 0.01},
           "reference": {"available": True},
           "content": {"items": [{"type": "video", "duration_s": 4, "fps": 30, "cut_s": 2}]}}
    source = FrameSource(scenario_from_dict(cfg))
    shown = [key for key, _ in source.state(3).segments]
    assert len(shown) == 2 and set(shown) <= {key for _, key, _ in source.source(3, ring=4)}
    lines = [source.truth(i) for i in range(len(source))]
    assert all(line["content"]["frames_in_exposure"] == 2 for line in lines)
    cuts = [line["i"] for line in lines if line["content"]["cut_in_exposure"]]
    assert cuts == []  # frames 0.5 s apart never straddle the 2 s cut (it falls between exposures)
    straddle = source.scenario.sequence.segments(Fraction(2) - Fraction(1, 100), Fraction(1, 30))
    assert not source.scenario.sequence.same_shot(straddle[0][0], straddle[1][0])  # a real cut
