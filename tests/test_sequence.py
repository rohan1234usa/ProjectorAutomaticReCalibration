"""Content over time: which picture is up when, exact exposure shares, deterministic pictures."""

from fractions import Fraction

import numpy as np
import pytest

from sim.sequence import from_config

SIZE = (320, 90)
DECK = {"loop": True, "items": [{"type": "deck", "densities": ["low", "medium", "high"], "slides": 3, "hold_s": 2},
                                {"type": "black", "hold_s": 1}]}


def test_deck_timeline_and_densities():
    seq = from_config(DECK, seed=4, size=SIZE, duration=Fraction(30))
    exposure = Fraction(1, 30)
    keys = [seq.segments(Fraction(t), exposure)[0][0] for t in (0, 1, 2, 5, 6, 7)]
    assert keys == [(0, 0, 0), (0, 0, 0), (0, 0, 1), (0, 0, 2), (1, 0, 0), (0, 1, 0)]  # 7 s cycle, then loop 1
    assert [seq.tag(k) for k in keys] == ["deck_low", "deck_low", "deck_medium", "deck_high", "black", "deck_low"]


def test_looped_decks_show_new_slides_but_black_stays_black():
    seq = from_config(DECK, seed=4, size=SIZE, duration=Fraction(30))
    assert not np.array_equal(seq.image((0, 0, 0)), seq.image((0, 1, 0)))
    exposure = Fraction(1, 30)
    assert seq.segments(Fraction(6), exposure)[0][0] == seq.segments(Fraction(13), exposure)[0][0] == (1, 0, 0)


def test_pictures_are_deterministic_per_key():
    a = from_config(DECK, seed=4, size=SIZE, duration=Fraction(30))
    b = from_config(DECK, seed=4, size=SIZE, duration=Fraction(30))
    c = from_config(DECK, seed=5, size=SIZE, duration=Fraction(30))
    assert np.array_equal(b.image((0, 0, 2)), a.image((0, 0, 2)))  # whatever order they are drawn in
    assert not np.array_equal(a.image((0, 0, 2)), c.image((0, 0, 2)))
    assert not a.image((0, 0, 2)).flags.writeable


def test_exposure_straddling_a_change_is_split_exactly():
    seq = from_config(DECK, seed=4, size=SIZE, duration=Fraction(30))
    exposure = Fraction(1, 30)
    segments = seq.segments(Fraction(2) - Fraction(1, 100), exposure)  # starts 10 ms before the 2 s slide change
    assert [k for k, _ in segments] == [(0, 0, 0), (0, 0, 1)]
    assert [w for _, w in segments] == [Fraction(3, 10), Fraction(7, 10)]
    for t in np.linspace(0, 20, 57):
        assert sum(w for _, w in seq.segments(Fraction(repr(float(t))), exposure)) == 1


def test_a_single_picture_mapping_is_held_for_the_whole_run():
    seq = from_config({"type": "slide", "border_frac": 0.1}, seed=1, size=SIZE, duration=Fraction(100))
    assert {seq.segments(Fraction(t), Fraction(1, 30))[0][0] for t in (0, 50, 99)} == {(0, 0, 0)}
    assert seq.tag((0, 0, 0)) == "held_medium"
    assert seq.image((0, 0, 0))[:5].max() == 0.0  # the black border


def test_flat_colour_and_held_density():
    seq = from_config([{"type": "flat", "value": [0.2, 0.4, 0.6], "hold_s": 5}, {"type": "held", "density": "high", "hold_s": 5}],
                      seed=1, size=SIZE, duration=Fraction(10))
    assert np.allclose(seq.image((0, 0, 0))[40, 100], [0.2, 0.4, 0.6])
    assert seq.tag((1, 0, 0)) == "held_high"


@pytest.mark.parametrize(
    "cfg, message",
    [
        ({"items": [{"type": "black", "hold_s": 2}]}, "set loop: true"),
        ({"items": [{"type": "video", "hold_s": 2}]}, "must be one of"),
        ({"items": [{"type": "deck", "density": "low", "densities": ["high"], "hold_s": 2}]}, "not both"),
        ({"items": [{"type": "deck", "density": "huge", "hold_s": 2}]}, "must be one of"),
        ({"items": [{"type": "flat", "hold_s": 0}]}, "positive"),
        ({"loop": "yes", "items": [{"type": "black", "hold_s": 2}]}, "true or false"),
        ([], "non-empty"),
    ],
)
def test_bad_content_fails_clearly(cfg, message):
    with pytest.raises(ValueError, match=message):
        from_config(cfg, seed=1, size=SIZE, duration=Fraction(10))
