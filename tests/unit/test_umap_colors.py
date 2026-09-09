"""Colours for the UMAP figures.

The plots and the browser's map both colour points by class. Both used to index
a fixed palette with ``i % N``, which silently gave two classes the same colour
once there were more classes than colours — the figure still drew, the legend
still listed every class, and two of them were simply indistinguishable.
"""
from __future__ import annotations

import pytest

from bioaccx.umap import distinct_colors, hex_colors, min_gap


@pytest.mark.parametrize("count", [1, 2, 9, 10, 11, 20, 21, 23, 30, 45, 64, 97])
def test_no_two_classes_share_a_colour(count):
    colors = distinct_colors(count)
    assert len(colors) == count
    assert len(set(colors)) == count, "two classes were given the same colour"


#: What the palettes this replaced managed, measured in Oklab ΔE: tab10 0.113,
#: tab20 0.054, the browser's twelve hand-picked colours 0.047. Anything under
#: about 0.08 reads as the same colour in a scatter plot of small dots.
@pytest.mark.parametrize("count,floor", [
    (5, 0.20), (8, 0.17), (10, 0.15), (12, 0.15), (16, 0.11), (20, 0.10), (24, 0.10),
])
def test_colours_stay_far_enough_apart_to_tell_apart(count, floor):
    gap = min_gap(distinct_colors(count))
    assert gap >= floor, f"{count} classes: closest pair only ΔE {gap:.3f} apart"


def test_it_beats_the_colormaps_it_replaced():
    """The regression that prompted this: similar colours well under 20 labels."""
    pytest.importorskip("matplotlib", reason="the colormaps need the [umap] extra")
    from matplotlib import pyplot as plt

    for count, name in [(10, "tab10"), (20, "tab20")]:
        curated = [tuple(c) for c in plt.get_cmap(name).colors[:count]]
        assert min_gap(distinct_colors(count)) > min_gap(curated) * 1.3, \
            f"no better than {name} at {count} classes"


def test_a_colour_belongs_to_its_class_not_to_the_run(count=40):
    """Same class count, same colours — so two runs can be compared by eye."""
    assert distinct_colors(count) == distinct_colors(count)


def test_hex_is_the_same_palette_the_browser_gets():
    colours = hex_colors(15)
    assert len(set(colours)) == 15
    assert all(len(c) == 7 and c.startswith("#") for c in colours)
    assert colours[0] == "#%02x%02x%02x" % tuple(
        round(channel * 255) for channel in distinct_colors(15)[0])


def test_generated_colours_stay_inside_the_visible_range(count=60):
    """Nothing so dark or so pale that it disappears against the figure."""
    for red, green, blue in distinct_colors(count):
        assert 0.0 <= min(red, green, blue) and max(red, green, blue) <= 1.0
        brightness = 0.299 * red + 0.587 * green + 0.114 * blue
        assert 0.1 < brightness < 0.9, "a generated colour is nearly black or white"


def test_consecutive_classes_are_not_neighbouring_colours(count=24):
    """Reading a legend top to bottom must not walk through one hue."""
    colors = distinct_colors(count)
    adjacent = min(min_gap([a, b]) for a, b in zip(colors, colors[1:]))
    assert adjacent > 0.25, f"neighbours in the legend only ΔE {adjacent:.3f} apart"
