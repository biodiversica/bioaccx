"""The logo the installers put on the shortcuts and the GUI shows in its tab."""
from __future__ import annotations

from pathlib import Path

import pytest

ASSETS = Path(__file__).resolve().parents[2] / "bioaccx" / "assets"


@pytest.mark.parametrize("name,magic", [
    ("bioaccx.png", b"\x89PNG"),
    ("bioaccx.ico", b"\x00\x00\x01\x00"),
    ("bioaccx.svg", b"<svg"),
])
def test_the_logo_ships_inside_the_package(name, magic):
    """The installers read it from the installed package, so it must be there."""
    assert (ASSETS / name).read_bytes().startswith(magic)


def test_the_ico_holds_the_sizes_windows_asks_for():
    """Read from the ICO directory itself: a count, then 16 bytes per image,
    starting with its width (0 meaning 256)."""
    data = (ASSETS / "bioaccx.ico").read_bytes()
    count = int.from_bytes(data[4:6], "little")
    widths = {data[6 + 16 * i] or 256 for i in range(count)}
    assert {16, 32, 48, 256} <= widths
