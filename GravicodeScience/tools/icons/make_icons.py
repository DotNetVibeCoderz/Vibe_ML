#!/usr/bin/env python3
"""Generates the NuGet package icons for every Gravicode.Science library.

The family system:

* Each library's tile colour is a real emission line - the colour a spectroscope sees when that
  element glows. GraviNum is H-alpha (656 nm), GraviFrame the sodium D doublet (589 nm), GraviLearn
  the mercury green line (546 nm), GraviText H-beta (486 nm), GraviGraph H-gamma (434 nm) and
  GraviProb calcium K (393 nm).
* Every tile carries the same lattice, bent towards one mass in the lower-right corner - the
  "Gravi" in the name. The bend is identical across the family, so the icons read as one set even
  at 32 px where the glyphs are all that is legible.
* SciNet, the SciPy layer that spans every domain, is the instrument rather than a line: an ink
  tile whose lattice and spectrum strip carry all six colours.

Usage:
    python tools/icons/make_icons.py          # writes assets/icons/*.svg, renders *.png and preview.png

PNG rendering uses a headless Chrome or Edge with a transparent background; no Python packages
are needed beyond the standard library.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "assets", "icons")
SIZE = 256

INK = "#16132E"
PAPER = "#FBF8F2"

# name -> (tile colour, glyph colour, line label)
LINES = {
    "GraviNum":   ("#D9363E", PAPER, "H\u03b1 656 nm"),
    "GraviFrame": ("#E8A317", INK,   "Na D 589 nm"),
    "GraviLearn": ("#2E9E68", PAPER, "Hg 546 nm"),
    "GraviText":  ("#1B95BC", PAPER, "H\u03b2 486 nm"),
    "GraviGraph": ("#4F46D6", PAPER, "H\u03b3 434 nm"),
    "GraviProb":  ("#9634C8", PAPER, "Ca K 393 nm"),
}
WAVELENGTHS = {"GraviNum": 656, "GraviFrame": 589, "GraviLearn": 546,
               "GraviText": 486, "GraviGraph": 434, "GraviProb": 393}

MASS = (204.0, 206.0)   # where the lattice bends to; the same point on every tile


def warp(x: float, y: float) -> tuple[float, float]:
    """Pulls a point towards MASS, strongly near it and gently far away."""
    dx, dy = MASS[0] - x, MASS[1] - y
    r2 = dx * dx + dy * dy
    pull = 2600.0 / (r2 + 2400.0)
    return x + dx * pull, y + dy * pull


def lattice(stroke: str | list[str], opacity: float, width: float = 1.6) -> str:
    """The bent grid, as polylines. A list of strokes colours the lines in turn."""
    lines = []
    step = 24
    coords = range(-step, SIZE + 2 * step, step)
    colours = stroke if isinstance(stroke, list) else [stroke]
    k = 0
    for fixed in coords:
        for vertical in (True, False):
            pts = []
            for t in range(-step, SIZE + step + 1, 4):
                x, y = (fixed, t) if vertical else (t, fixed)
                wx, wy = warp(x, y)
                pts.append(f"{wx:.1f},{wy:.1f}")
            colour = colours[k % len(colours)]
            k += 1
            lines.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{colour}" '
                         f'stroke-opacity="{opacity}" stroke-width="{width}"/>')
    return "\n    ".join(lines)


def tile(body: str, colour: str, lattice_svg: str, mass_fill: str, title: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIZE} {SIZE}" width="{SIZE}" height="{SIZE}">
  <title>{title}</title>
  <defs>
    <clipPath id="tile"><rect width="{SIZE}" height="{SIZE}" rx="54"/></clipPath>
    <radialGradient id="well" cx="{MASS[0] / SIZE:.3f}" cy="{MASS[1] / SIZE:.3f}" r="0.62">
      <stop offset="0" stop-color="#000" stop-opacity="0.34"/>
      <stop offset="0.55" stop-color="#000" stop-opacity="0.08"/>
      <stop offset="1" stop-color="#000" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <g clip-path="url(#tile)">
    <rect width="{SIZE}" height="{SIZE}" fill="{colour}"/>
    <rect width="{SIZE}" height="{SIZE}" fill="url(#well)"/>
    {lattice_svg}
    <circle cx="{MASS[0]}" cy="{MASS[1]}" r="9" fill="{mass_fill}"/>
  </g>
  {body}
</svg>
"""


# ---------------------------------------------------------------- glyphs (centred near 116,112)

def glyph_num(c: str, bg: str) -> str:
    """An n-dimensional array: three stacked 3x3 sheets."""
    parts = []
    for depth, (ox, oy, op) in enumerate([(34, -30, 0.45), (17, -15, 0.7), (0, 0, 1.0)]):
        x0, y0, s = 52 + ox, 82 + oy, 96
        parts.append(f'<rect x="{x0}" y="{y0}" width="{s}" height="{s}" rx="10" fill="{bg}" '
                     f'stroke="{c}" stroke-opacity="{op}" stroke-width="9"/>')
    x0, y0, s = 52, 82, 96
    for i in (1, 2):
        parts.append(f'<line x1="{x0 + s * i / 3:.1f}" y1="{y0}" x2="{x0 + s * i / 3:.1f}" y2="{y0 + s}" '
                     f'stroke="{c}" stroke-width="6"/>')
        parts.append(f'<line x1="{x0}" y1="{y0 + s * i / 3:.1f}" x2="{x0 + s}" y2="{y0 + s * i / 3:.1f}" '
                     f'stroke="{c}" stroke-width="6"/>')
    return "\n  ".join(parts)


def glyph_frame(c: str, bg: str) -> str:
    """A table: header band, index column, rows."""
    x0, y0, w, h = 44, 50, 146, 128
    parts = [f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" rx="14" fill="none" stroke="{c}" stroke-width="9"/>',
             f'<path d="M{x0} {y0 + 14} a14 14 0 0 1 14 -14 h{w - 28} a14 14 0 0 1 14 14 v20 h-{w} z" fill="{c}"/>',
             f'<rect x="{x0}" y="{y0 + 34}" width="34" height="{h - 34}" fill="{c}" fill-opacity="0.32"/>',
             f'<line x1="{x0 + 34}" y1="{y0 + 30}" x2="{x0 + 34}" y2="{y0 + h}" stroke="{c}" stroke-width="6"/>',
             f'<line x1="{x0 + 90}" y1="{y0 + 30}" x2="{x0 + 90}" y2="{y0 + h}" stroke="{c}" stroke-width="5" stroke-opacity="0.6"/>']
    for r in (1, 2):
        y = y0 + 34 + (h - 34) * r / 3
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + w}" y2="{y:.1f}" stroke="{c}" stroke-width="5" stroke-opacity="0.6"/>')
    return "\n  ".join(parts)


def glyph_learn(c: str, bg: str) -> str:
    """Two classes split by a curved decision boundary."""
    filled = [(60, 70), (92, 58), (66, 112), (50, 154)]
    hollow = [(160, 84), (150, 138), (184, 120), (146, 182)]
    parts = [f'<path d="M118 36 C 96 96, 150 112, 96 200" fill="none" stroke="{c}" stroke-width="7" '
             f'stroke-linecap="round" stroke-opacity="0.85"/>']
    parts += [f'<circle cx="{x}" cy="{y}" r="15" fill="{c}"/>' for x, y in filled]
    parts += [f'<circle cx="{x}" cy="{y}" r="12" fill="none" stroke="{c}" stroke-width="7"/>' for x, y in hollow]
    return "\n  ".join(parts)


def glyph_text(c: str, bg: str) -> str:
    """Tokens on three lines, one of them masked - the question a language model answers."""
    rows = [(58, [44, 30, 52]), (106, [28, 58, 34]), (154, [50, 40])]
    parts, k = [], 0
    for y, widths in rows:
        x = 40
        for w in widths:
            if k == 4:   # the masked token
                parts.append(f'<rect x="{x + 3}" y="{y + 3}" width="{w - 6}" height="22" rx="11" fill="none" '
                             f'stroke="{c}" stroke-width="6" stroke-dasharray="7 6"/>')
            else:
                parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="28" rx="14" fill="{c}"/>')
            x += w + 10
            k += 1
    return "\n  ".join(parts)


def glyph_graph(c: str, bg: str) -> str:
    """A small graph with one hub."""
    nodes = {"h": (112, 112, 22), "a": (52, 62, 13), "b": (176, 58, 13), "c": (50, 168, 13),
             "d": (164, 176, 13), "e": (196, 118, 10)}
    edges = [("h", "a"), ("h", "b"), ("h", "c"), ("h", "d"), ("b", "e"), ("d", "e"), ("a", "c")]
    parts = [f'<line x1="{nodes[u][0]}" y1="{nodes[u][1]}" x2="{nodes[v][0]}" y2="{nodes[v][1]}" '
             f'stroke="{c}" stroke-width="7" stroke-linecap="round"/>' for u, v in edges]
    for name, (x, y, r) in nodes.items():
        parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{c if name != "h" else bg}" '
                     f'stroke="{c}" stroke-width="{8 if name == "h" else 0}"/>')
    return "\n  ".join(parts)


def glyph_prob(c: str, bg: str) -> str:
    """A posterior with its highest-density interval shaded."""
    def curve(x):
        return 178 - 118 * math.exp(-((x - 112) / 34.0) ** 2 / 2)
    xs = [36 + i for i in range(0, 157, 2)]
    path = " ".join(f"{'M' if i == 0 else 'L'}{x} {curve(x):.1f}" for i, x in enumerate(xs))
    hdi = [x for x in xs if 64 <= x <= 160]
    area = f"M{hdi[0]} 184 " + " ".join(f"L{x} {curve(x):.1f}" for x in hdi) + f" L{hdi[-1]} 184 Z"
    return "\n  ".join([
        f'<path d="{area}" fill="{c}" fill-opacity="0.35"/>',
        f'<path d="{path}" fill="none" stroke="{c}" stroke-width="9" stroke-linecap="round" stroke-linejoin="round"/>',
        f'<line x1="36" y1="186" x2="188" y2="186" stroke="{c}" stroke-width="7" stroke-linecap="round"/>',
        f'<line x1="112" y1="66" x2="112" y2="186" stroke="{c}" stroke-width="5" stroke-dasharray="6 8"/>',
    ])


def wavelength_hex(name: str) -> str:
    return LINES[name][0]


def scinet() -> str:
    """The instrument: an integral over a damped wave, above the family's emission spectrum."""
    colours = [wavelength_hex(n) for n in LINES]
    lat = lattice(colours, 0.30, 1.5)
    xs = [i for i in range(94, 209, 2)]
    wave = " ".join(f"{'M' if i == 0 else 'L'}{x} {112 - 40 * math.exp(-(x - 94) / 64) * math.sin((x - 94) / 9.0):.1f}"
                    for i, x in enumerate(xs))
    # spectrum strip: a dark band with the six lines at their real relative wavelengths
    x0, x1, lo, hi = 46, 210, 380, 680
    strip = [f'<rect x="{x0 - 8}" y="176" width="{x1 - x0 + 16}" height="30" rx="8" fill="#0B0A1A" stroke="{PAPER}" stroke-opacity="0.25" stroke-width="2"/>']
    for name in LINES:
        wl = WAVELENGTHS[name]
        x = x0 + (wl - lo) / (hi - lo) * (x1 - x0)
        strip.append(f'<rect x="{x - 2.5:.1f}" y="180" width="5" height="22" rx="1.5" fill="{wavelength_hex(name)}"/>')
    body = "\n  ".join([
        # the integral sign, drawn rather than typeset so it renders identically everywhere
        f'<path d="M92 40 C 72 30, 60 50, 64 76 L 74 136 C 78 164, 62 176, 44 164" fill="none" '
        f'stroke="{PAPER}" stroke-width="13" stroke-linecap="round"/>',
        f'<path d="{wave}" fill="none" stroke="{PAPER}" stroke-width="8" stroke-linecap="round" stroke-linejoin="round"/>',
        *strip,
    ])
    return tile(body, INK, lat, "#0B0A1A", "SciNet").replace(
        f'<circle cx="{MASS[0]}" cy="{MASS[1]}" r="9" fill="#0B0A1A"/>', "")


GLYPHS = {"GraviNum": glyph_num, "GraviFrame": glyph_frame, "GraviLearn": glyph_learn,
          "GraviText": glyph_text, "GraviGraph": glyph_graph, "GraviProb": glyph_prob}


def library(name: str) -> str:
    colour, glyph, _ = LINES[name]
    lat = lattice(glyph, 0.20)
    return tile(GLYPHS[name](glyph, colour), colour, lat, glyph, name)


def find_browser() -> str | None:
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        "google-chrome", "chromium", "chromium-browser", "microsoft-edge",
    ]
    for c in candidates:
        if os.path.isfile(c) or shutil.which(c):
            return c
    return None


def render(svg_path: str, png_path: str, browser: str) -> None:
    # Chrome sizes the window including nothing else in headless=new, so the SVG fills it exactly.
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run([browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        f"--user-data-dir={profile}", "--default-background-color=00000000",
                        f"--window-size={SIZE},{SIZE}", f"--screenshot={png_path}",
                        "file:///" + svg_path.replace("\\", "/")],
                       check=True, capture_output=True, timeout=60)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    svgs = {name: library(name) for name in LINES}
    svgs["SciNet"] = scinet()
    browser = find_browser()
    for name, svg in svgs.items():
        path = os.path.join(OUT, f"{name}.svg")
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg)
        if browser:
            render(path, os.path.join(OUT, f"{name}.png"), browser)
        print(f"  {name}")
    if not browser:
        print("No Chrome/Edge found: SVGs written, PNGs not rendered.", file=sys.stderr)
    else:
        preview(list(svgs))
    return 0


def preview(names: list[str]) -> None:
    """The strip shown in the README. Needs Pillow; skipped quietly without it."""
    try:
        from PIL import Image
    except ImportError:
        return
    size, gap = 112, 16
    icons = [Image.open(os.path.join(OUT, f"{n}.png")).convert("RGBA").resize((size, size), Image.LANCZOS) for n in names]
    sheet = Image.new("RGBA", (len(icons) * size + (len(icons) - 1) * gap, size), (0, 0, 0, 0))
    for i, icon in enumerate(icons):
        sheet.alpha_composite(icon, (i * (size + gap), 0))
    sheet.save(os.path.join(OUT, "preview.png"))


if __name__ == "__main__":
    sys.exit(main())
