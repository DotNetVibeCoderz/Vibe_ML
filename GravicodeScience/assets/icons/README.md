# Package icons

One icon per shipping library, packed as `icon.png` by `Directory.Build.props`
(`assets/icons/$(MSBuildProjectName).png`). Regenerate with:

```bash
python tools/icons/make_icons.py
```

The script writes the SVG sources and renders 256×256 PNGs with a headless Chrome or Edge.

## The system

| Library | Tile colour | Emission line | Glyph |
|---|---|---|---|
| GraviNum | `#D9363E` | H-alpha, 656 nm | stacked n-dimensional array |
| GraviFrame | `#E8A317` | sodium D, 589 nm | table with header and index column |
| GraviLearn | `#2E9E68` | mercury, 546 nm | two classes split by a decision boundary |
| GraviText | `#1B95BC` | H-beta, 486 nm | token lines with one masked token |
| GraviGraph | `#4F46D6` | H-gamma, 434 nm | a graph with one hub |
| GraviProb | `#9634C8` | calcium K, 393 nm | posterior with its interval shaded |
| SciNet | `#16132E` ink | the whole spectrum | an integral over a damped wave, above all six lines |

Every tile shares one lattice bent towards the same mass in the lower-right corner — the "Gravi"
in the names — so the family reads as a set even at 32 px, where only the glyph is legible.
SciNet is the instrument rather than a line: its strip places the six library colours at their
real relative wavelengths.

A new library needs a colour from a real emission line not yet used, a glyph function in
`make_icons.py`, and an entry in `LINES`/`GLYPHS`.

Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.
