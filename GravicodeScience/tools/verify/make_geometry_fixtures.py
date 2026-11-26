"""Builds the geometry fixtures DaskNet.Geo's buffer/dissolve/overlay tests are pinned against.

The expected numbers come from shapely (GEOS), not from NetTopologySuite, so the .NET results are
checked against an independent implementation of the same operations.

    python -m venv venv && venv/Scripts/pip install shapely
    venv/Scripts/python tools/verify/make_geometry_fixtures.py

Output: tests/DaskNet.Geo.Tests/Fixtures/geometry.json
"""
import json
import math
import os
import random

import shapely
from shapely import wkt
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'DaskNet.Geo.Tests', 'Fixtures', 'geometry.json')


def blob(rng, cx, cy, r, n):
    """A star-shaped polygon: random radii around a centre, so it is simple but not convex."""
    pts = []
    for k in range(n):
        a = 2 * math.pi * k / n
        rr = r * (0.6 + 0.4 * rng.random())
        pts.append((round(cx + rr * math.cos(a), 6), round(cy + rr * math.sin(a), 6)))
    return Polygon(pts)


def main():
    rng = random.Random(20261127)
    left = []
    for i in range(60):
        g = blob(rng, rng.uniform(0, 20), rng.uniform(0, 20), rng.uniform(0.5, 2.0), rng.randint(5, 12))
        if i % 10 == 0:
            g = g.difference(box(g.centroid.x - 0.1, g.centroid.y - 0.1, g.centroid.x + 0.1, g.centroid.y + 0.1))  # a hole
        left.append({'id': i, 'zone': ['a', 'b', 'c'][i % 3], 'wkt': g.wkt})
    right = []
    for j in range(15):
        x, y = rng.uniform(0, 18), rng.uniform(0, 18)
        right.append({'rid': j, 'wkt': box(x, y, x + rng.uniform(1, 4), y + rng.uniform(1, 4)).wkt})

    lg = [wkt.loads(r['wkt']) for r in left]
    rg = [wkt.loads(r['wkt']) for r in right]

    buffer = [g.buffer(0.3, quad_segs=8).area for g in lg]
    shrink = [g.buffer(-0.2, quad_segs=8).area for g in lg]
    simplify = [shapely.simplify(g, 0.25, preserve_topology=True).area for g in lg]
    centroid = [[g.centroid.x, g.centroid.y] for g in lg]
    dissolve = {}
    for zone in 'abc':
        u = unary_union([g for g, r in zip(lg, left) if r['zone'] == zone])
        dissolve[zone] = {'area': u.area, 'length': u.length, 'count': sum(1 for r in left if r['zone'] == zone)}
    everything = unary_union(lg)
    pairs = []
    for i, a in enumerate(lg):
        for j, b in enumerate(rg):
            if a.intersects(b):
                s = a.intersection(b)
                if s.area > 0:
                    pairs.append({'id': i, 'rid': j, 'area': s.area})
    difference = []
    for i, a in enumerate(lg):
        hits = [b for b in rg if b.intersects(a)]
        rest = a.difference(unary_union(hits)) if hits else a
        difference.append({'id': i, 'area': rest.area, 'empty': rest.is_empty})

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({
            'writer': f'shapely {shapely.__version__} / GEOS {shapely.geos_version_string}',
            'left': left, 'right': right,
            'buffer': buffer, 'shrink': shrink, 'simplify': simplify, 'centroid': centroid,
            'dissolve': dissolve, 'union_area': everything.area,
            'intersection': pairs, 'difference': difference,
        }, f, indent=1)
    print('wrote', OUT, len(pairs), 'intersecting pairs')


if __name__ == '__main__':
    main()
