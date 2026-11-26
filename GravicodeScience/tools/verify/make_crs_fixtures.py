"""Builds the coordinate-system fixtures DaskNet.Geo's metric geometry tests are pinned against.

Two independent references, neither of them DaskNet's own code:
- PROJ (through pyproj) for the projections: the same local transverse Mercator DaskNet uses
  (centred on each geometry's centroid, scale 1), Web Mercator and the UTM zones;
- geodesic areas and lengths on the WGS 84 ellipsoid (pyproj.Geod, i.e. GeographicLib), which
  bound how far the local-plane measures may be from the true ones.

    python -m venv venv && venv/Scripts/pip install shapely pyproj
    venv/Scripts/python tools/verify/make_crs_fixtures.py

Output: tests/DaskNet.Geo.Tests/Fixtures/crs.json
"""
import json
import math
import os

import pyproj
import shapely
from shapely.geometry import LineString, Polygon, mapping
from shapely.ops import transform

OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'DaskNet.Geo.Tests', 'Fixtures', 'crs.json')
GEOD = pyproj.Geod(ellps='WGS84')


def local(g):
    """The local plane DaskNet uses: tmerc at the lon/lat centroid, scale 1."""
    c = g.centroid
    crs = pyproj.CRS.from_proj4(f'+proj=tmerc +lat_0=0 +lon_0={c.x!r} +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs')
    fwd = pyproj.Transformer.from_crs('EPSG:4326', crs, always_xy=True)
    inv = pyproj.Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
    return (lambda h: transform(fwd.transform, h)), (lambda h: transform(inv.transform, h))


def blob(lon, lat, radius_km, n, seed):
    """A star-shaped polygon whose vertices are true geodesic distances from the centre."""
    pts = []
    for k in range(n):
        az = 360.0 * k / n
        r = radius_km * 1000 * (0.65 + 0.35 * ((seed * 7919 + k * 104729) % 1000) / 1000)
        x, y, _ = GEOD.fwd(lon, lat, az, r)
        pts.append((round(x, 7), round(y, 7)))
    return Polygon(pts)


def measures(g):
    to_plane, _ = local(g)
    p = to_plane(g)
    if g.geom_type in ('LineString', 'MultiLineString'):
        geodesic_area, geodesic_length = 0.0, GEOD.geometry_length(g)
    else:
        a, perimeter = GEOD.geometry_area_perimeter(g)
        geodesic_area, geodesic_length = abs(a), perimeter
    return {'area': p.area, 'length': p.length, 'geodesic_area': geodesic_area, 'geodesic_length': geodesic_length}


def main():
    shapes = [
        ('jakarta', blob(106.8456, -6.2088, 3, 9, 1)),
        ('oslo', blob(10.7522, 59.9139, 5, 11, 2)),
        ('sydney', blob(151.2093, -33.8688, 2, 7, 3)),
        ('quito', blob(-78.4678, -0.1807, 8, 12, 4)),
        ('reykjavik', blob(-21.9426, 64.1466, 4, 10, 5)),
        ('tromso', blob(18.9553, 69.6492, 1.5, 8, 6)),
        ('bandung_region', blob(107.6191, -6.9175, 90, 16, 7)),
    ]
    holed = blob(-0.1276, 51.5072, 6, 12, 8).difference(blob(-0.1276, 51.5072, 1.5, 6, 9))
    shapes.append(('london_ring', holed))
    shapes.append(('jakarta_bogor', LineString([(106.8456, -6.2088), (106.83, -6.35), (106.81, -6.45), (106.8, -6.55), (106.7975, -6.595)])))

    rows = []
    for name, g in shapes:
        to_plane, back = local(g)
        p = to_plane(g)
        row = {'name': name, 'wkt': shapely.to_wkt(g, rounding_precision=-1), **measures(g)}
        b = back(p.buffer(500, quad_segs=8))
        row['buffer_500'] = measures(b)
        s = back(shapely.simplify(p, 150, preserve_topology=True))
        row['simplify_150'] = measures(s)
        c = back(p.centroid)
        row['centroid'] = [c.x, c.y]
        rows.append(row)

    points = [(106.8456, -6.2088), (113.99, -7.5), (114.01, -7.5), (10.7522, 59.9139), (-70.6693, -33.4489),
              (151.2093, -33.8688), (15.6356, 78.2232), (-179.9, 0.5), (0.0, 0.0), (-74.006, 40.7128)]
    conversions = []
    for lon, lat in points:
        zone = int(math.floor((lon + 180) / 6)) + 1
        epsg = (32600 if lat >= 0 else 32700) + zone
        utm = pyproj.Transformer.from_crs('EPSG:4326', f'EPSG:{epsg}', always_xy=True).transform(lon, lat)
        web = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True).transform(lon, lat)
        conversions.append({'lon': lon, 'lat': lat, 'utm_epsg': epsg, 'utm': list(utm), 'web': list(web)})

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8', newline='\n') as f:
        json.dump({'writer': f'pyproj {pyproj.__version__} / PROJ {pyproj.proj_version_str} / shapely {shapely.__version__}',
                   'shapes': rows, 'conversions': conversions}, f, indent=1)
    for r in rows:
        rel = abs(r['area'] - r['geodesic_area']) / r['geodesic_area'] if r['geodesic_area'] else 0
        print(f"{r['name']:16} area {r['area']:.6e}  geodesic rel {rel:.2e}  length rel {abs(r['length'] - r['geodesic_length']) / r['geodesic_length']:.2e}")
    print('wrote', OUT)


if __name__ == '__main__':
    main()
