"""
Extracts the canonical face geometry used for the facial transformation matrix from face_landmarker.task
(geometry_pipeline_metadata_landmarks.binarypb, a GeometryPipelineMetadata protobuf) and writes
src/MediaPipeNet.Tasks.Vision/Resources/face_geometry.bin:

    int32 count (468) | count x float32 (x, y, z) canonical metric landmarks | count x float32 Procrustes weights

    python tools/model-conversion/extract_face_geometry.py --task artifacts/models/_work/face_landmarker.task
"""
from __future__ import annotations

import argparse
import pathlib
import struct
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]


def varint(buf: bytes, i: int) -> tuple[int, int]:
    shift = result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, i
        shift += 7


def fields(buf: bytes):
    """Yields (field_number, wire_type, value) for a protobuf message."""
    i = 0
    while i < len(buf):
        key, i = varint(buf, i)
        num, wt = key >> 3, key & 7
        if wt == 0:
            v, i = varint(buf, i)
        elif wt == 1:
            v, i = buf[i:i + 8], i + 8
        elif wt == 2:
            n, i = varint(buf, i)
            v, i = buf[i:i + n], i + n
        elif wt == 5:
            v, i = buf[i:i + 4], i + 4
        else:
            raise ValueError(f"unsupported wire type {wt}")
        yield num, wt, v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=str(ROOT / "artifacts/models/_work/face_landmarker.task"))
    args = ap.parse_args()
    with zipfile.ZipFile(args.task) as z:
        data = z.read("geometry_pipeline_metadata_landmarks.binarypb")

    weights: dict[int, float] = {}
    vertex_buffer: list[float] = []
    for num, wt, v in fields(data):
        if num == 2 and wt == 2:  # WeightedLandmarkRef procrustes_landmark_basis
            lid, w = 0, 0.0
            for n2, w2, v2 in fields(v):
                if n2 == 1:
                    lid = v2
                elif n2 == 2:
                    w = struct.unpack("<f", v2)[0]
            weights[lid] = w
        elif num == 1 and wt == 2:  # Mesh3d canonical_mesh
            for n2, w2, v2 in fields(v):
                if n2 == 3 and w2 == 2:  # packed float vertex_buffer
                    vertex_buffer += list(struct.unpack(f"<{len(v2) // 4}f", v2))
                elif n2 == 3 and w2 == 5:
                    vertex_buffer.append(struct.unpack("<f", v2)[0])

    vertex_size = 5  # VERTEX_PT: x, y, z, u, v
    count = len(vertex_buffer) // vertex_size
    print(f"{count} vertices, {len(weights)} weighted landmarks")
    out = ROOT / "src/MediaPipeNet.Tasks.Vision/Resources/face_geometry.bin"
    with out.open("wb") as f:
        f.write(struct.pack("<i", count))
        for i in range(count):
            f.write(struct.pack("<3f", *vertex_buffer[i * vertex_size:i * vertex_size + 3]))
        for i in range(count):
            f.write(struct.pack("<f", weights.get(i, 0.0)))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
