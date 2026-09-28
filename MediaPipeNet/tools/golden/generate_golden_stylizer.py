"""
Golden reference for FaceStylizer, produced with the OFFICIAL MediaPipe Python package. MediaPipe 0.10.22+
dropped the face stylizer, so this needs its own environment:

    py -3.12 -m venv C:\\mpo && C:\\mpo\\Scripts\\pip install mediapipe==0.10.21
    C:\\mpo\\Scripts\\python tools/golden/generate_golden_stylizer.py --models artifacts/models/_work

The generator injects random noise, so single runs differ (mean |delta| ~ 20/255 per pixel). The reference is
the average of RUNS stylizations, downsampled to GRID x GRID, plus the spread of single runs around that
average; tests accept an output whose distance to the average is comparable to that spread.
Writes tests/assets/golden/mediapipe_face_stylizer_reference.json.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[2]
IMAGES = ROOT / "tests" / "assets" / "images"
OUT = ROOT / "tests" / "assets" / "golden" / "mediapipe_face_stylizer_reference.json"
RUNS = 16
GRID = 32


def downsample(rgb: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(rgb).resize((GRID, GRID), Image.BOX), np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="artifacts/models/_work")
    args = ap.parse_args()
    task = pathlib.Path(args.models) / "face_stylizer_color_sketch.task"
    stylizer = vision.FaceStylizer.create_from_options(
        vision.FaceStylizerOptions(base_options=BaseOptions(model_asset_path=str(task))))
    results = {}
    for name in ["portrait.jpg", "pose.jpg"]:
        image = mp.Image.create_from_file(str(IMAGES / name))
        runs = []
        for _ in range(RUNS):
            out = stylizer.stylize(image)
            if out is None:
                break
            runs.append(downsample(np.ascontiguousarray(out.numpy_view()[..., :3])))
        if not runs:
            results[name] = None
            continue
        mean = np.mean(runs, axis=0)
        spread = float(np.mean([np.abs(r - mean).mean() for r in runs]))
        results[name] = {
            "size": list(out.numpy_view().shape[:2]),
            "grid": GRID,
            "mean": [round(float(v), 1) for v in mean.reshape(-1)],
            "single_run_spread": spread,
            "mean_rgb": [float(v) for v in mean.reshape(-1, 3).mean(axis=0)],
        }
        print(name, "spread", spread, "mean rgb", results[name]["mean_rgb"])
    OUT.write_text(json.dumps({"mediapipe_version": mp.__version__, "runs": RUNS, "results": results}))


if __name__ == "__main__":
    main()
