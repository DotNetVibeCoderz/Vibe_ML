"""
Generates golden reference outputs with the OFFICIAL Google MediaPipe Python package, used by
MediaPipe.NET's cross-validation tests (tests/MediaPipeNet.Tasks.Tests). Each task runs on the
fixture images in tests/assets/images using the original TFLite / .task models, and the results are
written as JSON to tests/assets/golden/.

    python -m venv C:\\mpp && C:\\mpp\\Scripts\\pip install mediapipe
    C:\\mpp\\Scripts\\python tools/golden/generate_golden.py --models artifacts/models/_work
"""
from __future__ import annotations

import argparse
import json
import pathlib

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision

ROOT = pathlib.Path(__file__).resolve().parents[2]
IMAGES = ROOT / "tests" / "assets" / "images"
OUT = ROOT / "tests" / "assets" / "golden"


def lm(l, visibility=False):
    d = {"x": l.x, "y": l.y, "z": l.z}
    if visibility:
        d["visibility"] = l.visibility
        d["presence"] = l.presence
    return d


def cat(c):
    return {"index": c.index, "score": c.score, "name": c.category_name}


def det(d):
    b = d.bounding_box
    return {
        "box": {"x": b.origin_x, "y": b.origin_y, "width": b.width, "height": b.height},
        "categories": [cat(c) for c in d.categories],
        "keypoints": [{"x": k.x, "y": k.y} for k in (d.keypoints or [])],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(ROOT / "artifacts" / "models" / "_work"))
    args = ap.parse_args()
    models = pathlib.Path(args.models)
    OUT.mkdir(parents=True, exist_ok=True)
    base = lambda name: mpt.BaseOptions(model_asset_path=str(models / name))
    img = lambda name: mp.Image.create_from_file(str(IMAGES / name))
    out: dict[str, dict] = {}

    with vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(base_options=base("blaze_face_short_range.tflite"))) as t:
        out["face_detector"] = {n: [det(d) for d in t.detect(img(n)).detections] for n in ["portrait.jpg"]}

    with vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=base("face_landmarker.task"), output_face_blendshapes=True, num_faces=1)) as t:
        r = t.detect(img("portrait.jpg"))
        out["face_landmarker"] = {"portrait.jpg": {
            "landmarks": [[lm(l) for l in f] for f in r.face_landmarks],
            "blendshapes": [[cat(c) for c in f] for f in r.face_blendshapes]}}

    with vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=base("hand_landmarker.task"), num_hands=2)) as t:
        out["hand_landmarker"] = {}
        for n in ["thumb_up.jpg", "victory.jpg", "pointing_up.jpg"]:
            r = t.detect(img(n))
            out["hand_landmarker"][n] = {
                "landmarks": [[lm(l) for l in h] for h in r.hand_landmarks],
                "world": [[lm(l) for l in h] for h in r.hand_world_landmarks],
                "handedness": [[cat(c) for c in h] for h in r.handedness]}

    with vision.GestureRecognizer.create_from_options(vision.GestureRecognizerOptions(base_options=base("gesture_recognizer.task"))) as t:
        out["gesture_recognizer"] = {}
        for n in ["thumb_up.jpg", "victory.jpg", "pointing_up.jpg"]:
            r = t.recognize(img(n))
            out["gesture_recognizer"][n] = {"gestures": [[cat(c) for c in g] for g in r.gestures]}

    with vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(base_options=base("pose_landmarker_lite.task"))) as t:
        r = t.detect(img("pose.jpg"))
        out["pose_landmarker"] = {"pose.jpg": {"landmarks": [[lm(l, True) for l in p] for p in r.pose_landmarks]}}

    with vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=base("selfie_segmenter.tflite"), output_confidence_masks=True)) as t:
        r = t.segment(img("portrait.jpg"))
        m = r.confidence_masks[0].numpy_view()
        out["selfie_segmenter"] = {"portrait.jpg": {"width": int(m.shape[1]), "height": int(m.shape[0]),
                                                     "mean": float(np.mean(m)), "foreground_fraction": float(np.mean(m > 0.5))}}

    with vision.ObjectDetector.create_from_options(vision.ObjectDetectorOptions(
            base_options=base("efficientdet_lite0.tflite"), score_threshold=0.3)) as t:
        out["object_detector"] = {n: [det(d) for d in t.detect(img(n)).detections] for n in ["cats_and_dogs.jpg"]}

    with vision.ImageClassifier.create_from_options(vision.ImageClassifierOptions(
            base_options=base("efficientnet_lite0.tflite"), max_results=5)) as t:
        out["image_classifier"] = {n: [cat(c) for c in t.classify(img(n)).classifications[0].categories] for n in ["burger.jpg"]}

    path = OUT / "mediapipe_python_reference.json"
    path.write_text(json.dumps({"mediapipe_version": mp.__version__, "results": out}, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
