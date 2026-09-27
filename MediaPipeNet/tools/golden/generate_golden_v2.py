"""
Golden references for the MediaPipe.NET 0.2 / 0.3 tasks, produced with the OFFICIAL MediaPipe Python package.
Writes tests/assets/golden/mediapipe_python_reference_v2.json.

    C:\\mpp\\Scripts\\python tools/golden/generate_golden_v2.py --models artifacts/models/_work
"""
from __future__ import annotations

import argparse
import json
import pathlib

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import audio, text, vision
from mediapipe.tasks.python.components.containers import audio_data as audio_data_module
from mediapipe.tasks.python.components.containers import keypoint as keypoint_module

ROOT = pathlib.Path(__file__).resolve().parents[2]
IMAGES = ROOT / "tests" / "assets" / "images"
AUDIO = ROOT / "tests" / "assets" / "audio"
OUT = ROOT / "tests" / "assets" / "golden"

SENTENCES = [
    "It's beautiful outside.",
    "The movie was a complete waste of time, boring and far too long.",
    "Selamat pagi, apa kabar hari ini?",
    "Il fait très beau aujourd'hui.",
    "Das ist ein wunderbares Buch.",
]


def lms(ls):
    return [{"x": l.x, "y": l.y, "z": l.z} for l in ls]


def cat(c):
    return {"index": c.index, "score": c.score, "name": c.category_name}


def det(d):
    b = d.bounding_box
    return {"box": {"x": b.origin_x, "y": b.origin_y, "width": b.width, "height": b.height},
            "score": d.categories[0].score,
            "keypoints": [{"x": k.x, "y": k.y} for k in (d.keypoints or [])]}


def mask_stats(m):
    a = m.numpy_view()
    if a.dtype == np.uint8:
        vals, counts = np.unique(a, return_counts=True)
        return {"width": int(a.shape[1]), "height": int(a.shape[0]),
                "histogram": {int(v): float(c) / a.size for v, c in zip(vals, counts)}}
    return {"width": int(a.shape[1]), "height": int(a.shape[0]), "mean": float(a.mean()), "fraction": float((a > 0.5).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(ROOT / "artifacts" / "models" / "_work"))
    args = ap.parse_args()
    models = pathlib.Path(args.models)
    base = lambda name: mpt.BaseOptions(model_asset_path=str(models / name))
    img = lambda name: mp.Image.create_from_file(str(IMAGES / name))
    out: dict[str, dict] = {}

    errors = {}

    def run(fn):
        try:
            fn()
        except Exception as e:  # task not available in this MediaPipe build
            errors[fn.__name__] = f"{type(e).__name__}: {e}"
            print("skipped", fn.__name__, errors[fn.__name__])

    # --- 0.2.0 -------------------------------------------------------------------------------------
    def section_1():
        with vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(base_options=base("blaze_face_full_range.tflite"))) as t:
            out["face_detector_full_range"] = {n: [det(d) for d in t.detect(img(n)).detections] for n in ["pose.jpg", "portrait.jpg"]}
    
    run(section_1)

    def section_2():
        with vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                base_options=base("face_landmarker.task"), output_facial_transformation_matrixes=True)) as t:
            r = t.detect(img("portrait.jpg"))
            out["face_transformation_matrix"] = {"portrait.jpg": np.asarray(r.facial_transformation_matrixes[0]).tolist()}
    
    run(section_2)

    def section_3():
        with vision.HolisticLandmarker.create_from_options(vision.HolisticLandmarkerOptions(
                base_options=base("holistic_landmarker.task"))) as t:
            out["holistic_landmarker"] = {}
            for n in ["pose.jpg"]:
                r = t.detect(img(n))
                out["holistic_landmarker"][n] = {
                    "pose": lms(r.pose_landmarks),
                    "face": lms(r.face_landmarks),
                    "left_hand": lms(r.left_hand_landmarks),
                    "right_hand": lms(r.right_hand_landmarks),
                }
    
    run(section_3)

    # --- 0.3.0 -------------------------------------------------------------------------------------
    def section_4():
        with vision.ImageEmbedder.create_from_options(vision.ImageEmbedderOptions(
                base_options=base("mobilenet_v3_small.tflite"), l2_normalize=True)) as t:
            e = {n: t.embed(img(n)).embeddings[0] for n in ["burger.jpg", "burger_crop.jpg", "cat.jpg"]}
            out["image_embedder"] = {
                "dim": len(e["burger.jpg"].embedding),
                "first8": {n: e[n].embedding[:8].tolist() for n in e},
                "similarity_burger_burger_crop": vision.ImageEmbedder.cosine_similarity(e["burger.jpg"], e["burger_crop.jpg"]),
                "similarity_burger_cat": vision.ImageEmbedder.cosine_similarity(e["burger.jpg"], e["cat.jpg"]),
            }
    
    run(section_4)

    def section_5():
        out["image_segmenter"] = {}
        for model, image in [("selfie_multiclass_256x256.tflite", "portrait.jpg"), ("hair_segmenter.tflite", "portrait.jpg"),
                             ("deeplab_v3.tflite", "cats_and_dogs.jpg")]:
            with vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
                    base_options=base(model), output_category_mask=True, output_confidence_masks=True)) as t:
                r = t.segment(img(image))
                out["image_segmenter"][model] = {"image": image, "category": mask_stats(r.category_mask),
                                                 "confidence": [mask_stats(m) for m in r.confidence_masks]}
    
    run(section_5)

    def section_6():
        with vision.InteractiveSegmenter.create_from_options(vision.InteractiveSegmenterOptions(
                base_options=base("magic_touch.tflite"))) as t:
            roi = vision.InteractiveSegmenterRegionOfInterest(
                format=vision.InteractiveSegmenterRegionOfInterest.Format.KEYPOINT,
                keypoint=keypoint_module.NormalizedKeypoint(0.62, 0.55))
            r = t.segment(img("cats_and_dogs.jpg"), roi)
            out["interactive_segmenter"] = {"cats_and_dogs.jpg": {"keypoint": [0.62, 0.55],
                "category": mask_stats(r.category_mask) if getattr(r, "category_mask", None) is not None else None,
                "confidence": [mask_stats(m) for m in (r.confidence_masks or [])]}}
    
    run(section_6)

    def section_7():
        with vision.FaceStylizer.create_from_options(vision.FaceStylizerOptions(base_options=base("face_stylizer_color_sketch.task"))) as t:
            r = t.stylize(img("portrait.jpg"))
            a = r.numpy_view()
            out["face_stylizer"] = {"portrait.jpg": {"width": int(a.shape[1]), "height": int(a.shape[0]),
                                                     "mean_rgb": a[..., :3].reshape(-1, 3).mean(axis=0).tolist()}}
            mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(a[..., :3])).numpy_view()
    
    run(section_7)

    def section_10():
        import wave
        with wave.open(str(AUDIO / "speech_16000_hz_mono.wav")) as w:
            rate = w.getframerate()
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        with audio.AudioClassifier.create_from_options(audio.AudioClassifierOptions(base_options=base("yamnet.tflite"), max_results=3)) as t:
            results = t.classify(audio_data_module.AudioData.create_from_array(pcm, rate))
            out["audio_classifier"] = {"speech_16000_hz_mono.wav": [
                {"timestamp_ms": r.timestamp_ms, "categories": [cat(c) for c in r.classifications[0].categories]} for r in results]}
    
    run(section_10)

    def section_11():
        out["text_classifier"] = {}
        for model in ["bert_classifier.tflite", "average_word_classifier.tflite"]:
            with text.TextClassifier.create_from_options(text.TextClassifierOptions(base_options=base(model))) as t:
                out["text_classifier"][model] = {s: [cat(c) for c in t.classify(s).classifications[0].categories] for s in SENTENCES[:2]}
    
    run(section_11)

    def section_12():
        with text.TextEmbedder.create_from_options(text.TextEmbedderOptions(base_options=base("bert_embedder.tflite"), l2_normalize=True)) as t:
            a, b, c = (t.embed(s).embeddings[0] for s in ["I love sunny days.", "Sunny weather makes me happy.", "The server crashed at midnight."])
            out["text_embedder"] = {"dim": len(a.embedding), "first8": a.embedding[:8].tolist(),
                                    "similarity_related": text.TextEmbedder.cosine_similarity(a, b),
                                    "similarity_unrelated": text.TextEmbedder.cosine_similarity(a, c)}
    
    run(section_12)

    def section_13():
        with text.LanguageDetector.create_from_options(text.LanguageDetectorOptions(base_options=base("language_detector.tflite"), max_results=3)) as t:
            out["language_detector"] = {s: [{"language": p.language_code, "probability": p.probability} for p in t.detect(s).detections]
                                        for s in SENTENCES}
    
    
    run(section_13)

    path = OUT / "mediapipe_python_reference_v2.json"
    path.write_text(json.dumps({"mediapipe_version": mp.__version__, "results": out, "unavailable": errors}, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
