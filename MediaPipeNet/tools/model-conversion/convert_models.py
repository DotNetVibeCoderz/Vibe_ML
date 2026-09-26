"""
MediaPipe.NET model conversion tool.

Downloads the official Google MediaPipe TFLite models (Apache-2.0), unpacks `.task` bundles,
converts every TFLite graph to ONNX with tf2onnx and writes a manifest (name, file, size, sha256,
inputs, outputs) that MediaPipe.NET's model registry is generated from.

Usage (Python 3.12, a short venv path is recommended on Windows):

    python -m venv C:\\mpv
    C:\\mpv\\Scripts\\pip install tensorflow-cpu==2.17.1 tf2onnx==1.16.1 protobuf==3.20.3 "numpy<2" onnx==1.16.2 onnxruntime
    C:\\mpv\\Scripts\\python tools/model-conversion/convert_models.py --out artifacts/models

Created by Gravicode Studios, led by Kang Fadhil.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys
import urllib.request
import zipfile

BASE = "https://storage.googleapis.com/mediapipe-models"

# (source file name, url). Bundles (.task) are zip files holding several .tflite graphs.
SOURCES = [
    ("blaze_face_short_range.tflite", f"{BASE}/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"),
    ("face_landmarker.task", f"{BASE}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"),
    ("hand_landmarker.task", f"{BASE}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"),
    ("pose_landmarker_lite.task", f"{BASE}/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"),
    ("pose_landmarker_full.task", f"{BASE}/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"),
    # The float16 pose_detector inside the .task bundle has a weight layout tf2onnx cannot read; the
    # float32 BlazePose detector published alongside it (same architecture) converts cleanly.
    ("pose_detection.tflite", "https://storage.googleapis.com/mediapipe-assets/pose_detection.tflite"),
    ("selfie_segmenter.tflite", f"{BASE}/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"),
    ("efficientdet_lite0.tflite", f"{BASE}/object_detector/efficientdet_lite0/float32/latest/efficientdet_lite0.tflite"),
    ("efficientnet_lite0.tflite", f"{BASE}/image_classifier/efficientnet_lite0/float32/latest/efficientnet_lite0.tflite"),
    ("gesture_recognizer.task", f"{BASE}/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task"),
]

# Output ONNX name for each TFLite graph (path inside bundle -> MediaPipe.NET model id).
RENAMES = {
    "blaze_face_short_range.tflite": "face_detection_short_range",
    "face_landmarker.task/face_detector.tflite": None,  # identical to blaze_face_short_range
    "face_landmarker.task/face_landmarks_detector.tflite": "face_landmarks_detector",
    "face_landmarker.task/face_blendshapes.tflite": "face_blendshapes",
    "hand_landmarker.task/hand_detector.tflite": "palm_detection",
    "hand_landmarker.task/hand_landmarks_detector.tflite": "hand_landmarks_detector",
    "pose_landmarker_lite.task/pose_detector.tflite": None,
    "pose_detection.tflite": "pose_detection",
    "pose_landmarker_lite.task/pose_landmarks_detector.tflite": "pose_landmarks_detector_lite",
    "pose_landmarker_full.task/pose_detector.tflite": None,
    "pose_landmarker_full.task/pose_landmarks_detector.tflite": "pose_landmarks_detector_full",
    "selfie_segmenter.tflite": "selfie_segmenter",
    "efficientdet_lite0.tflite": "efficientdet_lite0",
    "efficientnet_lite0.tflite": "efficientnet_lite0",
    "gesture_recognizer.task/hand_gesture_recognizer.task/gesture_embedder.tflite": "gesture_embedder",
    "gesture_recognizer.task/hand_gesture_recognizer.task/canned_gesture_classifier.tflite": "canned_gesture_classifier",
}


NO_INTERPRETER: set[str] = set()

# HardSwish needs opset 14.
OPSET_OVERRIDES = {"selfie_segmenter": 14}


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def densify(path: pathlib.Path) -> pathlib.Path:
    """
    BlazePose's detector stores its weights as sparse tensors expanded at load time by DENSIFY ops,
    which tf2onnx cannot read. The TFLite interpreter evaluates those ops during allocation, so the
    dense values are read back from it and written into the graph as plain constant buffers, and
    the DENSIFY ops are removed.
    """
    import numpy as np
    import tensorflow as tf
    from tensorflow.lite.python import schema_py_generated as schema
    from tensorflow.lite.python.interpreter import Interpreter
    from tensorflow.lite.tools import flatbuffer_utils

    model = flatbuffer_utils.read_model(str(path))
    densify_codes = {i for i, c in enumerate(model.operatorCodes)
                     if schema.BuiltinOperator.DENSIFY in (c.builtinCode, c.deprecatedBuiltinCode)}
    if not densify_codes:
        return path

    interp = Interpreter(str(path), experimental_preserve_all_tensors=True,
                                 experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    interp.allocate_tensors()
    # DENSIFY runs during the first invocation; before that its output buffer is uninitialized.
    for d in interp.get_input_details():
        interp.set_tensor(d["index"], np.zeros(d["shape"], dtype=d["dtype"]))
    interp.invoke()
    graph = model.subgraphs[0]
    kept = []
    for op in graph.operators:
        if op.opcodeIndex not in densify_codes:
            kept.append(op)
            continue
        src, dst = int(op.inputs[0]), int(op.outputs[0])
        dense = np.ascontiguousarray(interp.get_tensor(dst))
        buf = schema.BufferT()
        buf.data = np.frombuffer(dense.tobytes(), dtype=np.uint8)
        model.buffers.append(buf)
        graph.tensors[dst].buffer = len(model.buffers) - 1
        graph.tensors[src].sparsity = None
        graph.tensors[src].buffer = 0
        graph.tensors[src].shape = np.array([0], dtype=np.int32)
    graph.operators = kept
    out = path.with_name(path.stem + "_dense.tflite")
    flatbuffer_utils.write_model(model, str(out))
    print(f"densified {len(densify_codes)} op type(s) -> {out.name}")
    return out


def lower_transpose_conv_bias(onnx_path: pathlib.Path) -> None:
    """
    MediaPipe's segmentation models end with the custom TFLite op `Convolution2DTransposeBias`
    (a transposed convolution with a fused bias; SAME padding, stride 2), which tf2onnx leaves in the
    graph as an unknown node. It is rewritten here as NHWC->NCHW Transpose, ONNX ConvTranspose and
    NCHW->NHWC Transpose. The TFLite kernel layout [O,H,W,I] becomes ONNX's [I,O,H,W].
    """
    import onnx
    from onnx import helper

    model = onnx.load(str(onnx_path))
    graph = model.graph
    changed = False
    for idx, node in enumerate(list(graph.node)):
        if "Convolution2DTransposeBias" not in node.op_type:
            continue
        x, w, b = node.input[:3]
        y = node.output[0]
        n = node.name or f"conv_t_bias_{idx}"
        new_nodes = [
            helper.make_node("Transpose", [x], [f"{n}__x_nchw"], perm=[0, 3, 1, 2], name=f"{n}__tx"),
            helper.make_node("Transpose", [w], [f"{n}__w_iohw"], perm=[3, 0, 1, 2], name=f"{n}__tw"),
            helper.make_node("ConvTranspose", [f"{n}__x_nchw", f"{n}__w_iohw", b], [f"{n}__y_nchw"],
                             strides=[2, 2], kernel_shape=[2, 2], name=f"{n}__convt"),
            helper.make_node("Transpose", [f"{n}__y_nchw"], [y], perm=[0, 2, 3, 1], name=f"{n}__ty"),
        ]
        pos = list(graph.node).index(node)
        graph.node.remove(node)
        for k, nn in enumerate(new_nodes):
            graph.node.insert(pos + k, nn)
        changed = True
    if changed:
        # drop the custom-domain opset import tf2onnx added for the unknown op
        keep = [o for o in model.opset_import if o.domain in ("", "ai.onnx")]
        del model.opset_import[:]
        model.opset_import.extend(keep)
        onnx.checker.check_model(model)
        onnx.save(model, str(onnx_path))
        print(f"lowered Convolution2DTransposeBias in {onnx_path.name}")


def download(url: str, dst: pathlib.Path) -> None:
    if dst.exists():
        return
    print(f"download {url}")
    with urllib.request.urlopen(url) as r, dst.open("wb") as f:
        shutil.copyfileobj(r, f)


def collect_tflite(path: pathlib.Path, prefix: str, work: pathlib.Path, found: dict[str, pathlib.Path]) -> None:
    """Recursively unpacks .task bundles and records every .tflite graph under its logical path."""
    if path.suffix == ".tflite":
        found[prefix] = path
        return
    if not zipfile.is_zipfile(path):
        return  # e.g. geometry_pipeline_metadata.binarypb â€” not needed at inference time
    target = work / (prefix.replace("/", "__") + ".d")
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as z:
        z.extractall(target)
    for child in sorted(target.iterdir()):
        collect_tflite(child, f"{prefix}/{child.name}", work, found)


def describe(onnx_path: pathlib.Path) -> dict:
    import onnxruntime as ort

    s = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    io = lambda xs: [{"name": x.name, "shape": [d if isinstance(d, int) else -1 for d in x.shape]} for x in xs]
    return {"inputs": io(s.get_inputs()), "outputs": io(s.get_outputs())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/models")
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--only", nargs="*", help="convert only these model ids (default: all)")
    args = ap.parse_args()

    out = pathlib.Path(args.out).resolve()
    work = out / "_work"
    work.mkdir(parents=True, exist_ok=True)

    import tensorflow as tf
    import tf2onnx

    # tf2onnx asks the TFLite interpreter for tensor shapes. For some MediaPipe graphs
    # (pose_detector) Interpreter.get_tensor_details() crashes the process natively, so for those
    # the interpreter is refused and tf2onnx falls back to the shapes stored in the flatbuffer.
    _interpreter = tf.lite.Interpreter
    state = {"use_interpreter": True}

    def _no_delegate_interpreter(*a, **kw):
        if not state["use_interpreter"]:
            raise RuntimeError("interpreter shape inference disabled for this model")
        kw.setdefault("experimental_op_resolver_type", tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
        return _interpreter(*a, **kw)

    tf.lite.Interpreter = _no_delegate_interpreter

    found: dict[str, pathlib.Path] = {}
    for name, url in SOURCES:
        src = work / name
        download(url, src)
        collect_tflite(src, name, work, found)

    manifest = []
    for logical, tfl in found.items():
        model_id = RENAMES.get(logical, "__unknown__")
        if model_id is None:
            continue
        if model_id == "__unknown__":
            print(f"skip (not mapped): {logical}")
            continue
        dst = out / f"{model_id}.onnx"
        if args.only and model_id not in args.only:
            continue
        if not dst.exists():
            print(f"convert {logical} -> {dst.name}", flush=True)
            state["use_interpreter"] = model_id not in NO_INTERPRETER
            tfl = densify(tfl)
            tf2onnx.convert.from_tflite(str(tfl), opset=OPSET_OVERRIDES.get(model_id, args.opset), output_path=str(dst))
            lower_transpose_conv_bias(dst)
        entry = {"id": model_id, "file": dst.name, "source": logical, "bytes": dst.stat().st_size, "sha256": sha256(dst)}
        entry.update(describe(dst))
        manifest.append(entry)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
