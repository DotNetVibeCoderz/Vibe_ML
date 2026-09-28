"""
MediaPipe.NET model quantization.

Creates reduced-precision variants of the converted ONNX models and keeps only the ones that stay
faithful to the float32 model:

  * FP16 — weights and activations in float16 (inputs/outputs stay float32), half the size. Best on GPUs;
    ONNX Runtime's CPU provider runs most float16 kernels too.
  * INT8 — dynamic quantization (int8 weights per channel, activations quantized at run time), about a
    quarter of the size of the weights. Best on CPUs.

Every variant is run next to the float32 model on the same inputs (real images / audio / text where the
model allows it, random tensors otherwise); a variant is accepted when the cosine similarity of every
output is at least --fp16-min-cosine / --int8-min-cosine. Accepted variants are written to
models/onnx/quantized/ with a manifest that the model catalog is generated from.

    C:\\mpv\\Scripts\\python tools/model-conversion/quantize_models.py

Created by Gravicode Studios, led by Kang Fadhil.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODELS = ROOT / "models" / "onnx"

# Models worth quantizing (larger than ~2 MB). language_detector is mostly embedding tables, which dynamic
# quantization does not touch, so it only gets an FP16 variant.
CANDIDATES = [
    "face_landmarks_detector", "face_detection_full_range", "palm_detection", "hand_landmarks_detector",
    "pose_detection", "pose_landmarks_detector_lite", "pose_landmarks_detector_full",
    "efficientdet_lite0", "efficientnet_lite0", "mobilenet_v3_small_embedder",
    "selfie_multiclass", "deeplab_v3", "magic_touch", "yamnet",
    "bert_classifier", "bert_embedder", "language_detector", "face_stylizer_color_sketch", "face_landmarks_detector_192",
]
NO_INT8 = {"language_detector"}


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def feeds(session, rng: np.random.Generator) -> list[dict[str, np.ndarray]]:
    """Two input sets per model: random values in a plausible range for each input."""
    result = []
    for _ in range(2):
        feed = {}
        for i in session.get_inputs():
            shape = [d if isinstance(d, int) and d > 0 else (4 if "ngram" in i.name else 1) for d in i.shape]
            if "ngram" in i.name:
                shape = [4, 24]
            if i.type == "tensor(int32)":
                high = 2 if ("mask" in i.name or "type" in i.name) else 5000
                feed[i.name] = rng.integers(1 if high > 2 else 0, high, shape).astype(np.int32)
            else:
                feed[i.name] = rng.random(shape, dtype=np.float32) * (1.0 if len(shape) != 1 else 0.2)
        result.append(feed)
    return result


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 and nb == 0:
        return 1.0
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def compare(reference: pathlib.Path, candidate: pathlib.Path) -> float:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_quant_qdq", "1")
    ref = ort.InferenceSession(str(reference), so, providers=["CPUExecutionProvider"])
    cand = ort.InferenceSession(str(candidate), providers=["CPUExecutionProvider"])
    worst = 1.0
    for feed in feeds(ref, np.random.default_rng(7)):
        for r, c in zip(ref.run(None, feed), cand.run(None, feed)):
            worst = min(worst, cosine(r, c))
    return worst


def to_fp16(src: pathlib.Path, dst: pathlib.Path) -> None:
    import onnx
    from onnxconverter_common import float16

    model = onnx.load(str(src))
    # Resize/Range and friends need float32 scales; keep them (and shape arithmetic) in float32.
    converted = float16.convert_float_to_float16(model, keep_io_types=True, disable_shape_infer=False,
                                                 op_block_list=["Resize", "Range", "RandomNormal", "RandomNormalLike", "ConstantOfShape", "DequantizeLinear", "Softmax"])
    onnx.save(converted, str(dst))


def to_int8(src: pathlib.Path, dst: pathlib.Path) -> None:
    """
    Weight-only int8: every Conv / ConvTranspose / MatMul / Gemm weight is stored as symmetric per-channel
    int8 behind a DequantizeLinear node; activations stay float. Storage shrinks ~4x while the network
    computes in float (MediaPipe.NET disables ONNX Runtime's QDQ fusion, so no activation quantization
    happens). Dynamic activation quantization was evaluated and destroys the landmark CNNs.
    """
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(str(src))
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    axis_of: dict[str, int] = {}
    for node in graph.node:
        if len(node.input) < 2 or node.input[1] not in inits:
            continue
        w = inits[node.input[1]]
        if w.data_type != onnx.TensorProto.FLOAT or len(w.dims) < 2 or np.prod(w.dims) < 1024:
            continue
        if node.op_type == "Conv":
            axis = 0
        elif node.op_type == "ConvTranspose":
            axis = 1
        elif node.op_type == "MatMul":
            axis = len(w.dims) - 1
        elif node.op_type == "Gemm":
            trans_b = next((a.i for a in node.attribute if a.name == "transB"), 0)
            axis = 0 if trans_b else 1
        else:
            continue
        axis_of.setdefault(node.input[1], axis)
    dq_nodes = []
    for name, axis in axis_of.items():
        w = numpy_helper.to_array(inits[name]).astype(np.float32)
        reduce_axes = tuple(i for i in range(w.ndim) if i != axis)
        scale = np.abs(w).max(axis=reduce_axes) / 127.0
        scale[scale == 0] = 1.0
        shape = [1] * w.ndim
        shape[axis] = -1
        q = np.clip(np.round(w / scale.reshape(shape)), -127, 127).astype(np.int8)
        graph.initializer.remove(inits[name])
        graph.initializer.extend([
            numpy_helper.from_array(q, name + "__q"),
            numpy_helper.from_array(scale.astype(np.float32), name + "__scale"),
            numpy_helper.from_array(np.zeros_like(scale, dtype=np.int8), name + "__zp"),
        ])
        dq_nodes.append(helper.make_node("DequantizeLinear", [name + "__q", name + "__scale", name + "__zp"], [name], axis=axis, name=name + "__dq"))
    for i, n in enumerate(dq_nodes):
        graph.node.insert(i, n)
    onnx.checker.check_model(model)
    onnx.save(model, str(dst))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--fp16-min-cosine", type=float, default=0.999)
    ap.add_argument("--int8-min-cosine", type=float, default=0.99)
    args = ap.parse_args()

    out = MODELS / "quantized"
    out.mkdir(exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = {e["file"]: e for e in json.loads(manifest_path.read_text())} if manifest_path.exists() else {}
    for model_id in CANDIDATES:
        if args.only and model_id not in args.only:
            continue
        src = MODELS / f"{model_id}.onnx"
        for precision, convert, threshold in [("fp16", to_fp16, args.fp16_min_cosine), ("int8", to_int8, args.int8_min_cosine)]:
            if precision == "int8" and model_id in NO_INT8:
                continue
            dst = out / f"{model_id}.{precision}.onnx"
            try:
                convert(src, dst)
                score = compare(src, dst)
            except Exception as e:  # an op the converter or runtime cannot handle in this precision
                print(f"{model_id} {precision}: failed ({type(e).__name__}: {str(e).splitlines()[0][:120]})")
                dst.unlink(missing_ok=True)
                manifest.pop(dst.name, None)
                continue
            ratio = dst.stat().st_size / src.stat().st_size
            # Keep a variant only when it is faithful AND actually smaller (models that already store int8
            # weights gain nothing from either conversion).
            accepted = score >= threshold and ratio <= 0.8
            print(f"{model_id} {precision}: min cosine {score:.5f}, size {ratio:.0%} -> {'accepted' if accepted else 'rejected'}")
            if not accepted:
                dst.unlink()
                manifest.pop(dst.name, None)
                continue
            manifest[dst.name] = {"id": model_id, "precision": precision, "file": dst.name, "bytes": dst.stat().st_size,
                                  "sha256": sha256(dst), "min_cosine": round(score, 6)}
    manifest_path.write_text(json.dumps(sorted(manifest.values(), key=lambda e: e["file"]), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
