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
    # 0.2.0
    ("blaze_face_full_range.tflite", f"{BASE}/face_detector/blaze_face_full_range/float16/latest/blaze_face_full_range.tflite"),
    ("hand_recrop.tflite", "https://storage.googleapis.com/mediapipe-assets/hand_recrop.tflite"),
    # 0.3.0
    ("holistic_landmarker.task", f"{BASE}/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"),
    ("mobilenet_v3_small.tflite", f"{BASE}/image_embedder/mobilenet_v3_small/float32/latest/mobilenet_v3_small.tflite"),
    ("selfie_multiclass_256x256.tflite", f"{BASE}/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"),
    ("hair_segmenter.tflite", f"{BASE}/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite"),
    ("deeplab_v3.tflite", f"{BASE}/image_segmenter/deeplab_v3/float32/latest/deeplab_v3.tflite"),
    ("magic_touch.tflite", f"{BASE}/interactive_segmenter/magic_touch/float32/latest/magic_touch.tflite"),
    ("face_stylizer_color_sketch.task", f"{BASE}/face_stylizer/blaze_face_stylizer/float32/latest/face_stylizer_color_sketch.task"),
    ("yamnet.tflite", f"{BASE}/audio_classifier/yamnet/float32/latest/yamnet.tflite"),
    ("bert_classifier.tflite", f"{BASE}/text_classifier/bert_classifier/float32/latest/bert_classifier.tflite"),
    ("average_word_classifier.tflite", f"{BASE}/text_classifier/average_word_classifier/float32/latest/average_word_classifier.tflite"),
    ("bert_embedder.tflite", f"{BASE}/text_embedder/bert_embedder/float32/latest/bert_embedder.tflite"),
    ("language_detector.tflite", f"{BASE}/language_detector/language_detector/float32/latest/language_detector.tflite"),
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
    "gesture_recognizer.task/hand_landmarker.task/hand_detector.tflite": None,        # same as hand_landmarker.task
    "gesture_recognizer.task/hand_landmarker.task/hand_landmarks_detector.tflite": None,
    # 0.2.0
    "blaze_face_full_range.tflite": "face_detection_full_range",
    "hand_recrop.tflite": None,  # superseded by holistic's hand_roi_refinement
    "holistic_landmarker.task/hand_roi_refinement.tflite": "hand_roi_refinement",
    "holistic_landmarker.task/face_blendshapes.tflite": None,         # identical to face_landmarker.task
    "holistic_landmarker.task/face_detector.tflite": None,            # BlazeFace short range (metadata differs)
    "holistic_landmarker.task/face_landmarks_detector.tflite": None,  # 468-point mesh; the 478-point model is used
    "holistic_landmarker.task/hand_landmarks_detector.tflite": None,  # same weights as hand_landmarker.task
    "holistic_landmarker.task/pose_detector.tflite": None,            # identical to pose_detection.tflite
    "holistic_landmarker.task/pose_landmarks_detector.tflite": None,  # BlazePose lite
    # 0.3.0
    "mobilenet_v3_small.tflite": "mobilenet_v3_small_embedder",
    "selfie_multiclass_256x256.tflite": "selfie_multiclass",
    "hair_segmenter.tflite": "hair_segmenter",
    "deeplab_v3.tflite": "deeplab_v3",
    "magic_touch.tflite": "magic_touch",
    "face_stylizer_color_sketch.task/face_detector.tflite": None,
    "face_stylizer_color_sketch.task/face_landmarks_detector.tflite": None,
    "face_stylizer_color_sketch.task/face_stylizer.tflite": "face_stylizer_color_sketch",
    "yamnet.tflite": "yamnet",
    "bert_classifier.tflite": "bert_classifier",
    "average_word_classifier.tflite": "average_word_classifier",
    "bert_embedder.tflite": "bert_embedder",
    "language_detector.tflite": "language_detector",
}


NO_INTERPRETER: set[str] = set()

# HardSwish needs opset 14.
OPSET_OVERRIDES = {"selfie_segmenter": 14, "mobilenet_v3_small_embedder": 14, "hair_segmenter": 14}


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


def tflite_custom_ops(tflite_path: pathlib.Path) -> dict[str, tuple[str, bytes, list[list[int]]]]:
    """Maps the first output tensor name of every custom op to (op name, custom options, input shapes)."""
    from tensorflow.lite.tools import flatbuffer_utils

    model = flatbuffer_utils.read_model(str(tflite_path))
    g = model.subgraphs[0]
    result = {}
    for op in g.operators:
        code = model.operatorCodes[op.opcodeIndex]
        if not code.customCode:
            continue
        name = g.tensors[op.outputs[0]].name.decode()
        shapes = [g.tensors[i].shape.tolist() if i >= 0 else [] for i in op.inputs]
        opts = bytes(op.customOptions) if op.customOptions is not None else b""
        result[name] = (code.customCode.decode(), opts, shapes)
    return result


def lower_custom_ops(onnx_path: pathlib.Path, tflite_path: pathlib.Path) -> None:
    """
    Rewrites MediaPipe's custom TFLite ops, which tf2onnx leaves as unknown nodes, as standard ONNX ops.
    Options come from the TFLite flatbuffer (TfLiteTransposeConvParams / TfLitePoolParams: int32 padding,
    stride_w, stride_h[, filter_w, filter_h]). TFLite tensors are NHWC; ONNX ops run in NCHW between Transposes.

    * Convolution2DTransposeBias -> ConvTranspose (+ bias). Kernel [O,H,W,I] -> [I,O,H,W].
    * MaxPoolingWithArgmax2D     -> MaxPool with Indices output.
    * MaxUnpooling2D             -> MaxUnpool (indices come from the matching MaxPool; both use ONNX's
                                     NCHW index convention, so the pair stays consistent).
    """
    import struct

    import onnx
    from onnx import helper

    custom = tflite_custom_ops(tflite_path)
    model = onnx.load(str(onnx_path))
    graph = model.graph
    pool_input_nchw: dict[str, str] = {}  # indices tensor -> NCHW input of the pool that produced it
    changed = []
    for idx, node in enumerate(list(graph.node)):
        op = node.op_type.removeprefix("TFL_")
        if op not in ("Convolution2DTransposeBias", "MaxPoolingWithArgmax2D", "MaxUnpooling2D"):
            continue
        tfl = custom.get(node.output[0])
        opts = struct.unpack("<5i", tfl[1][:20].ljust(20, b"\0")) if tfl else (1, 2, 2, 2, 2)
        _, stride_w, stride_h, filter_w, filter_h = opts
        n = (node.name or f"{op}_{idx}").replace(":", "_")
        if op == "Convolution2DTransposeBias":
            x, w, b = node.input[:3]
            k_h, k_w = (tfl[2][1][1], tfl[2][1][2]) if tfl else (2, 2)
            new = [
                helper.make_node("Transpose", [x], [f"{n}__x"], perm=[0, 3, 1, 2]),
                helper.make_node("Transpose", [w], [f"{n}__w"], perm=[3, 0, 1, 2]),
                helper.make_node("ConvTranspose", [f"{n}__x", f"{n}__w", b], [f"{n}__y"],
                                 strides=[stride_h, stride_w], kernel_shape=[k_h, k_w], auto_pad="SAME_UPPER"),
                helper.make_node("Transpose", [f"{n}__y"], [node.output[0]], perm=[0, 2, 3, 1]),
            ]
        elif op == "MaxPoolingWithArgmax2D":
            x = node.input[0]
            pool_input_nchw[node.output[1]] = f"{n}__x"
            new = [
                helper.make_node("Transpose", [x], [f"{n}__x"], perm=[0, 3, 1, 2]),
                helper.make_node("MaxPool", [f"{n}__x"], [f"{n}__y", node.output[1]],
                                 kernel_shape=[filter_h, filter_w], strides=[stride_h, stride_w]),
                helper.make_node("Transpose", [f"{n}__y"], [node.output[0]], perm=[0, 2, 3, 1]),
            ]
        else:  # MaxUnpooling2D
            x, indices = node.input[:2]
            new = [
                helper.make_node("Transpose", [x], [f"{n}__x"], perm=[0, 3, 1, 2]),
                helper.make_node("Shape", [pool_input_nchw[indices]], [f"{n}__shape"]),
                helper.make_node("MaxUnpool", [f"{n}__x", indices, f"{n}__shape"], [f"{n}__y"],
                                 kernel_shape=[filter_h, filter_w], strides=[stride_h, stride_w]),
                helper.make_node("Transpose", [f"{n}__y"], [node.output[0]], perm=[0, 2, 3, 1]),
            ]
        pos = list(graph.node).index(node)
        graph.node.remove(node)
        for k, nn in enumerate(new):
            graph.node.insert(pos + k, nn)
        changed.append(op)
    if changed:
        keep = [o for o in model.opset_import if o.domain in ("", "ai.onnx")]
        del model.opset_import[:]
        model.opset_import.extend(keep)
        # Stale type annotations of rewritten tensors (e.g. int32 argmax indices) would clash with ONNX's int64.
        rewritten = {o for nd in graph.node for o in nd.output}
        kept_info = [vi for vi in graph.value_info if vi.name not in rewritten or vi.type.tensor_type.elem_type == 1]
        del graph.value_info[:]
        graph.value_info.extend(kept_info)
        onnx.checker.check_model(model)
        onnx.save(model, str(onnx_path))
        print(f"lowered custom ops in {onnx_path.name}: {sorted(set(changed))}")


def strip_dynamic_quantization(onnx_path: pathlib.Path) -> None:
    """
    TFLite hybrid FULLY_CONNECTED ops with `asymmetric_quantize_inputs` (MobileBERT) are converted by tf2onnx
    into DynamicQuantizeLinear -> DequantizeLinear pairs on the input, the weights AND the bias (uint8,
    per-tensor), which is far coarser than TFLite's per-channel int8 weights and costs up to 7% cosine
    similarity on the output. Each pair is removed so the layer runs in float; the int8 weight storage
    (constant DequantizeLinear) is kept.
    """
    import onnx

    model = onnx.load(str(onnx_path))
    graph = model.graph
    producer = {o: n for n in graph.node for o in n.output}
    rename: dict[str, str] = {}
    dead = set()
    for node in graph.node:
        if node.op_type != "DequantizeLinear":
            continue
        q = producer.get(node.input[0])
        if q is None or q.op_type != "DynamicQuantizeLinear" or list(node.input[1:3]) != list(q.output[1:3]):
            continue
        rename[node.output[0]] = q.input[0]
        dead.add(id(node))
    if not rename:
        return
    for node in graph.node:
        for i, name in enumerate(node.input):
            while name in rename:
                name = rename[name]
            node.input[i] = name
    for out in graph.output:
        if out.name in rename:
            raise RuntimeError(f"graph output {out.name} is produced by a stripped DequantizeLinear")
    used = {i for n in graph.node if id(n) not in dead for i in n.input} | {o.name for o in graph.output}
    keep = [n for n in graph.node if id(n) not in dead and not (n.op_type == "DynamicQuantizeLinear" and not any(o in used for o in n.output))]
    removed = len(graph.node) - len(keep)
    del graph.node[:]
    graph.node.extend(keep)
    onnx.checker.check_model(model)
    onnx.save(model, str(onnx_path))
    print(f"stripped {len(rename)} dynamic quantization pairs ({removed} nodes) from {onnx_path.name}")


def build_language_detector(tflite_path: pathlib.Path, onnx_path: pathlib.Path) -> None:
    """
    The language detector starts with the string op NGramHash (character n-gram hashing), which has no ONNX
    equivalent, so the model is rebuilt by hand. NGramHash runs in C# (MediaPipeNet.Tasks.Text.NGramHasher)
    and produces the ONNX input `ngram_ids` [num_ngram_lengths, num_tokens] int32 (every id >= 1).

    KmeansEmbeddingLookup (product-quantized embeddings: uint8 codes [V, 6] into a [256, 4] codebook, averaged
    over tokens) is expanded into a dense float table [V, 24] and lowered to Gather + ReduceMean.
    The fully connected layers are hybrid int8 in TFLite; they are dequantized to float32 here.
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from tensorflow.lite.python import schema_py_generated as schema
    from tensorflow.lite.tools import flatbuffer_utils

    model = flatbuffer_utils.read_model(str(tflite_path))
    g = model.subgraphs[0]
    np_types = {0: np.float32, 2: np.int32, 3: np.uint8, 9: np.int8}

    def tensor(i: int) -> np.ndarray:
        t = g.tensors[i]
        a = np.frombuffer(bytes(model.buffers[t.buffer].data), dtype=np_types[t.type]).reshape(t.shape)
        if t.type == 9:  # int8 weights: per-tensor or per-channel (axis 0) symmetric quantization
            scale = np.asarray(t.quantization.scale, dtype=np.float32)
            a = a.astype(np.float32) * (scale.reshape(-1, 1) if scale.size > 1 else scale[0])
        return a

    def op_name(op) -> str:
        c = model.operatorCodes[op.opcodeIndex]
        if c.customCode:
            return c.customCode.decode()
        code = max(c.builtinCode, c.deprecatedBuiltinCode)
        return next(k for k, v in schema.BuiltinOperator.__dict__.items() if v == code)

    inits, nodes, heads, fcs = [], [], [], []
    for op in g.operators:
        name = op_name(op)
        if name == "KmeansEmbeddingLookup":
            codes, codebook = tensor(op.inputs[1]), tensor(op.inputs[2])
            heads.append(codebook[codes].reshape(codes.shape[0], -1).astype(np.float32))  # [V, 6*4]
        elif name == "FULLY_CONNECTED":
            relu = op.builtinOptions.fusedActivationFunction == schema.ActivationFunctionType.RELU
            fcs.append((tensor(op.inputs[1]), tensor(op.inputs[2]), relu))

    nodes.append(helper.make_node("Split", ["ngram_ids"], [f"ids_{k}" for k in range(len(heads))], axis=0))
    for k, table in enumerate(heads):
        inits.append(numpy_helper.from_array(table, f"embedding_{k}"))
        nodes.append(helper.make_node("Gather", [f"embedding_{k}", f"ids_{k}"], [f"emb_{k}"], axis=0))       # [1, T, 24]
        nodes.append(helper.make_node("ReduceMean", [f"emb_{k}"], [f"head_{k}"], axes=[1], keepdims=0))     # [1, 24]
    nodes.append(helper.make_node("Concat", [f"head_{k}" for k in range(len(heads))], ["x_0"], axis=1))
    for k, (w, b, relu) in enumerate(fcs):
        inits += [numpy_helper.from_array(w.T.copy(), f"fc_{k}_w"), numpy_helper.from_array(b, f"fc_{k}_b")]
        out = f"x_{k + 1}" if relu or k < len(fcs) - 1 else "logits"
        nodes.append(helper.make_node("Gemm", [f"x_{k}", f"fc_{k}_w", f"fc_{k}_b"], [f"fc_{k}" if relu else out]))
        if relu:
            nodes.append(helper.make_node("Relu", [f"fc_{k}"], [out]))
    nodes.append(helper.make_node("Softmax", ["logits"], ["probabilities"], axis=-1))

    graph = helper.make_graph(
        nodes, "language_detector",
        [helper.make_tensor_value_info("ngram_ids", TensorProto.INT32, [len(heads), "tokens"])],
        [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [1, fcs[-1][1].shape[0]])],
        inits)
    onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], producer_name="MediaPipe.NET")
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, str(onnx_path))
    print(f"built {onnx_path.name}: {len(heads)} n-gram heads, {len(fcs)} dense layers")


MANUAL_BUILDERS = {"language_detector": build_language_detector}


def convert_custom(src: pathlib.Path, model_id: str | None, out: pathlib.Path, opset: int) -> int:
    """
    Converts one user-supplied TFLite model (a MediaPipe Model Maker export, for instance) with the same
    pipeline as the catalog models, and writes the files MediaPipe.NET's custom-model support reads:
    {id}.onnx, {id}.labels.txt (from the TFLite metadata, when present) and {id}.vocab.txt (text models).
    Use them with e.g. ImageClassifierOptions.ModelPath / ObjectDetectorOptions.ModelPath.
    """
    import tensorflow as tf
    import tf2onnx

    _interpreter = tf.lite.Interpreter
    tf.lite.Interpreter = lambda *a, **kw: _interpreter(
        *a, **{**kw, "experimental_op_resolver_type": tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES})
    model_id = model_id or src.stem
    out.mkdir(parents=True, exist_ok=True)
    if src.suffix == ".task":
        # A bundle (e.g. a Model Maker gesture recognizer): convert every graph inside it as {id}_{graph}.
        found: dict[str, pathlib.Path] = {}
        collect_tflite(src, src.name, out / "_work", found)
        status = 0
        for logical, inner in found.items():
            status |= convert_custom(inner, f"{model_id}_{inner.stem}", out, opset)
        return status
    dst = out / f"{model_id}.onnx"
    work = out / "_work"
    work.mkdir(exist_ok=True)
    tfl = densify(src)
    last_error = None
    for candidate in dict.fromkeys([opset, 14, 15]):  # HardSwish and friends need a newer opset
        try:
            tf2onnx.convert.from_tflite(str(tfl), opset=candidate, output_path=str(dst))
            lower_custom_ops(dst, tfl)
            strip_dynamic_quantization(dst)
            describe(dst)
            last_error = None
            break
        except Exception as e:  # retry with a newer opset
            last_error = e
            dst.unlink(missing_ok=True)
    if last_error is not None:
        print(f"conversion failed: {type(last_error).__name__}: {last_error}")
        return 1
    if zipfile.is_zipfile(src):
        with zipfile.ZipFile(src) as z:
            for name in z.namelist():
                kind = "vocab" if "vocab" in name else "labels" if name.endswith(".txt") and name != "labelmap.txt" else None
                if kind:
                    (out / f"{model_id}.{kind}.txt").write_bytes(z.read(name))
                    print(f"{kind}: {out / f'{model_id}.{kind}.txt'} <- {name}")
    info = describe(dst)
    print(json.dumps({"id": model_id, "file": str(dst), "bytes": dst.stat().st_size, "sha256": sha256(dst), **info}, indent=2))
    return 0


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
        return  # e.g. geometry_pipeline_metadata.binarypb - not needed at inference time
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
    ap.add_argument("--custom", help="convert your own .tflite (e.g. a MediaPipe Model Maker export) instead of the catalog")
    ap.add_argument("--custom-id", help="output name for --custom (default: the file name)")
    args = ap.parse_args()

    out = pathlib.Path(args.out).resolve()
    if args.custom:
        return convert_custom(pathlib.Path(args.custom), args.custom_id, out, args.opset)
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
    failed: dict[str, str] = {}
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
            try:
                if model_id in MANUAL_BUILDERS:
                    MANUAL_BUILDERS[model_id](tfl, dst)
                else:
                    tf2onnx.convert.from_tflite(str(tfl), opset=OPSET_OVERRIDES.get(model_id, args.opset), output_path=str(dst))
                    lower_custom_ops(dst, tfl)
                    strip_dynamic_quantization(dst)
            except Exception as e:  # e.g. Flex / stateful ops that have no ONNX equivalent
                dst.unlink(missing_ok=True)
                failed[model_id] = f"{type(e).__name__}: {e}"
                print(f"FAILED {model_id}: {failed[model_id]}", flush=True)
                continue
        entry = {"id": model_id, "file": dst.name, "source": logical, "bytes": dst.stat().st_size, "sha256": sha256(dst)}
        entry.update(describe(dst))
        manifest.append(entry)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "failed.json").write_text(json.dumps(failed, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
