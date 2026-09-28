"""
Evaluates static (calibrated) INT8 quantization for the models that have no weight-only INT8 variant.
Activations are calibrated on the test images (percentile 99.99); every output is compared with float32 on
those images and latency is measured on the CPU provider.

    C:\mpv\Scripts\python tools/model-conversion/evaluate_static_int8.py hand_landmarks_detector efficientnet_lite0

Result (MediaPipe.NET 1.0): no model reaches the 0.99 cosine bar the shipped variants must meet (hand presence
0.56, EfficientNet top-1 probabilities 0.92, MagicTouch mask 0.98, pose landmarks < 0.1), so no static INT8
variants are shipped. MediaPipe's float models were not trained for quantization; quantization-aware training
would be needed. Intermediate files go to %TEMP%/mpn-static-int8.
"""
import pathlib, time, sys
import numpy as np
import onnx, onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType, CalibrationMethod
from onnxruntime.quantization.shape_inference import quant_pre_process
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[2]
M = ROOT / "models" / "onnx"
T = pathlib.Path(__import__("tempfile").gettempdir()) / "mpn-static-int8"
T.mkdir(exist_ok=True)
IMAGES = sorted((ROOT / "tests" / "assets" / "images").glob("*.jpg"))


def feeds_for(sess, lo=0.0, hi=1.0):
    out = []
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
    h, w, c = shape[1], shape[2], shape[3]
    for p in IMAGES:
        im = Image.open(p).convert("RGB")
        for box in [None, (0.2, 0.1, 0.8, 0.7)]:
            j = im if box is None else im.crop((int(box[0] * im.width), int(box[1] * im.height), int(box[2] * im.width), int(box[3] * im.height)))
            a = np.asarray(j.resize((w, h)), np.float32) / 255 * (hi - lo) + lo
            if c == 4:
                pt = np.zeros((h, w, 1), np.float32); pt[h // 2 - 4:h // 2 + 4, w // 2 - 4:w // 2 + 4] = 1
                a = np.concatenate([a, pt], -1)
            out.append({inp.name: a[None]})
    return out


class Reader(CalibrationDataReader):
    def __init__(self, feeds): self.it = iter(feeds)
    def get_next(self): return next(self.it, None)


def cos(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def bench(path, feed, n=30):
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    for _ in range(3): s.run(None, feed)
    t = time.perf_counter()
    for _ in range(n): s.run(None, feed)
    return (time.perf_counter() - t) / n * 1000


RANGES = {"efficientnet_lite0": (-1, 1)}
for mid in sys.argv[1:]:
    src = M / f"{mid}.onnx"
    pre = T / f"{mid}.pre.onnx"
    quant_pre_process(str(src), str(pre), skip_symbolic_shape=True)
    ref = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    feeds = feeds_for(ref, *RANGES.get(mid, (0, 1)))
    for fmt in [QuantFormat.QDQ]:
        dst = T / f"{mid}.{fmt.name}.onnx"
        try:
            quantize_static(str(pre), str(dst), Reader(feeds), quant_format=fmt, per_channel=True,
                            activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                            calibrate_method=CalibrationMethod.Percentile, extra_options={"ActivationSymmetric": False, "CalibPercentile": 99.99})
            cand = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
            worst = 1.0
            for f in feeds:
                for r, c in zip(ref.run(None, f), cand.run(None, f)):
                    worst = min(worst, cos(r, c)) if r.size > 8 else worst
            per = [round(np.mean([cos(ref.run(None, f)[k], cand.run(None, f)[k]) for f in feeds]), 4) for k in range(len(ref.get_outputs()))]
            print("  per-output mean cos", [(o.name[:20], o.shape, p) for o, p in zip(ref.get_outputs(), per)])
            print(mid, fmt.name, f"min cos {worst:.5f} size {dst.stat().st_size / src.stat().st_size:.0%} "
                  f"fp32 {bench(src, feeds[0]):.2f} ms int8 {bench(dst, feeds[0]):.2f} ms", flush=True)
        except Exception as e:
            print(mid, fmt.name, "failed", type(e).__name__, str(e)[:200], flush=True)
