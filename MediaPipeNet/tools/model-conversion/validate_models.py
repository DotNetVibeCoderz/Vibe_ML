"""
Cross-validates every converted ONNX model against its original TFLite graph: the same random input
is fed to the TFLite interpreter and to ONNX Runtime and the maximum absolute difference per output
is reported. Also extracts the label files embedded in TFLite metadata (efficientdet, efficientnet,
gesture classifier) next to the ONNX models.

    C:\\mpv\\Scripts\\python tools/model-conversion/validate_models.py --models artifacts/models
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import zipfile

import numpy as np


def extract_labels(tflite: pathlib.Path, out: pathlib.Path, model_id: str) -> None:
    # TFLite metadata stores associated files as a zip appended to the flatbuffer.
    if not zipfile.is_zipfile(tflite):
        return
    with zipfile.ZipFile(tflite) as z:
        for name in z.namelist():
            if name.endswith(".txt"):
                (out / f"{model_id}.labels.txt").write_bytes(z.read(name))
                print(f"labels {model_id} <- {name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="artifacts/models")
    args = ap.parse_args()
    out = pathlib.Path(args.models).resolve()
    manifest = json.loads((out / "manifest.json").read_text())

    import onnxruntime as ort
    from tensorflow.lite.python.interpreter import Interpreter, OpResolverType

    rng = np.random.default_rng(42)
    report = []
    for m in manifest:
        # nested bundles are extracted to "<a>.task__<b>.task.d"
        parts = m["source"].split("/")
        if len(parts) == 3:
            src = out / "_work" / f"{parts[0]}__{parts[1]}.d" / parts[2]
        elif len(parts) == 2:
            src = out / "_work" / f"{parts[0]}.d" / parts[1]
        else:
            src = out / "_work" / parts[0]
        # Always compare against the ORIGINAL graph (never an intermediate such as *_dense.tflite).
        extract_labels(src, out, m["id"])

        sess = ort.InferenceSession(str(out / m["file"]), providers=["CPUExecutionProvider"])
        try:
            interp = Interpreter(str(src), experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
            interp.allocate_tensors()
        except Exception as e:  # custom ops (selfie segmenter) cannot run in stock TFLite
            print(f"{m['id']}: tflite reference unavailable ({str(e).splitlines()[0]})")
            report.append({"id": m["id"], "maxAbsDiff": None})
            continue

        feeds = {}
        t_inputs = interp.get_input_details()
        by_name = {i.name: i for i in sess.get_inputs()}
        for pos, ti in enumerate(t_inputs):
            oi = by_name.get(ti["name"]) or next((v for k, v in by_name.items() if k.startswith(ti["name"])), sess.get_inputs()[pos])
            shape = [max(int(d), 1) for d in ti["shape"]]
            x = rng.random(shape, dtype=np.float32)
            interp.set_tensor(ti["index"], x)
            feeds[oi.name] = x
        interp.invoke()
        ref = {d["name"]: interp.get_tensor(d["index"]) for d in interp.get_output_details()}
        got = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, feeds)))
        diffs = {}
        for name, r in ref.items():
            g = got.get(name)
            if g is None:  # tf2onnx may suffix output names
                g = next((v for k, v in got.items() if k.startswith(name)), None)
            diffs[name] = None if g is None or g.size != r.size else float(np.max(np.abs(g.reshape(r.shape) - r)))
        worst = max((d for d in diffs.values() if d is not None), default=None)
        print(f"{m['id']}: max|diff| = {worst}  {diffs}")
        report.append({"id": m["id"], "maxAbsDiff": worst, "outputs": diffs})

    (out / "validation.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
