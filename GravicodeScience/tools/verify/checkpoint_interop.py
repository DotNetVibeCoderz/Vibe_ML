#!/usr/bin/env python3
"""Verifies TransformerCheckpoint against an independent NumPy implementation of the same encoder.

Loading weights is easy to get *nearly* right: a transposed projection, an off-by-one in the layer
index, or a layer norm applied before the residual instead of after all produce a model that runs
and returns plausible-looking vectors. None of that is visible from the output.

So this script does not check that the load "worked". It builds a small BERT-shaped checkpoint with
known random weights, writes it as ONNX under HuggingFace's naming, and then computes the encoder's
output twice — once in NumPy here, once through GraviText — and compares them elementwise. Only an
exactly correct load agrees.

    pip install numpy onnx
    python tools/verify/checkpoint_interop.py
"""

import os
import subprocess
import sys
import tempfile

try:
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper, TensorProto
except ImportError:
    sys.exit("numpy and onnx are required: pip install numpy onnx")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HARNESS = os.path.join(REPO, "tools", "verify", "CheckpointInterop")

VOCAB, HIDDEN, LAYERS, HEADS, INTERMEDIATE, POSITIONS = 40, 16, 2, 4, 32, 24

failures = []


def check(name, condition, detail=""):
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{('  — ' + detail) if detail and not condition else ''}")
    if not condition:
        failures.append(name)


def build_weights(seed=0):
    """Small random weights, in PyTorch's (out, in) orientation for the linear layers."""
    rng = np.random.default_rng(seed)
    w = {}

    w["bert.embeddings.word_embeddings.weight"] = rng.normal(0, 0.1, (VOCAB, HIDDEN))
    w["bert.embeddings.position_embeddings.weight"] = rng.normal(0, 0.1, (POSITIONS, HIDDEN))
    w["bert.embeddings.LayerNorm.weight"] = rng.uniform(0.8, 1.2, HIDDEN)
    w["bert.embeddings.LayerNorm.bias"] = rng.normal(0, 0.1, HIDDEN)

    for i in range(LAYERS):
        p = f"bert.encoder.layer.{i}"
        for part in ["attention.self.query", "attention.self.key",
                     "attention.self.value", "attention.output.dense"]:
            # PyTorch nn.Linear: (out_features, in_features).
            w[f"{p}.{part}.weight"] = rng.normal(0, 0.2, (HIDDEN, HIDDEN))
            w[f"{p}.{part}.bias"] = rng.normal(0, 0.1, HIDDEN)

        w[f"{p}.attention.output.LayerNorm.weight"] = rng.uniform(0.8, 1.2, HIDDEN)
        w[f"{p}.attention.output.LayerNorm.bias"] = rng.normal(0, 0.1, HIDDEN)

        w[f"{p}.intermediate.dense.weight"] = rng.normal(0, 0.2, (INTERMEDIATE, HIDDEN))
        w[f"{p}.intermediate.dense.bias"] = rng.normal(0, 0.1, INTERMEDIATE)
        w[f"{p}.output.dense.weight"] = rng.normal(0, 0.2, (HIDDEN, INTERMEDIATE))
        w[f"{p}.output.dense.bias"] = rng.normal(0, 0.1, HIDDEN)

        w[f"{p}.output.LayerNorm.weight"] = rng.uniform(0.8, 1.2, HIDDEN)
        w[f"{p}.output.LayerNorm.bias"] = rng.normal(0, 0.1, HIDDEN)

    return w


def write_onnx(weights, path):
    """Wraps the tensors as ONNX initializers. The graph itself is a stub — only the weights matter."""
    initializers = [numpy_helper.from_array(v.astype(np.float32), name=k) for k, v in weights.items()]

    graph = helper.make_graph(
        nodes=[helper.make_node("Identity", ["input"], ["output"])],
        name="checkpoint",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
        initializer=initializers,
    )
    model = helper.make_model(graph, producer_name="checkpoint_interop")
    model.opset_import[0].version = 13
    onnx.save(model, path)


def layer_norm(x, gamma, beta, eps=1e-12):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def gelu(x):
    # The tanh approximation, which is what BERT uses and what Activations.Gelu implements.
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def reference_forward(weights, token_ids):
    """The same encoder, written independently against the weights."""
    n = len(token_ids)
    head_size = HIDDEN // HEADS

    x = weights["bert.embeddings.word_embeddings.weight"][token_ids] \
        + weights["bert.embeddings.position_embeddings.weight"][:n]
    x = layer_norm(x, weights["bert.embeddings.LayerNorm.weight"],
                   weights["bert.embeddings.LayerNorm.bias"])

    for i in range(LAYERS):
        p = f"bert.encoder.layer.{i}"

        # x @ W.T + b, because the stored weight is (out, in).
        q = x @ weights[f"{p}.attention.self.query.weight"].T + weights[f"{p}.attention.self.query.bias"]
        k = x @ weights[f"{p}.attention.self.key.weight"].T + weights[f"{p}.attention.self.key.bias"]
        v = x @ weights[f"{p}.attention.self.value.weight"].T + weights[f"{p}.attention.self.value.bias"]

        context = np.zeros_like(x)
        for h in range(HEADS):
            lo, hi = h * head_size, (h + 1) * head_size
            scores = q[:, lo:hi] @ k[:, lo:hi].T / np.sqrt(head_size)
            scores = scores - scores.max(axis=-1, keepdims=True)
            weightsm = np.exp(scores)
            weightsm /= weightsm.sum(axis=-1, keepdims=True)
            context[:, lo:hi] = weightsm @ v[:, lo:hi]

        attended = context @ weights[f"{p}.attention.output.dense.weight"].T \
            + weights[f"{p}.attention.output.dense.bias"]

        x = layer_norm(x + attended,
                       weights[f"{p}.attention.output.LayerNorm.weight"],
                       weights[f"{p}.attention.output.LayerNorm.bias"])

        hidden = gelu(x @ weights[f"{p}.intermediate.dense.weight"].T
                      + weights[f"{p}.intermediate.dense.bias"])
        projected = hidden @ weights[f"{p}.output.dense.weight"].T + weights[f"{p}.output.dense.bias"]

        x = layer_norm(x + projected,
                       weights[f"{p}.output.LayerNorm.weight"],
                       weights[f"{p}.output.LayerNorm.bias"])

    return x


def run_csharp(*args):
    result = subprocess.run(
        ["dotnet", "run", "--project", HARNESS, "-c", "Release", "--", *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        sys.exit(f"the C# harness failed on: {' '.join(args)}")
    return result.stdout


def main():
    work = tempfile.mkdtemp(prefix="checkpoint-interop-")
    path = os.path.join(work, "tiny.onnx")

    weights = build_weights()
    write_onnx(weights, path)
    print(f"\nwrote a {LAYERS}-layer checkpoint with {len(weights)} tensors")

    token_ids = [5, 12, 3, 27, 9, 1]

    # ---------------------------------------------------------------- the real check
    print("\n=== 1. Forward pass against an independent NumPy encoder ===")

    expected = reference_forward(weights, token_ids)
    output = run_csharp("forward", path, ",".join(map(str, token_ids)))

    rows = [line for line in output.strip().splitlines() if line.startswith("row ")]
    actual = np.array([[float(v) for v in row.split(":", 1)[1].split(",")] for row in rows])

    check("shapes agree", actual.shape == expected.shape, f"{actual.shape} vs {expected.shape}")

    if actual.shape == expected.shape:
        largest = np.abs(actual - expected).max()
        # float32 initializers widened to float64, so agreement is at float32 precision.
        check("every element agrees to 1e-5", largest < 1e-5, f"largest difference {largest:.3e}")
        print(f"        largest elementwise difference: {largest:.3e}")

    # ---------------------------------------------------------------- reporting
    print("\n=== 2. The load reports what it did ===")

    report = run_csharp("report", path)
    check("every parameter was found", "missing 0" in report, report.strip())
    check("nothing in the file went unused", "unused 0" in report, report.strip())
    check("the model is marked pretrained", "pretrained True" in report, report.strip())

    # ---------------------------------------------------------------- failure modes
    print("\n=== 3. Wrong settings are caught, not absorbed ===")

    wrong = run_csharp("wrong-transpose", path)
    check("a wrong transpose is refused", "REFUSED" in wrong, wrong.strip()[:200])

    partial = os.path.join(work, "partial.onnx")
    trimmed = {k: v for k, v in weights.items() if "layer.1" not in k}
    write_onnx(trimmed, partial)

    strict = run_csharp("strict", partial)
    check("a partial checkpoint is refused in strict mode", "REFUSED" in strict, strict.strip()[:200])

    lenient = run_csharp("lenient", partial)
    check("lenient mode loads and says what is missing", "missing 16" in lenient, lenient.strip())
    check("a partial load is NOT marked pretrained", "pretrained False" in lenient, lenient.strip())

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED: {', '.join(failures)}")
        sys.exit(1)

    print("All checkpoint interop checks passed.")


if __name__ == "__main__":
    main()
