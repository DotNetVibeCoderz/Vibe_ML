"""
Tabulates the two halves of the comparison benchmark into Markdown.

Run both halves first, then:

    python report.py --python python/python.json --dotnet dotnet.json --out report.md

The ratio column is always "how many times longer HF.Net took". A number below 1 means HF.Net was
faster. Reporting it in one direction throughout avoids the usual trick of flipping the ratio
whenever it flatters the author.
"""

import argparse
import json
from pathlib import Path


def fmt(value, unit="", digits=2):
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:,.{digits}f}{unit}"
    return str(value)


def ratio(dotnet, python):
    """How many times longer .NET took. None when either side is missing."""
    if dotnet is None or python is None or python == 0:
        return None
    return dotnet / python


def ratio_text(value):
    if value is None:
        return "—"
    if value < 1:
        return f"**{1 / value:,.2f}x faster**"
    return f"{value:,.2f}x slower"


def row(label, key, py, net, unit=" ms", digits=2):
    p = py.get(key)
    n = net.get(key)
    return f"| {label} | {fmt(p, unit, digits)} | {fmt(n, unit, digits)} | {ratio_text(ratio(n, p))} |"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="python/python.json")
    parser.add_argument("--dotnet", default="dotnet.json")
    parser.add_argument("--out", default="report.md")
    args = parser.parse_args()

    py = json.loads(Path(args.python).read_text(encoding="utf-8"))
    net = json.loads(Path(args.dotnet).read_text(encoding="utf-8"))

    lines = []
    add = lines.append

    add("# HF.Net versus the Python reference implementation")
    add("")
    add("*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.*")
    add("")
    add("Both halves measure the same inputs on the same machine in the same session, and both")
    add("report the **best** of several timed runs after a warm-up. On a laptop that throttles, the")
    add("mean measures the thermal state of the room.")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| Machine | {py.get('platform', '?')} |")
    add(f"| CPU | {net.get('processor', '?')} |")
    add(f"| Python | {py.get('runtime', '?')}, transformers {py.get('transformers', '?')}, "
        f"tokenizers {py.get('tokenizers', '?')}, torch {py.get('torch', 'n/a')} |")
    add(f"| .NET | {net.get('runtime', '?')} ({net.get('configuration', '?')}) |")
    add(f"| torch threads | {py.get('torch_threads', 'n/a')} |")
    add("")

    # ------------------------------------------------------------------ tokenizer
    add("## Tokenization")
    add("")
    add("`bert-base-uncased`. The Python tokenizer is the Rust `tokenizers` crate behind a thin")
    add("binding; GraviTokenizers is managed C#.")
    add("")
    add("| Measure | Python | HF.Net | |")
    add("|---|---:|---:|---|")
    add(row("One document", "tokenizer_single_ms", py, net))
    add(row("1,000 documents", "tokenizer_batch_1000_ms", py, net))
    add("")

    py_rate = py.get("tokenizer_docs_per_second")
    net_rate = net.get("tokenizer_docs_per_second")
    if py_rate and net_rate:
        add(f"Throughput: **{py_rate:,.0f} docs/s** (Python) against "
            f"**{net_rate:,.0f} docs/s** (HF.Net).")
        add("")

    # Correctness is the headline, not the speed.
    py_ids = py.get("tokenizer_reference_ids")
    net_ids = net.get("tokenizer_reference_ids")
    if py_ids and net_ids:
        identical = list(py_ids) == list(net_ids)
        add(f"**Ids identical to the reference: {'yes' if identical else 'NO'}.**")
        add("")
        add("```")
        add(f"input   {'Hello, world! Tokenizers are unbelievable.'}")
        add(f"python  {' '.join(str(i) for i in py_ids)}")
        add(f"hf.net  {' '.join(str(i) for i in net_ids)}")
        add("```")
        add("")

    # ------------------------------------------------------------------ safetensors
    if "safetensors_header_ms" in py or "safetensors_header_ms" in net:
        add("## Reading safetensors")
        add("")
        size = py.get("safetensors_file_mb") or net.get("safetensors_file_mb")
        count = py.get("safetensors_tensor_count") or net.get("safetensors_tensor_count")
        add(f"`model.safetensors` for `bert-base-uncased` — {fmt(size, ' MB', 1)}, "
            f"{count} tensors.")
        add("")
        add("| Measure | Python | HF.Net | |")
        add("|---|---:|---:|---|")
        add(row("Open and list tensors", "safetensors_header_ms", py, net))
        add(row("Read one 30522x768 tensor", "safetensors_one_tensor_ms", py, net))
        add("")
        add("Reading one tensor costs HF.Net more because every value is widened to `double` on the")
        add("way out, where PyTorch hands back the F32 buffer as it lies on disk. Listing the")
        add("tensors touches only the header in both.")
        add("")

    # ------------------------------------------------------------------ inference
    if "inference_base_single_ms" in py or "inference_base_single_ms" in net:
        add("## Encoder inference")
        add("")
        tokens = py.get("inference_base_tokens") or net.get("inference_base_tokens")
        add(f"One forward pass over {tokens} tokens.")
        add("")
        add("| Model | Python (torch) | HF.Net (managed) | |")
        add("|---|---:|---:|---|")
        add(row("bert-base-uncased, 1 document", "inference_base_single_ms", py, net))
        add(row("bert-base-uncased, 8 documents", "inference_base_batch8_ms", py, net))
        add(row("bert-tiny, 1 document", "inference_tiny_single_ms", py, net))
        add("")
        add("**This is the gap, and it is the expected one.** The managed encoder is `double` end to")
        add("end and written for clarity; torch dispatches to hand-tuned single-precision kernels")
        add("with fused attention. GraviTransformers exists so a model can be loaded, inspected and")
        add("understood in pure .NET — for throughput the answer is an ONNX export.")
        add("")

    if "onnx_single_ms" in net:
        add("### The same work through ONNX Runtime")
        add("")
        add(f"`{net.get('onnx_provider', 'Cpu')}` provider, via GraviOptimum:")
        add("")
        add(f"- **{fmt(net.get('onnx_single_ms'), ' ms')}** best, "
            f"{fmt(net.get('onnx_single_median_ms'), ' ms')} median")
        add("")
        add("Measured on a small test model rather than on bert-base, so it is not comparable with")
        add("the table above. It is here to show the shape of the production path: the same")
        add("single-precision kernels torch uses, reached from .NET.")
        add("")

    # ------------------------------------------------------------------ fill-mask
    for key, prompt in (("france", "The capital of France is [MASK]."),
                        ("player", "He was a [MASK] player in the national team.")):
        p = py.get(f"fillmask_{key}")
        n = net.get(f"fillmask_{key}")
        if not p or not n:
            continue

        if key == "france":
            add("## Do they agree?")
            add("")
            add("Speed is the easy half. This is the half that matters: the same prompt, the same")
            add("checkpoint, and whether HF.Net's encoder reaches the same conclusions.")
            add("")

        add(f"**`{prompt}`**")
        add("")
        add("| Rank | Python | | HF.Net | |")
        add("|---|---|---:|---|---:|")
        for i in range(min(5, len(p), len(n))):
            add(f"| {i + 1} | {p[i]['token']} | {p[i]['score']:.2%} "
                f"| {n[i]['token']} | {n[i]['score']:.2%} |")
        add("")

        top_match = p[0]["token"] == n[0]["token"]
        overlap = len({x["token"] for x in p[:5]} & {x["token"] for x in n[:5]})
        add(f"Top prediction matches: **{'yes' if top_match else 'no'}**. "
            f"Overlap in the top five: **{overlap}/5**.")
        add("")

    add("## Reproducing this")
    add("")
    add("```bash")
    add("cd benchmarks/comparison")
    add("pip install tokenizers transformers")
    add("pip install torch --index-url https://download.pytorch.org/whl/cpu")
    add("python python/bench.py --out python/python.json")
    add("dotnet run -c Release --project HFNet.Comparison -- dotnet.json")
    add("python report.py")
    add("```")
    add("")
    add("Run them in the same session, and do not compare numbers taken on different days — on a")
    add("throttling laptop that difference exceeds most of what is measured here.")

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
