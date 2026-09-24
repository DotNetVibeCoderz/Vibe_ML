"""
The Python half of the HF.Net comparison benchmark.

Measures the reference implementation on exactly the inputs the .NET half uses, and writes the
results as JSON so the two can be tabulated together.

Run:  python bench.py --out python.json

Both halves report the BEST of several timed runs rather than the mean. On a laptop that throttles,
the mean measures the thermal state of the room; the best is the closest either runtime gets to the
work itself.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

MODEL = "bert-base-uncased"
TINY = "prajjwal1/bert-tiny"
VISION = "google/vit-base-patch16-224"

# Where the ONNX export of MODEL is written for the .NET half to load. Gitignored: it is 420 MB and
# rebuilt from the checkpoint on every run, so the graph and the torch weights cannot drift apart.
ONNX_PATH = Path(__file__).resolve().parent.parent / "onnx" / f"{MODEL}.onnx"

# The corpus both halves tokenize. Written out here rather than loaded, so the two runtimes cannot
# disagree about what they measured.
CORPUS = [
    "Hello, world! Tokenizers are unbelievable.",
    "The capital of France is Paris, and it has been for a very long time.",
    "Quarterly revenue exceeded analyst expectations by a comfortable margin.",
    "A golden retriever played fetch in the park until the sun went down.",
    "Masked language modelling asks a model to fill in a blank word.",
    "Transformers process every position in parallel rather than in sequence.",
    "Byte-level encoding means any input round-trips exactly, emoji included.",
    "The board approved the merger this morning after a short discussion.",
]


def best(fn, iterations, warmup):
    """Runs fn, discarding warmup runs, and returns (best, median) in seconds."""
    for _ in range(warmup):
        fn()

    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)

    return min(samples), statistics.median(samples)


def load_tokenizer(model_id):
    """Loads the Rust tokenizer straight from tokenizer.json.

    transformers 5.x refuses to build a backend tokenizer from vocab.txt without sentencepiece
    installed, and the AutoTokenizer wrapper adds Python overhead that is not what we want to
    measure anyway. Going to the tokenizers library directly compares HF.Net against the fastest
    thing Python has.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import RemoteEntryNotFoundError
    from tokenizers import Tokenizer

    try:
        return Tokenizer.from_file(hf_hub_download(model_id, "tokenizer.json"))
    except RemoteEntryNotFoundError:
        # Plenty of repositories never published a tokenizer.json - prajjwal1/bert-tiny among
        # them. HF.Net falls back to vocab.txt on its own; here the base vocabulary stands in,
        # which is the same 30,522 WordPiece entries that model was trained on.
        print(f"  {model_id} has no tokenizer.json; using {MODEL}'s")
        return Tokenizer.from_file(hf_hub_download(MODEL, "tokenizer.json"))


def bench_tokenizer(results):
    tokenizer = load_tokenizer(MODEL)

    # One document, to expose per-call overhead.
    single, single_median = best(lambda: tokenizer.encode(CORPUS[0]), iterations=200, warmup=50)

    # The whole corpus, repeated, to measure throughput.
    batch = CORPUS * 125  # 1000 documents
    total, total_median = best(lambda: tokenizer.encode_batch(batch), iterations=10, warmup=3)

    encoded = tokenizer.encode(CORPUS[0])

    results["tokenizer_single_ms"] = single * 1000
    results["tokenizer_single_median_ms"] = single_median * 1000
    results["tokenizer_batch_1000_ms"] = total * 1000
    results["tokenizer_batch_1000_median_ms"] = total_median * 1000
    results["tokenizer_docs_per_second"] = len(batch) / total
    results["tokenizer_reference_ids"] = encoded.ids
    results["tokenizer_reference_tokens"] = encoded.tokens


def bench_safetensors(results):
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    path = hf_hub_download(MODEL, "model.safetensors")

    def read_header():
        with safe_open(path, framework="pt") as f:
            return list(f.keys())

    header, header_median = best(read_header, iterations=20, warmup=5)

    def read_one():
        with safe_open(path, framework="pt") as f:
            return f.get_tensor("bert.embeddings.word_embeddings.weight")

    one, one_median = best(read_one, iterations=10, warmup=3)

    with safe_open(path, framework="pt") as f:
        names = list(f.keys())

    results["safetensors_header_ms"] = header * 1000
    results["safetensors_one_tensor_ms"] = one * 1000
    results["safetensors_tensor_count"] = len(names)
    results["safetensors_file_mb"] = Path(path).stat().st_size / (1024 * 1024)


def load_model(model_id):
    """Loads an encoder, naming the architecture where the config does not.

    prajjwal1/bert-tiny's config.json has no model_type key, so AutoModel refuses it outright.
    HF.Net defaults a missing model_type to bert and loads the checkpoint; here it has to be said
    explicitly.
    """
    from transformers import AutoModel, BertModel

    try:
        return AutoModel.from_pretrained(model_id)
    except ValueError:
        print(f"  {model_id} declares no model_type; loading it as BERT")
        return BertModel.from_pretrained(model_id)


def bench_inference(results):
    import torch

    torch.set_num_threads(torch.get_num_threads())

    for label, model_id in (("base", MODEL), ("tiny", TINY)):
        tokenizer = load_tokenizer(model_id)
        model = load_model(model_id)
        model.eval()

        ids = tokenizer.encode(CORPUS[0]).ids
        inputs = {"input_ids": torch.tensor([ids])}

        def forward():
            with torch.no_grad():
                return model(**inputs).last_hidden_state

        single, single_median = best(forward, iterations=20, warmup=10)

        # Padded to the longest, exactly as HF.Net's EncodeBatch does.
        encoded = tokenizer.encode_batch(CORPUS)
        width = max(len(e.ids) for e in encoded)
        pad = tokenizer.token_to_id("[PAD]") or 0

        padded = [e.ids + [pad] * (width - len(e.ids)) for e in encoded]
        mask = [[1] * len(e.ids) + [0] * (width - len(e.ids)) for e in encoded]

        batch_inputs = {
            "input_ids": torch.tensor(padded),
            "attention_mask": torch.tensor(mask),
        }

        def forward_batch():
            with torch.no_grad():
                return model(**batch_inputs).last_hidden_state

        batch, batch_median = best(forward_batch, iterations=10, warmup=5)

        results[f"inference_{label}_single_ms"] = single * 1000
        results[f"inference_{label}_single_median_ms"] = single_median * 1000
        results[f"inference_{label}_batch8_ms"] = batch * 1000
        results[f"inference_{label}_tokens"] = len(ids)

    results["torch_threads"] = torch.get_num_threads()


def vit_pixels(size):
    """The picture both halves classify: a formula, not a file, so no resampler sits between them.

    Any real photograph has to be resized first, and PIL and ImageSharp resize differently - which
    would put a difference into the comparison that has nothing to do with the model.
    """
    import torch

    ys = torch.arange(size, dtype=torch.float64).view(1, size, 1)
    xs = torch.arange(size, dtype=torch.float64).view(1, 1, size)
    cs = torch.arange(3, dtype=torch.float64).view(3, 1, 1)
    return 0.9 * torch.sin(0.05 * xs + 0.07 * ys + cs) * torch.cos(0.013 * xs * (cs + 1))


def bench_vision(results):
    import torch
    from transformers import ViTForImageClassification

    model = ViTForImageClassification.from_pretrained(VISION).eval()
    pixels = vit_pixels(224).float().unsqueeze(0)

    def forward():
        with torch.no_grad():
            return model(pixel_values=pixels).logits

    single, single_median = best(forward, iterations=10, warmup=3)
    results["vision_single_ms"] = single * 1000
    results["vision_single_median_ms"] = single_median * 1000

    # The answers in float64, the precision the managed encoder computes in, so any disagreement
    # is the implementation rather than float32 rounding.
    exact = ViTForImageClassification.from_pretrained(VISION, torch_dtype=torch.float64).eval()
    with torch.no_grad():
        probabilities = torch.softmax(exact(pixel_values=vit_pixels(224).unsqueeze(0)).logits[0], -1)

    top = torch.topk(probabilities, 5)
    results["vision_top5"] = [
        {"label": exact.config.id2label[int(i)], "index": int(i), "score": float(v)}
        for v, i in zip(top.values, top.indices)
    ]


def export_onnx(results):
    """Exports MODEL to ONNX for the .NET half's production-path measurement."""
    import torch
    from transformers import BertModel

    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)

    encoder = BertModel.from_pretrained(MODEL).eval()

    class Graph(torch.nn.Module):
        """Takes the three inputs by name; transformers 5 moved BertModel's positional ones."""

        def __init__(self):
            super().__init__()
            self.encoder = encoder

        def forward(self, input_ids, attention_mask, token_type_ids):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                               token_type_ids=token_type_ids)
            return out.last_hidden_state, out.pooler_output

    model = Graph().eval()

    ids = torch.tensor([[101, 7592, 1010, 2088, 999, 102]])
    names = ["input_ids", "attention_mask", "token_type_ids"]
    dynamic = {name: {0: "batch", 1: "sequence"} for name in names}
    dynamic["last_hidden_state"] = {0: "batch", 1: "sequence"}

    torch.onnx.export(
        model,
        (ids, torch.ones_like(ids), torch.zeros_like(ids)),
        str(ONNX_PATH),
        input_names=names,
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes=dynamic,
        opset_version=17,
        dynamo=False,
    )

    results["onnx_export"] = str(ONNX_PATH)


def bench_fill_mask(results):
    """Records the reference answers, so the .NET half can be checked against them rather than
    against itself."""
    from transformers import pipeline

    filler = pipeline("fill-mask", model=MODEL)

    for prompt, key in (
        ("The capital of France is [MASK].", "france"),
        ("He was a [MASK] player in the national team.", "player"),
    ):
        predictions = filler(prompt, top_k=5)
        results[f"fillmask_{key}"] = [
            {"token": p["token_str"].strip(), "score": p["score"]} for p in predictions
        ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="python.json")
    parser.add_argument("--skip-inference", action="store_true")
    args = parser.parse_args()

    import platform
    import sys

    results = {
        "runtime": f"Python {sys.version.split()[0]}",
        "platform": platform.platform(),
        "processor": platform.processor(),
    }

    try:
        import transformers
        import tokenizers

        results["transformers"] = transformers.__version__
        results["tokenizers"] = tokenizers.__version__
    except ImportError as error:
        print(f"missing package: {error}")
        return 1

    print("tokenizer ...")
    bench_tokenizer(results)

    print("safetensors ...")
    try:
        bench_safetensors(results)
    except Exception as error:
        print(f"  skipped: {error}")

    if not args.skip_inference:
        print("inference ...")
        try:
            import torch

            results["torch"] = torch.__version__
            bench_inference(results)
            print("fill-mask ...")
            bench_fill_mask(results)
            print("vision ...")
            bench_vision(results)
            print("onnx export ...")
            try:
                export_onnx(results)
            except Exception as error:
                print(f"  skipped: {error}")
        except ImportError:
            print("  torch not installed, skipping")

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
