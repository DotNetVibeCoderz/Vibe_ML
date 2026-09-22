using System.Diagnostics;
using System.Text.Json;
using Gravicode.HFNet.GraviHub;
using Gravicode.HFNet.GraviHub.Io;
using Gravicode.HFNet.GraviTokenizers;
using Gravicode.HFNet.GraviTransformers;

// The .NET half of the HF.Net comparison benchmark.
//
// Measures exactly the inputs benchmarks/comparison/python/bench.py measures, and writes the
// results as JSON so the two can be tabulated together.
//
// Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.

const string Model = "bert-base-uncased";
const string Tiny = "prajjwal1/bert-tiny";

// The same corpus as the Python half, written out here rather than loaded, so the two runtimes
// cannot disagree about what they measured.
string[] corpus =
[
    "Hello, world! Tokenizers are unbelievable.",
    "The capital of France is Paris, and it has been for a very long time.",
    "Quarterly revenue exceeded analyst expectations by a comfortable margin.",
    "A golden retriever played fetch in the park until the sun went down.",
    "Masked language modelling asks a model to fill in a blank word.",
    "Transformers process every position in parallel rather than in sequence.",
    "Byte-level encoding means any input round-trips exactly, emoji included.",
    "The board approved the merger this morning after a short discussion.",
];

var output = args.FirstOrDefault(a => a.EndsWith(".json", StringComparison.OrdinalIgnoreCase))
    ?? "dotnet.json";
var skipInference = args.Contains("--skip-inference");

var results = new Dictionary<string, object?>
{
    ["runtime"] = $".NET {Environment.Version}",
    ["platform"] = System.Runtime.InteropServices.RuntimeInformation.OSDescription,
    ["processor"] = $"{Environment.ProcessorCount} logical cores",
    ["configuration"] =
#if DEBUG
        "Debug",
#else
        "Release",
#endif
};

// ---------------------------------------------------------------- tokenizer
Console.WriteLine("tokenizer ...");
{
    var tokenizer = HfTokenizer.FromPretrained(Model);

    var (single, singleMedian) = Best(() => tokenizer.Encode(corpus[0]), iterations: 200, warmup: 50);

    var batch = Enumerable.Repeat(corpus, 125).SelectMany(c => c).ToList();
    var (total, totalMedian) = Best(() => tokenizer.EncodeBatch(batch), iterations: 10, warmup: 3);

    var encoded = tokenizer.Encode(corpus[0]);

    results["tokenizer_single_ms"] = single;
    results["tokenizer_single_median_ms"] = singleMedian;
    results["tokenizer_batch_1000_ms"] = total;
    results["tokenizer_batch_1000_median_ms"] = totalMedian;
    results["tokenizer_docs_per_second"] = batch.Count / (total / 1000.0);
    results["tokenizer_reference_ids"] = encoded.Ids;
    results["tokenizer_reference_tokens"] = encoded.Tokens;
}

// ---------------------------------------------------------------- safetensors
Console.WriteLine("safetensors ...");
try
{
    var path = Hub.DownloadFile(Model, "model.safetensors");

    var (header, _) = Best(() =>
    {
        using var reader = SafeTensors.Open(path);
        return reader.Tensors.Count;
    }, iterations: 20, warmup: 5);

    var (one, _) = Best(() =>
    {
        using var reader = SafeTensors.Open(path);
        return reader.Read("bert.embeddings.word_embeddings.weight");
    }, iterations: 10, warmup: 3);

    using var probe = SafeTensors.Open(path);

    results["safetensors_header_ms"] = header;
    results["safetensors_one_tensor_ms"] = one;
    results["safetensors_tensor_count"] = probe.Tensors.Count;
    results["safetensors_file_mb"] = new FileInfo(path).Length / (1024.0 * 1024);
}
catch (Exception error)
{
    Console.WriteLine($"  skipped: {error.Message}");
}

// ---------------------------------------------------------------- inference
if (!skipInference)
{
    Console.WriteLine("inference ...");

    foreach (var (label, id) in (( string Label, string Id )[])[("base", Model), ("tiny", Tiny)])
    {
        using var model = TransformerModel.Load(id);
        var encoded = model.Tokenizer.Encode(corpus[0]);

        var (single, singleMedian) = Best(() => model.Hidden(corpus[0]), iterations: 20, warmup: 10);
        var (batch, _) = Best(() => model.EmbedBatch(corpus), iterations: 10, warmup: 5);

        results[$"inference_{label}_single_ms"] = single;
        results[$"inference_{label}_single_median_ms"] = singleMedian;
        results[$"inference_{label}_batch8_ms"] = batch;
        results[$"inference_{label}_tokens"] = encoded.Length;
    }

    // Fill-mask, so the answers can be compared against the reference rather than against
    // this implementation's own output.
    Console.WriteLine("fill-mask ...");
    using var bert = TransformerModel.Load(Model);

    foreach (var (prompt, key) in ((string Prompt, string Key)[])
    [
        ("The capital of France is [MASK].", "france"),
        ("He was a [MASK] player in the national team.", "player"),
    ])
    {
        results[$"fillmask_{key}"] = bert.FillMask(prompt, topK: 5)
            .Select(f => new Dictionary<string, object> { ["token"] = f.Token, ["score"] = f.Score })
            .ToList();
    }
}

// ---------------------------------------------------------------- onnx
Console.WriteLine("onnx ...");
try
{
    const string OnnxModel = "hf-internal-testing/tiny-random-BertModel";

    using var session = Gravicode.HFNet.GraviOptimum.Optimum.Optimize(OnnxModel, "auto");
    var tokenizer = HfTokenizer.FromPretrained(OnnxModel);
    var feeds = Gravicode.HFNet.GraviOptimum.Optimum.BuildEncoderInputs(tokenizer, corpus[0], session.Session);

    var report = session.Measure(feeds, iterations: 50);

    results["onnx_provider"] = session.Target.ToString();
    results["onnx_single_ms"] = report.Best.TotalMilliseconds;
    results["onnx_single_median_ms"] = report.Median.TotalMilliseconds;
}
catch (Exception error)
{
    Console.WriteLine($"  skipped: {error.Message}");
}

File.WriteAllText(output, JsonSerializer.Serialize(results, new JsonSerializerOptions
{
    WriteIndented = true,
}));

Console.WriteLine($"wrote {output}");

/// <summary>
/// Runs an operation, discards the warm-up, and returns the best and median time in milliseconds.
/// </summary>
/// <remarks>
/// The best rather than the mean, for the same reason the Python half reports the best: on a
/// throttling laptop the mean measures the thermal state of the room. The warm-up must exceed
/// about thirty calls for the tiered JIT to have recompiled the hot path - a shorter one times the
/// wrong code entirely.
/// </remarks>
static (double Best, double Median) Best<T>(Func<T> work, int iterations, int warmup)
{
    for (var i = 0; i < warmup; i++) work();

    var samples = new List<double>(iterations);
    for (var i = 0; i < iterations; i++)
    {
        var stopwatch = Stopwatch.StartNew();
        work();
        stopwatch.Stop();
        samples.Add(stopwatch.Elapsed.TotalMilliseconds);
    }

    samples.Sort();
    return (samples[0], samples[samples.Count / 2]);
}
