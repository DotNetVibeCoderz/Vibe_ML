using BenchmarkDotNet.Attributes;
using Gravicode.HFNet.GraviOptimum;
using Gravicode.HFNet.GraviTransformers;
using Gravicode.Science.GraviNum;

namespace Gravicode.HFNet.Benchmarks;

/// <summary>GraviOptimum: the latency ONNX Runtime takes off a bert-base forward pass.</summary>
/// <remarks>
/// <para>
/// The same 12-token input through the managed encoder and through ONNX Runtime's CPU provider. The
/// ONNX file is the export the comparison benchmark writes to
/// <c>benchmarks/comparison/onnx/bert-base-uncased.onnx</c>; run <c>python/bench.py</c> there once
/// to create it. Without it, the ONNX row reports itself unavailable rather than measuring
/// something else.
/// </para>
/// <para>
/// The build references the CPU package of ONNX Runtime. CUDA or DirectML need their own ORT
/// package, and with this one a GPU target falls back to the CPU, so no GPU row is claimed.
/// </para>
/// </remarks>
[BenchmarkCategory("GraviOptimum")]
public class OptimumBenchmarks
{
    private TransformerModel _managed = null!;
    private OnnxSession? _onnx;
    private Dictionary<string, NdArray> _feeds = [];
    private int[] _ids = [];
    private int[] _types = [];
    private int[] _mask = [];

    [GlobalSetup]
    public void Setup()
    {
        _managed = TransformerModel.Load("bert-base-uncased");
        var encoding = _managed.Tokenizer.Encode("HF.Net runs Hugging Face models natively in .NET.");
        _ids = encoding.ToIdArray();
        _types = [.. encoding.TypeIds];
        _mask = encoding.ToMaskArray();

        var path = FindExport();
        if (path is null)
        {
            Console.WriteLine("// bert-base-uncased.onnx not found; run benchmarks/comparison/python/bench.py to export it");
            return;
        }

        _onnx = OnnxSession.Open(path, ExecutionTarget.Cpu);
        _feeds = Optimum.BuildEncoderInputs(_managed.Tokenizer, "HF.Net runs Hugging Face models natively in .NET.", _onnx);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _managed.Dispose();
        _onnx?.Dispose();
    }

    private static string? FindExport()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null)
        {
            var candidate = Path.Combine(directory.FullName, "benchmarks", "comparison", "onnx", "bert-base-uncased.onnx");
            if (File.Exists(candidate)) return candidate;
            directory = directory.Parent;
        }

        return null;
    }

    [Benchmark(Baseline = true, Description = "bert-base, managed encoder")]
    public NdArray Managed() => _managed.Forward(_ids, _types, _mask);

    [Benchmark(Description = "bert-base, ONNX Runtime CPU")]
    public int Onnx() => _onnx is null ? 0 : _onnx.Run(_feeds).Count;
}
