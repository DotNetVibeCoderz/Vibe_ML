using BenchmarkDotNet.Attributes;
using Gravicode.HFNet.GraviAccelerate;
using Gravicode.HFNet.GraviTransformers;
using Gravicode.Science.GraviNum;

namespace Gravicode.HFNet.Benchmarks;

/// <summary>
/// GraviTransformers: one feed-forward-sized product on the CPU's SIMD units and on an ILGPU device,
/// then whole-encoder latency by sequence length.
/// </summary>
/// <remarks>
/// The product is <c>[128, 768] x [768, 3072]</c>, bert-base's first feed-forward layer at 128 tokens.
/// Three paths: HF.Net's register-blocked GEMM, the foundation's CPU <c>MatMul</c>, and the
/// foundation's ILGPU backend. Everything is double precision, which consumer GPUs run at a fraction
/// of their float32 rate - that is what the GPU row measures.
/// </remarks>
[BenchmarkCategory("GraviTransformers")]
public class TransformerProductBenchmarks
{
    private const int Rows = 128, Inputs = 768, Outputs = 3072;

    private Linear _layer = null!;
    private float[] _packed = [];
    private double[] _input = [];
    private NdArray _a = null!;
    private NdArray _b = null!;
    private Accelerator _cpu = null!;
    private Accelerator _gpu = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = new GraviRandom(1);
        var weight = new NdArray([.. Enumerable.Range(0, Outputs * Inputs).Select(_ => (double)(float)random.Normal(0, 0.02))], Outputs, Inputs);

        _layer = Linear.From(weight, null);
        _packed = Gemm.Pack([.. weight.ToArray().Select(v => (float)v)], Inputs, Outputs);
        _input = [.. Enumerable.Range(0, Rows * Inputs).Select(_ => random.Normal(0, 1))];
        _a = new NdArray(_input, Rows, Inputs);
        _b = weight.T.AsContiguous();
        _cpu = new Accelerator(DeviceKind.Cpu);
        _gpu = new Accelerator(DeviceKind.Gpu);
    }

    [GlobalCleanup]
    public void Cleanup() => Accelerator.ReleaseGpu();

    [Benchmark(Baseline = true, Description = "HF.Net GEMM (CPU, AVX2)")]
    public double[] HfNetGemm() => _layer.Apply(_input, Rows);

    [Benchmark(Description = "HF.Net GEMM, float32 (Precision.Single)")]
    public double[] HfNetGemmSingle() => Gemm.MultiplySingle(_input, Rows, _packed, Inputs, Outputs);

    [Benchmark(Description = "Foundation MatMul (CPU)")]
    public NdArray FoundationCpu() => _cpu.Dot(_a, _b);

    [Benchmark(Description = "Foundation ILGPU (GPU if present)")]
    public NdArray FoundationGpu() => _gpu.Dot(_a, _b);
}

/// <summary>GraviTransformers: bert-base forward-pass latency by sequence length.</summary>
[BenchmarkCategory("GraviTransformers")]
public class TransformerLatencyBenchmarks
{
    private TransformerModel _model = null!;
    private int[] _ids = [];
    private int[] _types = [];
    private int[] _mask = [];

    [Params(8, 32, 128, 512)]
    public int Tokens { get; set; }

    [Params(Precision.Double, Precision.Single)]
    public Precision Precision { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _model = TransformerModel.Load("bert-base-uncased");
        _ids = [101, .. Enumerable.Range(0, Tokens - 2).Select(i => 2000 + i % 5000), 102];
        _types = new int[Tokens];
        _mask = [.. Enumerable.Repeat(1, Tokens)];
        ComputeOptions.LinearLayers = Precision;
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        ComputeOptions.LinearLayers = Precision.Double;
        _model.Dispose();
    }

    [Benchmark(Description = "bert-base-uncased forward pass")]
    public NdArray Forward() => _model.Forward(_ids, _types, _mask);
}

/// <summary>GraviTransformers: GPT-2 generation, ViT and CLIP, each in both precisions.</summary>
[BenchmarkCategory("GraviTransformers")]
public class TransformerModelBenchmarks
{
    private CausalLanguageModel _gpt = null!;
    private Gravicode.HFNet.GraviTransformers.Vision.VisionTransformer _vit = null!;
    private Gravicode.HFNet.GraviTransformers.Vision.ClipModel _clip = null!;
    private NdArray _pixels = null!;
    private NdArray _clipPixels = null!;
    private int[] _prompt = [];

    [Params(Precision.Double, Precision.Single)]
    public Precision Precision { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _gpt = CausalLanguageModel.Load("gpt2");
        _prompt = [.. _gpt.Tokenizer!.Encode("The lighthouse keeper opened the door and", addSpecialTokens: false).Ids];
        _vit = Gravicode.HFNet.GraviTransformers.Vision.VisionTransformer.Load("google/vit-base-patch16-224");
        _clip = Gravicode.HFNet.GraviTransformers.Vision.ClipModel.Load("openai/clip-vit-base-patch32");

        var random = new GraviRandom(2);
        _pixels = new NdArray([.. Enumerable.Range(0, 3 * 224 * 224).Select(_ => random.Normal(0, 1))], 3, 224, 224);
        _clipPixels = _pixels;
        ComputeOptions.LinearLayers = Precision;
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        ComputeOptions.LinearLayers = Precision.Double;
        _gpt.Dispose();
        _vit.Dispose();
        _clip.Dispose();
    }

    [Benchmark(Description = "gpt2, 32 new tokens (greedy, KV cache)")]
    public int Gpt2() => _gpt.GenerateIds(_prompt, new GenerationSettings(MaxNewTokens: 32, StopAtEndToken: false)).Count();

    [Benchmark(Description = "vit-base-patch16-224, one image")]
    public NdArray Vit() => _vit.Forward(_pixels);

    [Benchmark(Description = "clip-vit-base-patch32, image vs 5 labels")]
    public double[] Clip() => _clip.Logits(_clipPixels, ["a bee", "a flower", "a butterfly", "a bird", "a cat"]);
}
