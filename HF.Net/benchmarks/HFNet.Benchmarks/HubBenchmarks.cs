using BenchmarkDotNet.Attributes;
using Gravicode.HFNet.GraviHub;
using Gravicode.HFNet.GraviHub.Io;

namespace Gravicode.HFNet.Benchmarks;

/// <summary>GraviHub: transfer speed from the Hub, and reading a checkpoint once it is here.</summary>
/// <remarks>
/// The download goes into a fresh cache directory every iteration, so each one really transfers the
/// file rather than finding it cached. It measures this connection as much as the client: divide
/// 17.7 MB by the time for the rate, and compare it with a browser download of the same file
/// before reading anything into it.
/// </remarks>
[BenchmarkCategory("GraviHub")]
public class HubBenchmarks
{
    private string _cacheRoot = "";
    private string _checkpoint = "";

    [GlobalSetup]
    public void Setup()
    {
        _cacheRoot = Path.Combine(Path.GetTempPath(), "hfnet-bench-hub");
        _checkpoint = Hub.DownloadFile("bert-base-uncased", "model.safetensors");
    }

    [IterationSetup(Target = nameof(Download))]
    public void Clear()
    {
        if (Directory.Exists(_cacheRoot)) Directory.Delete(_cacheRoot, recursive: true);
    }

    [Benchmark(Description = "Download prajjwal1/bert-tiny pytorch_model.bin (17.7 MB)")]
    public long Download()
    {
        var client = new HubClient(new HubOptions { CacheRoot = _cacheRoot });
        var path = client.DownloadFileAsync("prajjwal1/bert-tiny", "pytorch_model.bin").GetAwaiter().GetResult();
        return new FileInfo(path).Length;
    }

    [Benchmark(Description = "Open bert-base model.safetensors, list 206 tensors")]
    public int ListTensors()
    {
        using var reader = SafeTensors.Open(_checkpoint);
        return reader.Tensors.Count;
    }
}
