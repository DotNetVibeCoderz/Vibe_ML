using System.Diagnostics;
using MediaPipeNet;
using MediaPipeNet.Cli;
using SixLabors.ImageSharp;
using MediaPipeNet.Imaging;
using MediaPipeNet.Inference;
using MediaPipeNet.Inference.Models;
using MediaPipeNet.Tasks.Vision;
using MediaPipeNet.Video.OpenCv;

// mediapipenet-cli — MediaPipe.NET command line. Created by Gravicode Studios, led by Kang Fadhil.
var cli = new Args(args);
if (cli.Command is null or "help" or "--help" or "-h") return Help();

try
{
    return cli.Command switch
    {
        "info" => Info(),
        "models" => await Models(cli),
        "benchmark" => Benchmark(cli),
        "video" => await Video(cli),
        _ when CliTaskFactory.Descriptions.ContainsKey(cli.Command) => RunImage(cli),
        _ => Fail($"Unknown command '{cli.Command}'. Run 'mediapipenet-cli help'."),
    };
}
catch (Exception e) when (e is MediaPipeNet.MediaPipeException or IOException or ArgumentException or InvalidOperationException)
{
    return Fail(e.Message);
}

static int Help()
{
    Console.WriteLine("""
        mediapipenet-cli — MediaPipe.NET from the command line
        Created by Gravicode Studios, led by Kang Fadhil

        USAGE
          mediapipenet-cli <task> <image> [--output annotated.png] [--json result.json] [options]
          mediapipenet-cli video <task> [--input video.mp4 | --camera 0] [--frames 300]
          mediapipenet-cli benchmark <task> <image> [--iterations 50]
          mediapipenet-cli models list | download [ids...] | verify
          mediapipenet-cli info

        TASKS
        """);
    foreach (var (k, v) in CliTaskFactory.Descriptions) Console.WriteLine($"  {k,-11} {v}");
    Console.WriteLine("""

        OPTIONS
          --provider auto|cpu|directml|cuda|coreml   Execution provider (default auto)
          --models-dir <dir>                          Directory with the .onnx models
          --threads <n>                               Intra-op threads
          --no-download                               Never download missing models
        """);
    return 0;
}

static int Info()
{
    Console.WriteLine($"MediaPipe.NET {ModelStore.LibraryVersion} — Gravicode Studios (Kang Fadhil)");
    Console.WriteLine($".NET {Environment.Version} on {System.Runtime.InteropServices.RuntimeInformation.OSDescription} ({System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture})");
    Console.WriteLine($"ONNX Runtime {ExecutionProviderSelector.GetRuntimeVersion() ?? "(native library not found)"}");
    Console.WriteLine($"Execution providers: {string.Join(", ", ExecutionProviderSelector.GetAvailableProviders())}");
    Console.WriteLine($"Model cache: {ModelStore.DefaultCacheDirectory}");
    return 0;
}

static async Task<int> Models(Args cli)
{
    var store = CreateStore(cli);
    switch (cli.Positional.ElementAtOrDefault(0) ?? "list")
    {
        case "list":
            Console.WriteLine($"{"ID",-30} {"SIZE",9}  {"PACKAGE",-38} LOCAL");
            foreach (var m in ModelCatalog.All)
                Console.WriteLine($"{m.Id,-30} {m.SizeBytes / 1048576.0,7:F1}MB  {m.PackageId,-38} {store.FindLocal(m) ?? "-"}");
            return 0;
        case "download":
            var ids = cli.Positional.Skip(1).ToArray();
            var models = ids.Length == 0 ? ModelCatalog.All : ids.Select(id => ModelCatalog.Find(id) ?? throw new ArgumentException($"Unknown model '{id}'.")).ToArray();
            var progress = new Progress<ModelDownloadProgress>(p => Console.Write($"\r  {p.Model.Id}: {p.Stage} {(p.Fraction is { } f ? f.ToString("P0") : "")}      "));
            foreach (var m in models)
            {
                var path = await store.GetModelPathAsync(m, progress);
                Console.WriteLine($"\r  {m.Id}: {path}                ");
            }
            return 0;
        case "verify":
            int bad = 0;
            foreach (var m in ModelCatalog.All)
            {
                var path = store.FindLocal(m);
                if (path is null) { Console.WriteLine($"  {m.Id,-30} missing"); continue; }
                bool ok = string.Equals(await ModelStore.ComputeSha256Async(path), m.Sha256, StringComparison.OrdinalIgnoreCase);
                if (!ok) bad++;
                Console.WriteLine($"  {m.Id,-30} {(ok ? "OK" : "CHECKSUM MISMATCH")}");
            }
            return bad == 0 ? 0 : 1;
        default:
            return Fail("Usage: mediapipenet-cli models list|download|verify");
    }
}

static int RunImage(Args cli)
{
    var input = cli.Positional.ElementAtOrDefault(0) ?? throw new ArgumentException("An input image path is required.");
    using var task = CliTaskFactory.Create(cli.Command!, CreateBaseOptions(cli), RunningMode.Image);
    using var image = MPImage.Load(input);
    var sw = Stopwatch.StartNew();
    var result = task.Run(image);
    sw.Stop();
    Console.WriteLine($"{cli.Command}: {task.Summarize(result)}  [{sw.Elapsed.TotalMilliseconds:F1} ms]");
    if (cli.Get("--json") is { } json)
    {
        File.WriteAllText(json, task.ToJson(result));
        Console.WriteLine($"JSON written to {json}");
    }
    else if (cli.Has("--json-stdout"))
    {
        Console.WriteLine(task.ToJson(result));
    }
    if (cli.Get("--output") is { } output)
    {
        using var annotated = image.ToImage();
        task.Render(annotated, result);
        annotated.Save(output);
        Console.WriteLine($"Annotated image written to {output}");
    }
    return 0;
}

static int Benchmark(Args cli)
{
    var name = cli.Positional.ElementAtOrDefault(0) ?? throw new ArgumentException("A task name is required.");
    var input = cli.Positional.ElementAtOrDefault(1) ?? throw new ArgumentException("An input image path is required.");
    int iterations = int.Parse(cli.Get("--iterations") ?? "50", System.Globalization.CultureInfo.InvariantCulture);
    using var task = CliTaskFactory.Create(name, CreateBaseOptions(cli), RunningMode.Image);
    using var image = MPImage.Load(input);
    for (int i = 0; i < 5; i++) task.Run(image); // warm-up
    var times = new double[iterations];
    for (int i = 0; i < iterations; i++)
    {
        long t0 = Stopwatch.GetTimestamp();
        task.Run(image);
        times[i] = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
    }
    Array.Sort(times);
    Console.WriteLine($"{name} on {Path.GetFileName(input)} ({image.Width}x{image.Height}), {iterations} runs:");
    Console.WriteLine($"  mean {times.Average():F2} ms | p50 {times[iterations / 2]:F2} ms | p95 {times[(int)(iterations * 0.95)]:F2} ms | min {times[0]:F2} ms | {1000 / times.Average():F1} FPS");
    return 0;
}

static async Task<int> Video(Args cli)
{
    var name = cli.Positional.ElementAtOrDefault(0) ?? throw new ArgumentException("A task name is required.");
    int maxFrames = int.Parse(cli.Get("--frames") ?? "0", System.Globalization.CultureInfo.InvariantCulture);
    IFrameSource source = cli.Get("--input") is { } file
        ? new VideoFileFrameSource(file)
        : new WebcamFrameSource(int.Parse(cli.Get("--camera") ?? "0", System.Globalization.CultureInfo.InvariantCulture));
    using var task = CliTaskFactory.Create(name, CreateBaseOptions(cli), RunningMode.Video);
    using var cts = new CancellationTokenSource();
    Console.CancelKeyPress += (_, e) => { e.Cancel = true; cts.Cancel(); };
    await using var processor = new MediaPipeNet.Tasks.Vision.LiveStreamProcessor<object>(source, task.RunVideo);
    processor.ResultReady += (_, r) =>
    {
        Console.Write($"\r#{r.Stats.FramesProcessed,5} {r.Stats.ProcessingFps,5:F1} fps {r.Latency.TotalMilliseconds,6:F1} ms  dropped {r.Stats.FramesDropped,4}  {task.Summarize(r.Result),-60}");
        if (maxFrames > 0 && r.Stats.FramesProcessed >= maxFrames) cts.Cancel();
    };
    Console.WriteLine($"Processing {source.Name} — press Ctrl+C to stop");
    await processor.RunAsync(cts.Token);
    Console.WriteLine();
    return 0;
}

static ModelStore CreateStore(Args cli) =>
    ModelStore.CreateDefault(cli.Get("--models-dir"), allowDownload: !cli.Has("--no-download"));

static BaseOptions CreateBaseOptions(Args cli)
{
    var provider = (cli.Get("--provider") ?? "auto").ToLowerInvariant() switch
    {
        "cpu" => ExecutionProvider.Cpu,
        "directml" or "dml" => ExecutionProvider.DirectML,
        "cuda" => ExecutionProvider.Cuda,
        "coreml" => ExecutionProvider.CoreML,
        _ => ExecutionProvider.Auto,
    };
    int threads = int.Parse(cli.Get("--threads") ?? "0", System.Globalization.CultureInfo.InvariantCulture);
    return new BaseOptions
    {
        ModelStore = CreateStore(cli),
        Inference = new InferenceOptions { Provider = provider, IntraOpThreads = threads },
    };
}

static int Fail(string message)
{
    Console.Error.WriteLine($"error: {message}");
    return 1;
}

/// <summary>Minimal argument parser: first token is the command, --name value pairs, the rest positional.</summary>
internal sealed class Args
{
    private readonly Dictionary<string, string?> _options = new(StringComparer.OrdinalIgnoreCase);

    public Args(string[] args)
    {
        var positional = new List<string>();
        for (int i = 0; i < args.Length; i++)
        {
            if (i == 0 && !args[0].StartsWith("--", StringComparison.Ordinal)) { Command = args[0].ToLowerInvariant(); continue; }
            if (args[i].StartsWith("--", StringComparison.Ordinal))
            {
                bool hasValue = i + 1 < args.Length && !args[i + 1].StartsWith("--", StringComparison.Ordinal) && args[i] is not ("--no-download" or "--json-stdout");
                _options[args[i]] = hasValue ? args[++i] : null;
            }
            else positional.Add(args[i]);
        }
        Positional = positional;
    }

    public string? Command { get; }
    public IReadOnlyList<string> Positional { get; }
    public string? Get(string name) => _options.GetValueOrDefault(name);
    public bool Has(string name) => _options.ContainsKey(name);
}
