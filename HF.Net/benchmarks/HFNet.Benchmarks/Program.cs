using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Exporters;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;

// The per-library suites. Run everything with `dotnet run -c Release`, or one library with
// `dotnet run -c Release -- --filter *Tokenizer*`. Results land in BenchmarkDotNet.Artifacts.
//
// A short job on purpose: several suites load real models, and a laptop that throttles measures
// its own temperature if a run goes on long enough. The numbers to trust are the medians.
var config = ManualConfig.Create(DefaultConfig.Instance)
    .AddJob(Job.Default.WithWarmupCount(3).WithIterationCount(8).WithLaunchCount(1))
    .AddDiagnoser(MemoryDiagnoser.Default)
    .AddExporter(MarkdownExporter.GitHub)
    .WithOptions(ConfigOptions.JoinSummary);

BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args, config);
