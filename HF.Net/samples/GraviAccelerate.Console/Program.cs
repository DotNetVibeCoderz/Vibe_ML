using Gravicode.HFNet.GraviAccelerate;
using Gravicode.HFNet.GraviDatasets;
using Gravicode.Science.GraviLearn.Linear;
using Gravicode.Science.GraviNum;

// GraviAccelerate sample. Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.
Console.WriteLine("=== GraviAccelerate ===\n");

// --- 1. What this machine has -----------------------------------------------------
foreach (var device in Accelerator.Devices()) Console.WriteLine($"  {device}");
Console.WriteLine();

// --- 2. The blueprint's example: Accelerator.Train(model, dataset) ----------------
var titanic = Dataset.Load("titanic");
var accelerator = new Accelerator(DeviceKind.Cpu);
var report = accelerator.Train(
    new LogisticRegression(0.1, 500, 0.01, 1e-6),
    titanic,
    labelColumn: "survived",
    featureColumns: ["pclass", "sibsp", "parch", "fare"]);
Console.WriteLine($"logistic regression on titanic: {report}");
Console.WriteLine();

// --- 3. Training speed scaling: one data-parallel gradient, more workers ---------
// A logistic-regression gradient over 400,000 synthetic rows. Each worker takes a shard, and the
// shards' gradients are averaged weighted by size - the same reduction a multi-device trainer does.
const int Rows = 400_000, Features = 32;
var random = new GraviRandom(1);
var x = new double[Rows * Features];
var y = new double[Rows];
for (var i = 0; i < x.Length; i++) x[i] = random.Normal(0, 1);
for (var i = 0; i < Rows; i++) y[i] = x[i * Features] + 0.5 * x[i * Features + 1] > 0 ? 1 : 0;
var weights = new double[Features];

NdArray Gradient(int start, int count)
{
    var gradient = new double[Features];
    for (var r = start; r < start + count; r++)
    {
        var z = 0.0;
        for (var f = 0; f < Features; f++) z += x[r * Features + f] * weights[f];
        var error = 1 / (1 + Math.Exp(-z)) - y[r];
        for (var f = 0; f < Features; f++) gradient[f] += error * x[r * Features + f];
    }

    for (var f = 0; f < Features; f++) gradient[f] /= count;
    return new NdArray(gradient, Features);
}

TimeSpan? single = null;
foreach (var workers in new[] { 1, 2, 4, 8 })
{
    var parallel = new Accelerator(DeviceKind.Cpu, workers);
    var time = Accelerator.Measure(() => parallel.ParallelGradient(Rows, shard => Gradient(shard.Start, shard.Count)), iterations: 5, warmup: 2);
    single ??= time;
    Console.WriteLine($"  {workers} worker(s): {time.TotalMilliseconds,7:F1} ms   {single.Value / time,4:F2}x");
}

Console.WriteLine();

// --- 4. CPU against GPU on one large product ---------------------------------------
var a = new NdArray([.. Enumerable.Range(0, 512 * 512).Select(i => random.Normal(0, 1))], 512, 512);
var cpu = new Accelerator(DeviceKind.Cpu);
Console.WriteLine($"512x512 product on the CPU: {Accelerator.Measure(() => cpu.Dot(a, a), 5, 2).TotalMilliseconds:F1} ms");

var gpu = new Accelerator(DeviceKind.Gpu);
Console.WriteLine($"512x512 product on {(Accelerator.Devices().Any(d => d.Kind == DeviceKind.Gpu && d.Available) ? "the GPU" : "the CPU (no GPU found)")}: "
    + $"{Accelerator.Measure(() => gpu.Dot(a, a), 5, 2).TotalMilliseconds:F1} ms");
Console.WriteLine("Everything here is double precision, which consumer GPUs run at a fraction of their float32 rate.");
Accelerator.ReleaseGpu();
