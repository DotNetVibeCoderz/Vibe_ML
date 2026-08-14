using System.Diagnostics;
using System.Globalization;
using Gravicode.Science.GraviLearn.Distributed;
using Gravicode.Science.GraviNum;

// Verifies that FileTransport genuinely crosses a process boundary, by spawning real worker
// processes rather than threads.
//
// The in-process tests establish that the ARITHMETIC is right. They cannot establish that the
// transport works between processes, because threads share a heap and a rename that is not atomic
// would still look fine. This runs the same computation as N separate OS processes coordinating
// only through a directory, and checks the aggregate against the single-process answer.
//
//   dotnet run --project tools/verify/DistributedInterop -c Release -- coordinate 4

if (args.Length == 0)
{
    Console.Error.WriteLine("usage: DistributedInterop <coordinate <workers> | worker <dir> <index> <workers>>");
    return 1;
}

// The problem every process agrees on: a fixed set of samples, sharded by index.
const int Samples = 1000;
const int Rows = 3;
const int Columns = 4;

static NdArray SampleAt(int index)
{
    // Derived from the index alone, so every process generates the same data without shipping it.
    var rng = new GraviRandom(1000 + index);
    return rng.StandardNormal(Rows, Columns);
}

switch (args[0])
{
    // ------------------------------------------------------------------ worker
    case "worker":
    {
        var directory = args[1];
        var index = int.Parse(args[2]);
        var workers = int.Parse(args[3]);

        var shard = DataParallel.Partition(Samples, workers)[index];

        // The mean over this worker's shard only.
        var partial = NdArray.Zeros(Rows, Columns);
        foreach (var i in shard.Indices)
        {
            var sample = SampleAt(i);
            for (var k = 0; k < partial.Size; k++)
                partial.SetAt(k, partial.At(k) + sample.At(k) / shard.Count);
        }

        using var transport = new FileTransport(directory, workers, TimeSpan.FromMinutes(2));
        new ParameterServer(transport).Contribute(index, partial, shard.Count);

        Console.WriteLine($"worker {index} published {shard} ({shard.Count} samples)");
        return 0;
    }

    // ------------------------------------------------------------- coordinator
    case "coordinate":
    {
        var workers = args.Length > 1 ? int.Parse(args[1]) : 4;
        var directory = Path.Combine(Path.GetTempPath(), $"gravi-dist-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);

        try
        {
            // The answer a single process would compute.
            var expected = NdArray.Zeros(Rows, Columns);
            for (var i = 0; i < Samples; i++)
            {
                var sample = SampleAt(i);
                for (var k = 0; k < expected.Size; k++)
                    expected.SetAt(k, expected.At(k) + sample.At(k) / Samples);
            }

            var self = Environment.ProcessPath ?? "dotnet";
            var assembly = System.Reflection.Assembly.GetEntryAssembly()!.Location;

            var processes = new List<Process>();
            for (var w = 0; w < workers; w++)
            {
                var start = new ProcessStartInfo
                {
                    FileName = self,
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                };

                // Launched through the host when running from a framework-dependent build.
                if (!self.EndsWith("DistributedInterop.exe", StringComparison.OrdinalIgnoreCase))
                    start.ArgumentList.Add(assembly);

                start.ArgumentList.Add("worker");
                start.ArgumentList.Add(directory);
                start.ArgumentList.Add(w.ToString());
                start.ArgumentList.Add(workers.ToString());

                processes.Add(Process.Start(start)!);
            }

            var pids = new List<int>();
            foreach (var process in processes)
            {
                pids.Add(process.Id);
                Console.Write("  " + process.StandardOutput.ReadToEnd());
                process.WaitForExit();

                if (process.ExitCode != 0)
                {
                    Console.WriteLine($"FAIL worker exited with {process.ExitCode}");
                    return 1;
                }
            }

            Console.WriteLine($"  {pids.Distinct().Count()} distinct process ids: {string.Join(", ", pids)}");

            // Collect what the separate processes wrote and aggregate.
            using var transport = new FileTransport(directory, workers, TimeSpan.FromSeconds(30));
            var aggregated = new ParameterServer(transport).Aggregate();

            var largest = 0.0;
            for (var k = 0; k < expected.Size; k++)
                largest = Math.Max(largest, Math.Abs(expected.At(k) - aggregated.At(k)));

            Console.WriteLine($"  largest difference from the single-process answer: " +
                              $"{largest.ToString("E3", CultureInfo.InvariantCulture)}");

            // Weighted averaging of shard means is exact in real arithmetic; in floating point the
            // regrouping costs a few ulps, so this is not asserted bit-for-bit.
            if (largest < 1e-12)
            {
                Console.WriteLine("PASS cross-process aggregate matches single-process");
                return 0;
            }

            Console.WriteLine("FAIL cross-process aggregate diverged");
            return 1;
        }
        finally
        {
            if (Directory.Exists(directory)) Directory.Delete(directory, recursive: true);
        }
    }

    default:
        Console.Error.WriteLine($"unknown command '{args[0]}'");
        return 1;
}
