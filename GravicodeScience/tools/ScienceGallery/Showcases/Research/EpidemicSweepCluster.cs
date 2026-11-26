using Gravicode.Science.DaskNet;
using Gravicode.Science.DaskNet.Distributed;
using Gravicode.Science.GraviNum;
using Matplotlib.Net.Artists;
using Matplotlib.Net.Rendering;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;

namespace ScienceGallery.Showcases.Research;

/// <summary>A parameter sweep of a stochastic epidemic model, run as delayed tasks on a local Dask.Net cluster.</summary>
public sealed class EpidemicSweepCluster : Showcase
{
    public override string Title => "Sweep an epidemic model across R0 on a four-worker cluster";
    public override Domain Domain => Domain.Research;
    public override GraviLibrary[] Libraries => [GraviLibrary.DaskNet, GraviLibrary.GraviNum];
    public override string Summary => "480 stochastic SIR simulations as delayed tasks on a local cluster, checked against the final-size equation.";
    public override string Scenario =>
        "An epidemiologist wants to know how large an outbreak gets for each basic reproduction number R0, and " +
        "how often it fizzles out by chance. Each simulation is cheap but there are hundreds of them, so the sweep " +
        "is a graph of delayed tasks sent to a Dask.Net cluster of four workers. The tasks are [RemoteTask] " +
        "methods - their names cross the wire, never code - so the same graph would run unchanged on workers on " +
        "other machines. The mean of the outbreaks that took off is compared with the classical final-size equation.";

    // What a worker may run: public static methods marked [RemoteTask]. Arguments and results are plain
    // numbers and arrays, which the wire codec carries.

    /// <summary>A batch of stochastic SIR epidemics (Gillespie), each started by two cases in a town of
    /// 2,000. Returns the fraction of the town ever infected in each run.</summary>
    [RemoteTask]
    public static double[] Simulate(double r0, double seed, double runs)
    {
        const int population = 2_000;
        var rng = new GraviRandom((int)seed);
        var sizes = new double[(int)runs];
        for (var run = 0; run < sizes.Length; run++)
        {
            int s = population - 2, i = 2;
            while (i > 0)
            {
                var infection = r0 * s * i / population;     // recovery rate 1 per infectious person
                if (rng.NextDouble() * (infection + i) < infection) { s--; i++; } else i--;
            }
            sizes[run] = (population - s) / (double)population;
        }
        return sizes;
    }

    /// <summary>Four batches combined: [mean size of the outbreaks that took off (over 10%), share that
    /// took off, then every run's size].</summary>
    [RemoteTask]
    public static double[] Summarise(double[] a, double[] b, double[] c, double[] d)
    {
        double[] sizes = [.. a, .. b, .. c, .. d];
        var major = sizes.Where(z => z > 0.1).ToArray();
        return [major.Length == 0 ? 0 : major.Average(), major.Length / (double)sizes.Length, .. sizes];
    }

    public override void Run(ShowcaseOutput output)
    {
        var r0s = Enumerable.Range(0, 24).Select(k => 0.6 + 0.1 * k).ToArray();   // 0.6 .. 2.9
        const int batches = 4, runsPerBatch = 5, replicates = batches * runsPerBatch;

        // The graph: four Simulate batches per R0, each feeding that R0's Summarise. Every task names a
        // [RemoteTask] method, so the client can ship all of it (a lambda anywhere would be refused).
        var summaries = new Delayed<double[]>[r0s.Length];
        for (var k = 0; k < r0s.Length; k++)
        {
            var batch = Enumerable.Range(0, batches)
                .Select(b => Dask.Delayed(Simulate, Dask.Value(r0s[k]), Dask.Value(1000.0 * k + b), Dask.Value((double)runsPerBatch)))
                .ToArray();
            summaries[k] = Dask.Delayed(Summarise, batch[0], batch[1], batch[2], batch[3]);
        }
        var graph = Dask.Graph(summaries);

        // Four workers in this process. They still talk over TCP with serialised tasks and an
        // authenticated handshake, so what runs here runs on a real cluster.
        double[][] summary;
        RunStatistics statistics;
        var cluster = LocalCluster.StartAsync(workers: 4, threadsPerWorker: 2).GetAwaiter().GetResult();
        try
        {
            summary = Dask.Compute(summaries, cluster.Client);
            statistics = cluster.Client.LastRun!;
        }
        finally
        {
            cluster.DisposeAsync().AsTask().GetAwaiter().GetResult();
        }
        var allSizes = summary.SelectMany(s => s.Skip(2)).ToArray();

        // Theory: a major outbreak infects the fraction z solving z = 1 - exp(-R0 z), and one started by
        // two cases takes off with probability 1 - (1/R0)^2.
        double FinalSizeEquation(double r0)
        {
            var z = 0.9;
            for (var it = 0; it < 200; it++) z = 1 - Math.Exp(-r0 * z);
            return r0 <= 1 ? 0 : z;
        }

        var figure = GalleryCharts.Figure(11, 4.8);
        var (left, right) = (figure.AddSubplot(1, 2, 1), figure.AddSubplot(1, 2, 2));
        var scatterX = r0s.SelectMany(r => Enumerable.Repeat(r, replicates)).ToArray();
        left.Scatter(scatterX, allSizes.Select(z => z * 100).ToArray(), "single simulations", Color.Parse("#C9D0D8"), sizes: [14]);
        left.Plot(r0s, summary.Select(s => s[0] * 100).ToArray(), "mean of major outbreaks", GalleryCharts.Of(GraviLibrary.DaskNet),
                  marker: MarkerStyle.Circle, markerSize: 4);
        var fine = Enumerable.Range(0, 231).Select(k => 0.6 + 0.01 * k).ToArray();
        left.Plot(fine, fine.Select(r => FinalSizeEquation(r) * 100).ToArray(), "final-size equation", Color.Parse(GalleryCharts.Ink),
                  lineWidth: 1.2, style: LineStyle.Dashed);
        left.Plot(r0s, summary.Select(s => s[1] * 100).ToArray(), "% of runs that took off", GalleryCharts.Of(GraviLibrary.GraviNum),
                  lineWidth: 1.4, style: LineStyle.Dotted);
        left.SetTitle("Share of the town ever infected").SetXLabel("basic reproduction number R0").SetYLabel("%");
        left.Legend(LegendLocation.UpperLeft);

        var workers = statistics.TasksPerWorker.OrderBy(w => w.Key, StringComparer.Ordinal).ToArray();
        var wx = Enumerable.Range(0, workers.Length).Select(i => (double)i).ToArray();
        right.Bar(wx, workers.Select(w => (double)w.Value).ToArray(), 0.6, color: GalleryCharts.Of(GraviLibrary.DaskNet));
        right.SetXTicks(wx, workers.Select(w => w.Key).ToArray());
        right.SetTitle("Tasks run by each worker").SetYLabel("tasks");
        output.Chart(figure);

        var atTwo = Array.FindIndex(r0s, r => Math.Abs(r - 2.0) < 1e-9);
        var gap = r0s.Select((r, k) => (r, k)).Where(p => p.r >= 1.5)
                     .Max(p => Math.Abs(summary[p.k][0] - FinalSizeEquation(p.r))) * 100;
        output.Metric("Simulations", (r0s.Length * replicates).ToString("N0"), $"{graph.Count:N0} tasks in the graph after fusion");
        output.Metric("Tasks run on the cluster", statistics.TasksRun.ToString("N0"), $"{workers.Length} workers, {statistics.Elapsed.TotalMilliseconds:F0} ms");
        output.Metric("Outbreak size at R0 = 2", $"{summary[atTwo][0] * 100:F1}%", $"equation says {FinalSizeEquation(2.0) * 100:F1}%");
        output.Metric("Worst gap to theory", $"{gap:F1} points", "R0 ≥ 1.5, finite-town and sampling effects");
        output.Line($"Below R0 = 1 every run fizzles out; at R0 = 2, {(1 - summary[atTwo][1]) * 100:F0}% of the {replicates} introductions still died out by chance " +
                    $"(theory (1/R0)² = {Math.Pow(0.5, 2) * 100:F0}%). The client placed each task on the worker holding its inputs, " +
                    "so the per-R0 summaries ran where their simulations already were.");
    }
}
