using System.Globalization;
using Gravicode.Science.DaskNet;
using Gravicode.Science.GraviNum;
using Matplotlib.Net.Artists;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;

namespace ScienceGallery.Showcases.Utility;

/// <summary>A folder of web-server access logs read in byte blocks as a Bag: parse, filter, count, fold.</summary>
public sealed class WebLogAnalytics : Showcase
{
    public override string Title => "Find the failing endpoint in a day of web-server logs";
    public override Domain Domain => Domain.Utility;
    public override GraviLibrary[] Libraries => [GraviLibrary.DaskNet, GraviLibrary.GraviNum];
    public override string Summary => "Six log files read as a Bag in 256 KB blocks: status frequencies, errors per 10 minutes, slowest endpoints.";
    public override string Scenario =>
        "A site went flaky in the afternoon and the on-call engineer has six access-log files, one per server. " +
        "Rather than grep them one at a time, Dask.Net reads the folder as a Bag of lines split into byte blocks " +
        "(a line belongs to the block its first byte is in, so none is counted twice), parses each line, and " +
        "answers three questions in parallel: which status codes came back, when the errors happened, and which " +
        "endpoint was slow.";

    /// <summary>One parsed access-log line.</summary>
    private readonly record struct Hit(int Minute, string Method, string Path, int Status, double Milliseconds);

    public override void Run(ShowcaseOutput output)
    {
        string[] paths = ["/", "/search", "/product", "/cart", "/checkout", "/api/stock", "/login", "/static/app.js"];
        double[] weights = [0.16, 0.18, 0.24, 0.1, 0.05, 0.12, 0.05, 0.1];
        double[] typical = [40, 120, 70, 60, 180, 35, 90, 8];             // ms

        // Six servers, one log file each, in the common log format with the response time appended.
        // From 14:20 to 15:10 the stock service times out, which slows and breaks /api/stock and /checkout.
        var folder = Path.Combine(Path.GetTempPath(), "gallery-logs-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(folder);
        var rng = new GraviRandom(404);
        var day = new DateTime(2025, 6, 18);
        try
        {
            for (var server = 0; server < 6; server++)
            {
                using var writer = new StreamWriter(Path.Combine(folder, $"web-{server + 1}.log"));
                for (var second = 0; second < 86_400; second += 1 + rng.Poisson(2))
                {
                    var p = rng.Categorical(weights);
                    var incident = second is >= 14 * 3600 + 20 * 60 and < 15 * 3600 + 10 * 60 && (p == 5 || p == 4);
                    var ms = typical[p] * rng.LogNormal(0, 0.35) * (incident ? 30 : 1);
                    var status = incident && rng.NextBernoulli(0.45) ? 504
                        : rng.NextBernoulli(0.004) ? 500 : p == 6 && rng.NextBernoulli(0.08) ? 401 : rng.NextBernoulli(0.01) ? 404 : 200;
                    var method = p is 3 or 4 or 6 ? "POST" : "GET";
                    var time = day.AddSeconds(second).ToString("dd/MMM/yyyy:HH:mm:ss", CultureInfo.InvariantCulture);
                    writer.WriteLine($"10.0.{server}.{rng.Next(2, 250)} - - [{time} +0700] \"{method} {paths[p]} HTTP/1.1\" {status} {rng.Next(200, 9000)} {ms:F1}");
                }
            }

            // The Bag: lines in byte blocks, parsed lazily. Nothing has been read yet.
            var hits = Bag.ReadText(Path.Combine(folder, "*.log"), blockSize: 256 << 10)
                .Map(Parse);
            var statusCounts = hits.Map(h => h.Status).Frequencies();
            var errorsPerTenMinutes = hits.Filter(h => h.Status >= 500)
                .FoldBy(h => h.Minute / 10, () => 0L, (n, _) => n + 1, (a, b) => a + b);
            var perEndpoint = hits.FoldBy(h => h.Path, () => (Count: 0L, Total: 0.0, Slow: 0L),
                (acc, h) => (acc.Count + 1, acc.Total + h.Milliseconds, acc.Slow + (h.Milliseconds > 1000 ? 1 : 0)),
                (a, b) => (a.Count + b.Count, a.Total + b.Total, a.Slow + b.Slow));
            var slowest = hits.TopK(5, h => h.Milliseconds);

            // One graph for the three questions: the files are read and parsed once and all three are answered from that pass.
            var (statuses, errors, endpoints) = Dask.Compute(statusCounts, errorsPerTenMinutes, perEndpoint);
            var top = slowest.Compute();
            var total = statuses.Sum(s => s.Count);
            var blocks = Bag.ReadText(Path.Combine(folder, "*.log"), blockSize: 256 << 10).PartitionCount;

            var figure = GalleryCharts.Figure(11, 4.8);
            var (left, right) = (figure.AddSubplot(1, 2, 1), figure.AddSubplot(1, 2, 2));
            var buckets = Enumerable.Range(0, 144).Select(b => (double)b).ToArray();
            var errorSeries = buckets.Select(b => (double)errors.Where(e => e.Key == (int)b).Sum(e => e.Value)).ToArray();
            left.Bar(buckets.Select(b => b / 6).ToArray(), errorSeries, 1.0 / 6, color: Color.Parse("#D9363E"), label: "5xx responses");
            left.SetTitle("Server errors per 10 minutes").SetXLabel("hour of day").SetYLabel("responses");
            left.Legend(LegendLocation.UpperLeft);

            var ordered = endpoints.OrderBy(e => e.Value.Total / e.Value.Count).ToArray();
            var positions = Enumerable.Range(0, ordered.Length).Select(i => (double)i).ToArray();
            var means = ordered.Select(e => e.Value.Total / e.Value.Count).ToArray();
            right.BarH(positions, means,
                       colors: means.Select(m => m > 200 ? Color.Parse("#D9363E") : GalleryCharts.Of(GraviLibrary.DaskNet)).ToArray());
            right.SetYTicks(positions, ordered.Select(e => e.Key).ToArray());
            right.SetTitle("Mean response time by endpoint").SetXLabel("ms");
            output.Chart(figure);

            var worst = ordered[^1];
            var peak = errors.MaxBy(e => e.Value);
            var share5xx = statuses.Where(s => s.Item >= 500).Sum(s => s.Count) * 100.0 / total;
            output.Metric("Requests", total.ToString("N0"), $"6 files, {blocks} blocks of 256 KB");
            output.Metric("Server errors", $"{share5xx:F2}%", string.Join(", ", statuses.OrderByDescending(s => s.Count).Select(s => $"{s.Item}: {s.Count:N0}")));
            output.Metric("Worst 10 minutes", $"{peak.Key / 6:00}:{peak.Key % 6 * 10:00}", $"{peak.Value:N0} errors");
            output.Metric("Slowest endpoint", worst.Key, $"mean {worst.Value.Total / worst.Value.Count:F0} ms, {worst.Value.Slow:N0} requests over 1 s");
            output.Line($"The errors are concentrated between 14:20 and 15:10, and the slowest single requests were all to " +
                        $"{string.Join(" and ", top.Select(h => h.Path).Distinct())} ({top[0].Milliseconds / 1000:F1} s at worst): " +
                        "the stock service is the place to look, not the web servers.");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    /// <summary>Parses <c>ip - - [dd/MMM/yyyy:HH:mm:ss +zone] "METHOD path HTTP/1.1" status bytes ms</c>.</summary>
    private static Hit Parse(string line)
    {
        var open = line.IndexOf('[');
        var minute = int.Parse(line.AsSpan(open + 13, 2)) * 60 + int.Parse(line.AsSpan(open + 16, 2));
        var quote = line.IndexOf('"');
        var request = line[(quote + 1)..line.IndexOf('"', quote + 1)].Split(' ');
        var tail = line[(line.LastIndexOf('"') + 2)..].Split(' ');
        return new Hit(minute, request[0], request[1], int.Parse(tail[0]),
                       double.Parse(tail[2], CultureInfo.InvariantCulture));
    }
}
