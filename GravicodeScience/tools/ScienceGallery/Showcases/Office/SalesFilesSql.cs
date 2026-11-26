using Gravicode.Science.DaskNet.Sql;
using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviNum;
using Matplotlib.Net.Artists;
using Matplotlib.Net.Rendering;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;

namespace ScienceGallery.Showcases.Office;

/// <summary>Twelve monthly order exports queried as one table with SQL: running totals and a product ranking.</summary>
public sealed class SalesFilesSql : Showcase
{
    public override string Title => "Ask SQL questions of a folder of monthly sales files";
    public override Domain Domain => Domain.Office;
    public override GraviLibrary[] Libraries => [GraviLibrary.DaskNet, GraviLibrary.GraviFrame, GraviLibrary.GraviNum];
    public override string Summary => "Twelve CSV exports become one SQL table: a year-to-date running total per region and a product ranking.";
    public override string Scenario =>
        "The finance office receives one CSV export of the order system per month and is asked the same two " +
        "questions every quarter: how far along is each region towards its annual target, and which products " +
        "carry the year. Dask.Net's SqlContext registers the whole folder as one table, a window function keeps " +
        "a running total without a spreadsheet of cumulative formulas, and RANK() orders the products. The " +
        "answer is checked against the same numbers computed on one GraviFrame holding every row.";

    public override void Run(ShowcaseOutput output)
    {
        string[] regions = ["North", "South", "East", "West"];
        (string Name, double Price)[] products =
            [("Desk", 2.9), ("Chair", 1.4), ("Cabinet", 2.2), ("Lamp", 0.35), ("Whiteboard", 0.9), ("Shelf", 1.1)]; // Rp million
        double[] regionWeight = [1.3, 0.8, 1.0, 0.6];
        double[] productWeight = [0.14, 0.26, 0.08, 0.22, 0.12, 0.18];

        // One CSV per month in a scratch folder, the way the order system exports them. East has a
        // strong second half (a new key account from July); everyone peaks before year-end.
        var folder = Path.Combine(Path.GetTempPath(), "gallery-sales-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(folder);
        var rng = new GraviRandom(17);
        var all = new List<DataFrame>();
        try
        {
            for (var m = 1; m <= 12; m++)
            {
                var (month, region, product, units, revenue) = (new List<double>(), new List<string>(), new List<string>(), new List<double>(), new List<double>());
                for (var r = 0; r < regions.Length; r++)
                {
                    var seasonal = (m == 12 ? 1.45 : 1.0) * (r == 2 && m >= 7 ? 1.6 : 1.0);
                    var orders = rng.Poisson(260 * regionWeight[r] * seasonal);
                    for (var o = 0; o < orders; o++)
                    {
                        var p = rng.Categorical(productWeight);
                        var qty = 1 + rng.Poisson(p == 3 ? 4 : 1.2);
                        month.Add(m); region.Add(regions[r]); product.Add(products[p].Name); units.Add(qty);
                        revenue.Add(Math.Round(qty * products[p].Price * rng.Uniform(0.92, 1.0), 3));
                    }
                }
                var frame = new DataFrame(
                [
                    new NumericSeries("month", month.ToArray()), new TextSeries("region", region.ToArray()),
                    new TextSeries("product", product.ToArray()), new NumericSeries("units", units.ToArray()),
                    new NumericSeries("revenue", revenue.ToArray()),
                ]);
                frame.WriteCsv(Path.Combine(folder, $"orders-2025-{m:00}.csv"));
                all.Add(frame);
            }

            // The whole folder is one table; each file is a partition. Targets are a small in-memory table.
            var targets = new DataFrame(
            [
                new TextSeries("region", regions),
                new NumericSeries("target", [13200, 7400, 12000, 6100]),   // Rp million for the year
            ]);
            var sql = new SqlContext()
                .RegisterCsv("orders", Path.Combine(folder, "orders-*.csv"))
                .Register("targets", targets);

            const string runningTotal = """
                SELECT o.region, o.month, SUM(o.revenue) AS revenue,
                       SUM(SUM(o.revenue)) OVER (PARTITION BY o.region ORDER BY o.month) AS ytd,
                       t.target
                FROM orders o JOIN targets t ON o.region = t.region
                GROUP BY o.region, o.month, t.target
                ORDER BY o.region, o.month
                """;
            const string productRanking = """
                SELECT product, SUM(units) AS units, SUM(revenue) AS revenue,
                       RANK() OVER (ORDER BY SUM(revenue) DESC) AS rank
                FROM orders
                GROUP BY product
                ORDER BY rank
                """;
            var ytd = sql.Execute(runningTotal);
            var ranking = sql.Execute(productRanking);

            // The same numbers on one in-memory GraviFrame: concatenate, group, then a cumulative sum by hand.
            var ledger = DataFrame.Concat(all);
            var monthly = ledger.GroupBy("region", "month").Sum("revenue").SortBy([("region", true), ("month", true)]);
            var worst = 0.0;
            foreach (var region in regions)
            {
                var single = monthly.Filter(r => r.String("region") == region).Numeric("revenue").Values;
                var viaSql = ytd.Filter(r => r.String("region") == region).Numeric("ytd").Values;
                var running = 0.0;
                for (var i = 0; i < single.Length; i++)
                {
                    running += single[i];
                    worst = Math.Max(worst, Math.Abs(running - viaSql[i]));
                }
            }

            var figure = GalleryCharts.Figure(11, 4.8);
            var (left, right) = (figure.AddSubplot(1, 2, 1), figure.AddSubplot(1, 2, 2));
            var x = Enumerable.Range(1, 12).Select(i => (double)i).ToArray();
            var share = new Dictionary<string, double>();
            for (var r = 0; r < regions.Length; r++)
            {
                var rows = ytd.Filter(row => row.String("region") == regions[r]);
                var target = rows.Numeric("target")[0];
                var pct = rows.Numeric("ytd").Values.Select(v => v / target * 100).ToArray();
                share[regions[r]] = pct[^1];
                left.Plot(x, pct, regions[r], GalleryCharts.Cycle[r], marker: MarkerStyle.Circle);
            }
            left.AxHLine(100, Color.Parse(GalleryCharts.Ink), LineStyle.Dashed, 1.2, "annual target");
            left.SetXTicks(x, ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]);
            left.SetTitle("Year-to-date revenue, % of target (running SUM OVER)").SetYLabel("% of annual target");
            left.Legend(LegendLocation.UpperLeft);

            var names = Enumerable.Range(0, ranking.RowCount).Select(i => ranking.Text("product")[i] ?? "").ToArray();
            var positions = Enumerable.Range(0, names.Length).Select(i => (double)(names.Length - 1 - i)).ToArray();
            right.BarH(positions, ranking.Numeric("revenue").Values.Select(v => v / 1000).ToArray(),
                       color: GalleryCharts.Of(GraviLibrary.DaskNet));
            right.SetYTicks(positions, names.Select((n, i) => $"#{ranking.Numeric("rank")[i]:0} {n}").ToArray());
            right.SetTitle("Products ranked by revenue (RANK() OVER)").SetXLabel("Rp billion");
            output.Chart(figure);

            var leader = share.MaxBy(s => s.Value);
            var behind = share.MinBy(s => s.Value);
            output.Metric("Files queried as one table", "12", $"{ledger.RowCount:N0} order lines");
            output.Metric("Ahead of target", leader.Key, $"{leader.Value:F0}% of the annual target");
            output.Metric("Furthest behind", behind.Key, $"{behind.Value:F0}% of the annual target");
            output.Metric("Max difference vs GraviFrame", $"{worst:E1}", "running totals, Rp million");
            output.Line($"{names[0]} leads the ranking with Rp {ranking.Numeric("revenue")[0] / 1000:F2} billion; " +
                        $"{names[^1]} is last. East's second half is visible as the steeper slope from July.");
            output.Line("The optimised plan for the running-total query:\n" + sql.Explain(runningTotal));
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }
}
