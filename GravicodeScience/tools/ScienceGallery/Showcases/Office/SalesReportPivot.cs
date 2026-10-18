using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviNum;
using Matplotlib.Net.Artists;
using ScienceGallery.Models;
using ScienceGallery.Services;

namespace ScienceGallery.Showcases.Office;

/// <summary>The monthly sales report: a ledger of orders pivoted into months by region and products by region.</summary>
public sealed class SalesReportPivot : Showcase
{
    public override string Title => "Turn a year of orders into the monthly sales report";
    public override Domain Domain => Domain.Office;
    public override GraviLibrary[] Libraries => [GraviLibrary.GraviFrame, GraviLibrary.GraviNum];
    public override string Summary => "Group-by and pivot a 2,900-row order ledger into revenue by month, region and product.";
    public override string Scenario =>
        "Every month a sales administrator exports the order ledger and rebuilds the same pivot tables by hand. " +
        "Here the ledger of an office-equipment distributor with four branches goes through GraviFrame: a pivot " +
        "of revenue by month and branch, a pivot of product by branch, and a half-year comparison that shows " +
        "which branch is growing. The same code runs unchanged next month.";

    public override void Run(ShowcaseOutput output)
    {
        string[] months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
        string[] branches = ["Jakarta", "Bandung", "Surabaya", "Medan"];
        (string Name, double Price)[] products = [("Laptop", 11.5), ("Monitor", 2.4), ("Printer", 3.1), ("Toner", 0.65)]; // Rp million

        // The ledger: one row per order. Demand peaks before the fiscal year-end in December, and the
        // Surabaya branch, opened last year, is still ramping up.
        var rng = new GraviRandom(2025);
        var (month, branch, product) = (new List<string>(), new List<string>(), new List<string>());
        var (units, revenue) = (new List<double>(), new List<double>());
        for (var m = 0; m < 12; m++)
            for (var b = 0; b < branches.Length; b++)
            {
                var orders = rng.Poisson((b == 2 ? 30 + 5 * m : new[] { 85, 55, 0, 40 }[b]) * (m == 11 ? 1.5 : 1.0));
                for (var o = 0; o < orders; o++)
                {
                    var p = rng.Categorical([0.18, 0.22, 0.15, 0.45]);
                    var qty = 1 + rng.Poisson(p == 3 ? 6 : 1.5);
                    month.Add(months[m]); branch.Add(branches[b]); product.Add(products[p].Name);
                    units.Add(qty);
                    revenue.Add(Math.Round(qty * products[p].Price * rng.Uniform(0.9, 1.0), 2));   // up to 10% discount
                }
            }
        var ledger = new DataFrame(
        [
            new TextSeries("month", month.ToArray()), new TextSeries("branch", branch.ToArray()),
            new TextSeries("product", product.ToArray()),
            new NumericSeries("units", units.ToArray()), new NumericSeries("revenue", revenue.ToArray()),
        ]);

        // Pivot 1: month × branch, summed. Pivot 2: product × branch.
        var byMonth = ledger.Pivot("month", "branch", "revenue", aggregate: "sum");
        var byProduct = ledger.Pivot("product", "branch", "revenue", aggregate: "sum");
        var totals = ledger.GroupBy("branch").Sum("revenue");

        // Half-year growth per branch: second half against first half.
        var growth = branches.ToDictionary(b => b, b =>
        {
            var series = byMonth.Numeric(b).Values;
            return series.Skip(6).Sum() / series.Take(6).Sum() - 1;
        });
        var fastest = growth.MaxBy(g => g.Value).Key;
        var bestBranch = totals.SortBy("revenue", ascending: false).Text("branch")[0];

        var figure = GalleryCharts.Figure(11, 4.8);
        var (left, right) = (figure.AddSubplot(1, 2, 1), figure.AddSubplot(1, 2, 2));
        var x = Enumerable.Range(0, 12).Select(i => (double)i).ToArray();
        var bottom = new double[12];
        foreach (var b in branches)
        {
            var values = byMonth.Numeric(b).Values.Select(v => double.IsNaN(v) ? 0 : v / 1000).ToArray();   // Rp billion
            left.Bar(x, values, 0.75, bottoms: (double[])bottom.Clone(), label: b);
            for (var i = 0; i < 12; i++) bottom[i] += values[i];
        }
        left.SetXTicks(x, months);
        left.SetTitle("Revenue by month and branch").SetYLabel("Rp billion");
        left.Legend(LegendLocation.UpperLeft, columns: 2);

        var grid = new double[products.Length, branches.Length];
        var productNames = byProduct.Text("product");
        for (var p = 0; p < products.Length; p++)
            for (var b = 0; b < branches.Length; b++)
                grid[p, b] = byProduct.Numeric(branches[b])[p] / 1000;
        right.Heatmap(grid, Enumerable.Range(0, products.Length).Select(i => productNames[i] ?? "").ToArray(), branches,
                      colormap: "YlOrBr", annotate: true, annotationFormat: "0.00");
        right.SetTitle("Revenue by product and branch (Rp billion)");
        output.Chart(figure);

        output.Metric("Orders in the ledger", ledger.RowCount.ToString("N0"));
        output.Metric("Total revenue", $"Rp {revenue.Sum() / 1000:F1} billion");
        output.Metric("Largest branch", bestBranch ?? "", $"Rp {totals.SortBy("revenue", ascending: false).Numeric("revenue")[0] / 1000:F1} billion");
        output.Metric("Fastest growing", fastest, $"second half {growth[fastest] * 100:+0;-0}% on the first");
        var jakarta = byMonth.Numeric("Jakarta").Values;
        var december = jakarta[11] / jakarta.Take(11).Average();
        var laptopRow = Enumerable.Range(0, products.Length).First(p => productNames[p] == "Laptop");
        output.Line($"December runs {(december - 1) * 100:F0}% above an ordinary month in Jakarta: stock and staff for the year-end " +
                    $"push should be planned from November. Laptops bring in {byProduct.Numeric("Jakarta")[laptopRow] / jakarta.Sum() * 100:F0}% of Jakarta's revenue from {ledger.Filter(r => r.String("branch") == "Jakarta" && r.String("product") == "Laptop").RowCount * 100.0 / ledger.Filter(r => r.String("branch") == "Jakarta").RowCount:F0}% of its orders.");
    }
}
