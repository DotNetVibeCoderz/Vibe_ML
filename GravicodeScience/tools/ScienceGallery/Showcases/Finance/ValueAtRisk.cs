using Gravicode.Science.GraviFrame;
using Gravicode.Science.SciNet;
using Matplotlib.Net.Artists;
using Matplotlib.Net.Rendering;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;
using Visual.Net.Core.Statistics;

namespace ScienceGallery.Showcases.Finance;

/// <summary>One-day Value at Risk and Expected Shortfall from real prices, under normal and fat-tailed models.</summary>
public sealed class ValueAtRisk : Showcase
{
    public override string Title => "How much could this position lose on a bad day?";
    public override Domain Domain => Domain.Finance;
    public override GraviLibrary[] Libraries => [GraviLibrary.SciNet, GraviLibrary.GraviFrame];
    public override string Summary => "Fit normal and Student-t laws to daily returns and compare their 99% VaR and Expected Shortfall.";
    public override string Scenario =>
        "A risk manager must report the one-day 99% Value at Risk of a Rp 1 billion position in GRVC shares: the " +
        "loss exceeded on only one trading day in a hundred. The normal distribution is the textbook choice, but " +
        "many stocks have fatter tails, where it understates big losses. Fitting a Student-t by maximum likelihood " +
        "to two years of prices from datasets/finance_timeseries.csv tests that instead of assuming it: its " +
        "degrees of freedom ν say how fat the tails are, and Expected Shortfall gives the average loss on the worst days.";

    public override void Run(ShowcaseOutput output)
    {
        const double position = 1_000_000_000;                     // Rp
        var prices = DataFrame.ReadCsv(Datasets.Path("finance_timeseries.csv"))
            .Filter(row => row.String("ticker") == "GRVC")
            .Numeric("close").Values;
        var returns = prices.Zip(prices.Skip(1), (a, b) => Math.Log(b / a)).ToArray();   // daily log returns

        var normal = Stats.Norm().FitTo(returns);
        var studentT = Stats.T(df: 5).FitTo(returns);
        var degrees = studentT.Shapes[0];

        // VaR is a quantile of the loss; Expected Shortfall is the mean loss beyond it.
        const double level = 0.99;
        double Loss(double logReturn) => position * (1 - Math.Exp(logReturn));
        double VaR(ContinuousDistribution d) => Loss(d.Ppf(1 - level));
        double Shortfall(ContinuousDistribution d) =>
            Loss(d.Expect(r => r, ub: d.Ppf(1 - level), conditional: true));
        var historicalVaR = Loss(Stats.Quantile(returns, 1 - level));
        // Backtest: on how many of the observed days did the loss exceed each model's VaR?
        int Breaches(double var) => returns.Count(r => Loss(r) > var);

        var figure = GalleryCharts.Figure(11, 4.8);
        var (left, right) = (figure.AddSubplot(1, 2, 1), figure.AddSubplot(1, 2, 2));
        var grid = Enumerable.Range(0, 300).Select(i => -0.09 + i * 0.0006).ToArray();
        left.Hist(returns.Select(r => r * 100).ToArray(), bins: 50, norm: PlotStats.HistNorm.Density, label: "GRVC daily returns", color: Color.Parse("#C9D0D8"));
        left.Plot(grid.Select(r => r * 100).ToArray(), grid.Select(r => normal.Pdf(r) / 100).ToArray(), "normal fit");
        left.Plot(grid.Select(r => r * 100).ToArray(), grid.Select(r => studentT.Pdf(r) / 100).ToArray(), $"Student-t fit (ν = {degrees:F1})");
        left.SetTitle("Distribution of daily returns").SetXLabel("daily return (%)").SetYLabel("density");
        left.Legend(LegendLocation.UpperLeft);

        // The left tail on a log scale, where the models disagree: P(loss larger than x).
        var losses = returns.Select(r => -r).OrderBy(l => l).ToArray();
        var tail = Enumerable.Range(0, losses.Length).Where(i => losses[i] > 0.005).ToArray();
        right.Scatter(tail.Select(i => losses[i] * 100).ToArray(), tail.Select(i => (losses.Length - i) / (double)losses.Length).ToArray(),
                      "observed", Color.Parse("#5F6B7A"), sizes: [14]);
        var x = Enumerable.Range(0, 200).Select(i => 0.005 + i * 0.0004).ToArray();
        right.Plot(x.Select(v => v * 100).ToArray(), x.Select(v => normal.Cdf(-v)).ToArray(), "normal");
        right.Plot(x.Select(v => v * 100).ToArray(), x.Select(v => studentT.Cdf(-v)).ToArray(), "Student-t");
        right.AxHLine(1 - level, Color.Parse("#D9363E"), LineStyle.Dotted, 1.2, "1 day in 100");
        right.SetYScale("log").SetYLim(5e-4, 1).SetTitle("Left tail: chance of a loss this large").SetXLabel("daily loss (%)").SetYLabel("probability");
        right.Legend(LegendLocation.UpperRight);
        output.Chart(figure);

        output.Metric("99% VaR, normal", $"Rp {VaR(normal) / 1e6:F1} million", $"{Breaches(VaR(normal))} breaches in {returns.Length} days (expect {returns.Length * (1 - level):F1})");
        output.Metric("99% VaR, Student-t", $"Rp {VaR(studentT) / 1e6:F1} million", $"{Breaches(VaR(studentT))} breaches in {returns.Length} days");
        output.Metric("99% VaR, historical", $"Rp {historicalVaR / 1e6:F1} million", "the empirical 1% quantile");
        output.Metric("Expected Shortfall", $"Rp {Shortfall(studentT) / 1e6:F1} million", $"Student-t; normal says Rp {Shortfall(normal) / 1e6:F1} million");
        output.Line(degrees < 15
            ? $"The fitted ν = {degrees:F1} shows clearly fat tails: the Student-t VaR is {(VaR(studentT) / VaR(normal) - 1) * 100:F0}% above the normal one, " +
              "and the normal model would under-reserve on exactly the days that matter."
            : $"The fitted ν = {degrees:F0} is far above the 3–6 typical of real equities: these returns show no fat tails, so the two models agree " +
              $"within {Math.Abs(VaR(studentT) / VaR(normal) - 1) * 100:F1}% and the normal VaR is defensible for this stock. The check, not the assumption, is the point.");
    }
}
