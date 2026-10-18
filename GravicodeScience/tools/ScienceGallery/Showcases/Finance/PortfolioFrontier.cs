using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Matplotlib.Net.Artists;
using Matplotlib.Net.Rendering;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;

namespace ScienceGallery.Showcases.Finance;

/// <summary>Markowitz portfolio optimisation: the efficient frontier and the best risk-adjusted mix.</summary>
public sealed class PortfolioFrontier : Showcase
{
    public override string Title => "The best mix of five assets for each level of risk";
    public override Domain Domain => Domain.Finance;
    public override GraviLibrary[] Libraries => [GraviLibrary.SciNet, GraviLibrary.GraviNum];
    public override string Summary => "Combine return views with covariances estimated from history, then trace the efficient frontier with SLSQP.";
    public override string Scenario =>
        "A pension fund's investment committee holds bank shares, telecoms, consumer goods, mining and " +
        "government bonds. Diversification lowers risk only as far as the assets move independently, which the " +
        "covariance matrix measures. For each target return, a constrained minimiser finds the least risky mix " +
        "(fully invested, no short selling, at most 50% in any one asset); the curve those mixes trace is the " +
        "efficient frontier, and the mix with the best return per unit of risk is the usual starting point.";

    public override void Run(ShowcaseOutput output)
    {
        string[] assets = ["Banks", "Telecoms", "Consumer", "Mining", "Gov. bonds"];
        const double riskFree = 0.055;                                        // annual, a deposit rate

        // Expected annual returns are the research team's forward-looking views: three years of history pin
        // a mean return down only to about ±15% for a volatile share, far too loosely to optimise on.
        double[] mu = [0.13, 0.09, 0.10, 0.16, 0.065];
        double[] annualVol = [0.24, 0.18, 0.16, 0.34, 0.05];
        double[,] correlation =
        {
            { 1.0, 0.55, 0.45, 0.40, -0.10 },
            { 0.55, 1.0, 0.40, 0.25, -0.05 },
            { 0.45, 0.40, 1.0, 0.20, 0.00 },
            { 0.40, 0.25, 0.20, 1.0, -0.15 },
            { -0.10, -0.05, 0.00, -0.15, 1.0 },
        };
        var dailyCov = NdArray.Zeros(5, 5);
        for (var i = 0; i < 5; i++)
            for (var j = 0; j < 5; j++) dailyCov[i, j] = correlation[i, j] * annualVol[i] * annualVol[j] / 252;
        var rng = new GraviRandom(8);
        // Risk, on the other hand, is estimated well from history: three years of daily returns (simulated
        // here from a known market), turned into an annualised sample covariance matrix.
        var daily = rng.MultivariateNormal(NdArray.FromValues(mu.Select(m => m / 252)), dailyCov, samples: 3 * 252);
        var sigma = Statistics.CovarianceMatrix(daily) * 252.0;
        double Return(double[] w) => w.Zip(mu, (a, b) => a * b).Sum();
        double Risk(double[] w) => Math.Sqrt(LinAlg.Inner(NdArray.FromValues(w), LinAlg.Dot(sigma, NdArray.FromValues(w))));

        // Least risk for a given return: fully invested, long only, at most 50% per asset.
        var bounds = new Bounds(Enumerable.Repeat(0.0, 5).ToArray(), Enumerable.Repeat(0.5, 5).ToArray());
        var budget = new OptimizeConstraint(OptimizeConstraintType.Eq, w => w.Sum() - 1);
        double[] MinRisk(double target) => Optimize.Minimize(w => Risk(w) * Risk(w), Enumerable.Repeat(0.2, 5).ToArray(), MinimizeMethod.Slsqp,
            bounds: bounds, constraints: [budget, new OptimizeConstraint(OptimizeConstraintType.Eq, w => Return(w) - target)]).X;

        // The frontier starts at the minimum-variance portfolio; below it, more risk buys less return.
        var minVariance = Optimize.Minimize(w => Risk(w) * Risk(w), Enumerable.Repeat(0.2, 5).ToArray(), MinimizeMethod.Slsqp,
                                            bounds: bounds, constraints: [budget]).X;
        var lowest = Return(minVariance);
        var highest = 0.5 * mu.OrderByDescending(m => m).Take(2).Sum() - 0.003;   // best reachable with the 50% cap
        var frontier = NdArray.Linspace(lowest, highest, 30).ToArray().Select(MinRisk).ToArray();
        // The tangency portfolio: the best Sharpe ratio, (return − deposit rate) / risk.
        var tangency = Optimize.Minimize(w => -(Return(w) - riskFree) / Risk(w), Enumerable.Repeat(0.2, 5).ToArray(), MinimizeMethod.Slsqp,
                                         bounds: bounds, constraints: [budget]).X;
        var equal = Enumerable.Repeat(0.2, 5).ToArray();

        var figure = GalleryCharts.Figure(11, 4.8);
        var (left, right) = (figure.AddSubplot(1, 2, 1), figure.AddSubplot(1, 2, 2));
        left.Plot(frontier.Select(w => Risk(w) * 100).ToArray(), frontier.Select(w => Return(w) * 100).ToArray(), "efficient frontier");
        left.Scatter(Enumerable.Range(0, 5).Select(i => Math.Sqrt(sigma[i, i]) * 100).ToArray(), mu.Select(m => m * 100).ToArray(), "single assets",
                     Color.Parse("#8A94A3"), sizes: [50]);
        for (var i = 0; i < 5; i++) left.Text(Math.Sqrt(sigma[i, i]) * 100 + 0.6, mu[i] * 100, assets[i], fontSize: 10, baseline: TextBaseline.Middle);
        left.Scatter([Risk(equal) * 100], [Return(equal) * 100], "equal weights", Color.Parse("#E8A317"), sizes: [80], marker: MarkerStyle.Square);
        left.Scatter([Risk(tangency) * 100], [Return(tangency) * 100], "best Sharpe ratio", Color.Parse("#D9363E"), sizes: [140], marker: MarkerStyle.Star);
        left.SetXLim(0, Enumerable.Range(0, 5).Max(i => Math.Sqrt(sigma[i, i])) * 100 * 1.2);
        left.SetTitle("Risk and return, annualised").SetXLabel("risk: volatility (%)").SetYLabel("expected return (%)");
        left.Legend(LegendLocation.LowerRight);

        var x = Enumerable.Range(0, 5).Select(i => (double)i).ToArray();
        right.Bar(x.Select(v => v - 0.2).ToArray(), equal.Select(w => w * 100).ToArray(), 0.4, label: "equal weights", color: Color.Parse("#E8A317"));
        right.Bar(x.Select(v => v + 0.2).ToArray(), tangency.Select(w => w * 100).ToArray(), 0.4, label: "best Sharpe ratio", color: Color.Parse("#D9363E"));
        right.SetXTicks(x, assets).SetTitle("Portfolio weights").SetYLabel("% of the fund");
        right.Legend(LegendLocation.UpperRight);
        output.Chart(figure);

        double Sharpe(double[] w) => (Return(w) - riskFree) / Risk(w);
        output.Metric("Best-Sharpe portfolio", $"{Return(tangency) * 100:F1}% return, {Risk(tangency) * 100:F1}% risk", $"Sharpe {Sharpe(tangency):F2}");
        output.Metric("Equal weights", $"{Return(equal) * 100:F1}% return, {Risk(equal) * 100:F1}% risk", $"Sharpe {Sharpe(equal):F2}");
        var sameReturn = MinRisk(Return(equal));
        output.Metric("Same return, less risk", $"{Risk(sameReturn) * 100:F1}% risk", $"the frontier portfolio earning {Return(equal) * 100:F1}%");
        output.Line($"For the equal-weight portfolio's return, the frontier mix carries {(1 - Risk(sameReturn) / Risk(equal)) * 100:F0}% less risk; " +
                    $"the best-Sharpe mix holds the most {assets[Array.IndexOf(tangency, tangency.Max())]} ({tangency.Max() * 100:F0}%) and the least " +
                    $"{assets[Array.IndexOf(tangency, tangency.Min())]} ({tangency.Min() * 100:F0}%).");
    }
}
