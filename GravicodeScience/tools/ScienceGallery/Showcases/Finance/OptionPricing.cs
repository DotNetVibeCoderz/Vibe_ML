using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Matplotlib.Net.Artists;
using Matplotlib.Net.Rendering;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;

namespace ScienceGallery.Showcases.Finance;

/// <summary>Pricing a European call by the Black–Scholes formula and checking it by Monte Carlo simulation.</summary>
public sealed class OptionPricing : Showcase
{
    public override string Title => "What is this stock option worth today?";
    public override Domain Domain => Domain.Finance;
    public override GraviLibrary[] Libraries => [GraviLibrary.SciNet, GraviLibrary.GraviNum];
    public override string Summary => "Black–Scholes with the normal CDF, against a Monte Carlo simulation that converges to it.";
    public override string Scenario =>
        "A treasury desk is offered a six-month call option on a stock trading at Rp 9,500: the right to buy it " +
        "at Rp 10,000. Black–Scholes gives a closed-form price from the normal distribution function. Banks " +
        "also price by simulation, because simulation still works for contracts no formula covers; checking " +
        "it against the formula on a contract where both apply shows how many paths it needs.";

    public override void Run(ShowcaseOutput output)
    {
        const double spot = 9_500, strike = 10_000, years = 0.5, rate = 0.06, volatility = 0.32;

        // Black–Scholes: C = S N(d1) − K e^(−rT) N(d2).
        double Call(double s, double k)
        {
            var d1 = (Math.Log(s / k) + (rate + volatility * volatility / 2) * years) / (volatility * Math.Sqrt(years));
            var d2 = d1 - volatility * Math.Sqrt(years);
            return s * Special.Ndtr(d1) - k * Math.Exp(-rate * years) * Special.Ndtr(d2);
        }
        var price = Call(spot, strike);
        var delta = Special.Ndtr((Math.Log(spot / strike) + (rate + volatility * volatility / 2) * years) / (volatility * Math.Sqrt(years)));

        // Monte Carlo: simulate the price at expiry under the risk-neutral law, average the discounted payoff.
        // Antithetic pairs (z and −z) cancel much of the noise for free.
        var rng = new GraviRandom(1);
        var payoffs = new List<double>();
        var counts = new List<double>();
        var (estimates, errors) = (new List<double>(), new List<double>());
        var drift = (rate - volatility * volatility / 2) * years;
        var shock = volatility * Math.Sqrt(years);
        for (var checkpoint = 100; checkpoint <= 200_000; checkpoint = (int)(checkpoint * 1.5))
        {
            while (payoffs.Count < checkpoint)
            {
                var z = rng.Normal();
                var up = Math.Max(spot * Math.Exp(drift + shock * z) - strike, 0);
                var down = Math.Max(spot * Math.Exp(drift - shock * z) - strike, 0);
                payoffs.Add(Math.Exp(-rate * years) * (up + down) / 2);
            }
            var mean = payoffs.Average();
            counts.Add(payoffs.Count * 2);
            estimates.Add(mean);
            errors.Add(Math.Sqrt(payoffs.Select(p => (p - mean) * (p - mean)).Sum() / (payoffs.Count - 1) / payoffs.Count));
        }

        var strikes = NdArray.Linspace(7_000, 13_000, 60).ToArray();
        var figure = GalleryCharts.Figure(11, 4.8);
        var (left, right) = (figure.AddSubplot(1, 2, 1), figure.AddSubplot(1, 2, 2));
        left.FillBetween(counts.ToArray(), estimates.Zip(errors, (e, s) => e - 1.96 * s).ToArray(), estimates.Zip(errors, (e, s) => e + 1.96 * s).ToArray(),
                         "95% interval", GalleryCharts.Of(GraviLibrary.GraviNum), alpha: 0.18);
        left.Plot(counts.ToArray(), estimates.ToArray(), "Monte Carlo estimate", GalleryCharts.Of(GraviLibrary.GraviNum), marker: MarkerStyle.Circle, markerSize: 4);
        left.AxHLine(price, Color.Parse(GalleryCharts.Ink), LineStyle.Dashed, 1.2, $"Black–Scholes Rp {price:N0}");
        left.SetXScale("log").SetTitle("Simulation converging to the formula").SetXLabel("simulated price paths").SetYLabel("option price (Rp)");
        left.Legend(LegendLocation.UpperRight);

        right.Plot(strikes, strikes.Select(k => Call(spot, k)).ToArray(), "option value today");
        right.Plot(strikes, strikes.Select(k => Math.Max(spot - k, 0)).ToArray(), "value if exercised now", Color.Parse("#8A94A3"), style: LineStyle.Dashed);
        right.AxVLine(strike, Color.Parse("#D9363E"), LineStyle.Dotted, 1.2, "this contract's strike");
        right.SetTitle("Price across strikes (spot Rp 9,500)").SetXLabel("strike (Rp)").SetYLabel("call price (Rp)");
        right.Legend(LegendLocation.UpperRight);
        output.Chart(figure);

        output.Metric("Black–Scholes price", $"Rp {price:N0}", $"{price / spot * 100:F1}% of the share price");
        output.Metric("Monte Carlo price", $"Rp {estimates[^1]:N0} ± {1.96 * errors[^1]:N0}", $"{counts[^1]:N0} paths, 95% interval");
        output.Metric("Delta", delta.ToString("F3"), "shares to hold per option to hedge small moves");
        output.Line($"The simulation's error shrinks like 1/√n: {counts[^1]:N0} paths pin the price to within Rp {1.96 * errors[^1]:N0}, " +
                    $"and the formula sits {(Math.Abs(estimates[^1] - price) < 1.96 * errors[^1] ? "inside" : "outside")} that interval.");
    }
}
