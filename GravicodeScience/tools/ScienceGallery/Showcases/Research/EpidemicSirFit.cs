using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Matplotlib.Net.Artists;
using Matplotlib.Net.Rendering;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;

namespace ScienceGallery.Showcases.Research;

/// <summary>Fitting an SIR epidemic model to the first weeks of an outbreak to forecast its peak.</summary>
public sealed class EpidemicSirFit : Showcase
{
    public override string Title => "When will the outbreak peak, and how big will it be?";
    public override Domain Domain => Domain.Research;
    public override GraviLibrary[] Libraries => [GraviLibrary.SciNet, GraviLibrary.GraviNum];
    public override string Summary => "Fit the SIR model's transmission and recovery rates to early case counts, then forecast.";
    public override string Scenario =>
        "Six weeks into an influenza outbreak in a town of 60,000, the district health office has daily case " +
        "counts and needs to know when the hospital will be busiest. The SIR model splits the town into " +
        "susceptible, infected and recovered people. Least squares finds the transmission and recovery rates that " +
        "reproduce the counts so far; integrating forward gives the expected peak date and the basic reproduction " +
        "number R₀. The rest of the season, revealed afterwards, shows how good the forecast was.";

    public override void Run(ShowcaseOutput output)
    {
        const double population = 60_000;
        const int seasonDays = 120, knownDays = 42;

        // dS/dt = -β S I / N,  dI/dt = β S I / N - γ I,  dR/dt = γ I. Cumulative infections = N - S,
        // so the new cases on day d are S(d-1) - S(d).
        double[] DailyCases(double beta, double gamma, double initialInfected, int days)
        {
            var sol = Integrate.SolveIvp((t, u, du) =>
            {
                var infection = beta * u[0] * u[1] / population;
                du[0] = -infection;
                du[1] = infection - gamma * u[1];
            }, 0, days, [population - initialInfected, initialInfected],
                rtol: 1e-8, atol: 1e-6, tEval: Enumerable.Range(0, days + 1).Select(d => (double)d).ToArray());
            return Enumerable.Range(1, days).Select(d => sol.Y[0, d - 1] - sol.Y[0, d]).ToArray();
        }

        // The season as it really unfolded (β = 0.38/day, recovery after ~4 days), with day-to-day
        // reporting noise. Only the first six weeks are known when the forecast is made.
        var rng = new GraviRandom(42);
        var observed = DailyCases(0.38, 0.25, 12, seasonDays).Select(c => (double)rng.Poisson(c)).ToArray();
        var known = observed.Take(knownDays).ToArray();

        // Before the peak, case counts only reveal the growth rate β − γ: a fast-spreading, short illness
        // and a slow, long one grow alike. So, as epidemiologists do, the infectious period comes from
        // clinical studies (about four days for influenza) and the fit estimates β and the starting size.
        const double gamma = 1 / 4.0;
        // Residuals on the square-root scale: Poisson counts have variance equal to their mean, and the
        // square root makes the noise roughly the same size on quiet and busy days.
        var fit = Optimize.LeastSquares(p =>
        {
            var model = DailyCases(p[0], gamma, p[1], knownDays);
            return model.Select((m, d) => Math.Sqrt(Math.Max(m, 0)) - Math.Sqrt(known[d])).ToArray();
        }, [0.5, 5], bounds: new Bounds([0.01, 1], [3, 500]));
        var (beta, start) = (fit.X[0], fit.X[1]);
        // The uncertainty of β from the Jacobian at the solution: cov ≈ s²(JᵀJ)⁻¹.
        var jtj = LinAlg.Dot(fit.Jac.T, fit.Jac);
        var betaError = Math.Sqrt(2 * fit.Cost / (knownDays - 2) * LinAlg.Inverse(jtj)[0, 0]);
        var forecast = DailyCases(beta, gamma, start, seasonDays);

        var forecastPeak = Array.IndexOf(forecast, forecast.Max()) + 1;
        var actualSmooth = Signal.SavgolFilter(observed, 9, 2);              // smooth the noise to read the real peak
        var actualPeak = Array.IndexOf(actualSmooth, actualSmooth.Max()) + 1;
        var attack = forecast.Sum() / population;

        var days = Enumerable.Range(1, seasonDays).Select(d => (double)d).ToArray();
        var figure = GalleryCharts.Figure(9, 4.6);
        var ax = figure.AddSubplot();
        ax.AxVSpan(0.5, knownDays + 0.5, Color.Parse("#E4E8ED"), 0.6, "known when forecasting");
        ax.Bar(days.Take(knownDays).ToArray(), known, 0.8, label: "reported cases", color: GalleryCharts.Of(GraviLibrary.GraviText));
        ax.Scatter(days.Skip(knownDays).ToArray(), observed.Skip(knownDays).ToArray(), "cases reported later", color: Color.Parse("#8A94A3"), sizes: [14]);
        ax.Plot(days, forecast, "SIR forecast", Color.Parse("#D9363E"));
        ax.AxVLine(forecastPeak, Color.Parse("#D9363E"), LineStyle.Dotted, 1.4, $"forecast peak, day {forecastPeak}");
        ax.SetTitle($"Influenza season, town of {population:N0}").SetXLabel("day of outbreak").SetYLabel("new cases per day");
        ax.Legend(LegendLocation.UpperRight);
        output.Chart(figure);

        output.Metric("R₀ = β / γ", $"{beta / gamma:F2} ± {betaError / gamma:F2}", "each case infects this many in a fully susceptible town");
        output.Metric("Transmission rate β", $"{beta:F3} ± {betaError:F3} per day", $"infectious period fixed at {1 / gamma:F0} days");
        output.Metric("Forecast peak", $"day {forecastPeak}", $"{forecast.Max():N0} new cases per day; actual peak around day {actualPeak}");
        output.Metric("Share of town infected", $"{attack * 100:F0}%", "over the whole season, forecast");
        output.Line($"Fitted on the first {knownDays} days, {actualPeak - knownDays} days before the real peak, the model placed the peak " +
                    $"within {Math.Abs(forecastPeak - actualPeak)} day(s) of it: lead time the hospital can use to open extra beds.");
    }
}
