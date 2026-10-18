using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviProb;
using Matplotlib.Net.Artists;
using Matplotlib.Net.Rendering;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;

namespace ScienceGallery.Showcases.Industry;

/// <summary>Forecasting a building's electricity demand for the next two days, with an honest uncertainty band.</summary>
public sealed class EnergyLoadForecast : Showcase
{
    public override string Title => "How much power will the building draw in the next 48 hours?";
    public override Domain Domain => Domain.Industry;
    public override GraviLibrary[] Libraries => [GraviLibrary.GraviProb, GraviLibrary.GraviFrame, GraviLibrary.GraviNum];
    public override string Summary => "A Gaussian process with a weekly periodic kernel forecasts hourly load with a 95% band.";
    public override string Scenario =>
        "A facilities manager buys electricity a day ahead and pays a penalty when the building's demand exceeds " +
        "the contracted peak. Three weeks of smart-meter readings show the weekday office pattern, quieter " +
        "weekends and a slow drift with the weather. A Gaussian process with a weekly periodic kernel plus a " +
        "smooth trend learns all three and forecasts the next two days with an uncertainty band, so the " +
        "contracted peak can be set with a known margin.";

    public override void Run(ShowcaseOutput output)
    {
        // Three weeks of hourly smart-meter readings starting on a Monday, plus the two days that follow
        // (kept back to score the forecast). Load in kW.
        const int history = 21 * 24, horizon = 48;
        var rng = new GraviRandom(6);
        var drift = 0.0;
        var load = new double[history + horizon];
        for (var h = 0; h < load.Length; h++)
        {
            var (day, hour) = (h / 24, h % 24);
            var weekend = day % 7 >= 5;
            var office = Math.Exp(-Math.Pow((hour - 13) / 4.2, 6));                 // 08:00–18:00 plateau
            drift += rng.Normal(0, 1.5) - 0.01 * drift;                             // weather, slowly wandering
            load[h] = 340 + (weekend ? 70 : 320) * office + drift + rng.Normal(0, 14);
        }
        var meter = DataFrame.FromColumns(new Dictionary<string, double[]>
        {
            ["hour"] = Enumerable.Range(0, history).Select(h => (double)h).ToArray(),
            ["day"] = Enumerable.Range(0, history).Select(h => (double)(h / 24)).ToArray(),
            ["hour_of_day"] = Enumerable.Range(0, history).Select(h => (double)(h % 24)).ToArray(),
            ["weekend"] = Enumerable.Range(0, history).Select(h => h / 24 % 7 >= 5 ? 1.0 : 0.0).ToArray(),
            ["load_kw"] = load.Take(history).ToArray(),
        });

        // What the meter data already says, before any model: daily peaks and the typical weekday shape.
        var dailyPeak = meter.GroupBy("day").Max("load_kw").Numeric("load_kw").Values;
        var weekdayProfile = meter.FilterBy("weekend", w => w == 0).GroupBy("hour_of_day").Mean("load_kw").Numeric("load_kw").Values;
        var busiestHour = Array.IndexOf(weekdayProfile, weekdayProfile.Max());
        var smoothed = meter.Numeric("load_kw").Rolling(window: 24).Mean().Values;

        // The Gaussian process: a weekly periodic pattern (period 168 h; a short length scale lets it
        // carry the daily shape within the week) plus a slow trend, plus measurement noise (kW²).
        var kernel = new SumKernel(new PeriodicKernel(period: 168, lengthScale: 0.12, variance: 160 * 160),
                                   new RbfKernel(lengthScale: 24, variance: 40 * 40));
        var gp = new GaussianProcess(kernel, noise: 16 * 16).Fit(meter.ToNdArray("hour"), meter.Numeric("load_kw").ToNdArray());
        var future = Enumerable.Range(history, horizon).Select(h => (double)h).ToArray();
        var prediction = gp.Predict(new NdArray(future, horizon, 1));
        var (lower, upper) = prediction.Interval(0.95);
        var forecast = prediction.Mean.ToArray();
        var actual = load.Skip(history).ToArray();
        var mape = actual.Zip(forecast, (a, f) => Math.Abs(a - f) / a).Average() * 100;
        var covered = actual.Where((a, i) => a >= lower.At(i) && a <= upper.At(i)).Count() * 100.0 / horizon;

        var figure = GalleryCharts.Figure(11, 4.8);
        var ax = figure.AddSubplot();
        var shownFrom = history - 7 * 24;
        var past = Enumerable.Range(shownFrom, history - shownFrom).Select(h => h / 24.0).ToArray();
        var ahead = future.Select(h => h / 24.0).ToArray();
        ax.Plot(past, load.Skip(shownFrom).Take(history - shownFrom).ToArray(), "metered", Color.Parse("#5F6B7A"), lineWidth: 1.2);
        ax.Plot(past, smoothed.Skip(shownFrom).ToArray(), "24-hour rolling mean", Color.Parse("#E8A317"), lineWidth: 1.6);
        ax.FillBetween(ahead, lower.ToArray(), upper.ToArray(), "95% band", GalleryCharts.Of(GraviLibrary.GraviProb), alpha: 0.2);
        ax.Plot(ahead, forecast, "forecast", GalleryCharts.Of(GraviLibrary.GraviProb));
        ax.Plot(ahead, actual, "what happened", Color.Parse(GalleryCharts.Ink), lineWidth: 1.0, style: LineStyle.Dotted);
        ax.AxVLine(history / 24.0, Color.Parse("#8A94A3"), LineStyle.Solid, 1.0);
        ax.SetTitle("Building electricity demand, last week and the next two days").SetXLabel("day").SetYLabel("load (kW)");
        ax.Legend(LegendLocation.OutsideRight);
        output.Chart(figure);

        var peakForecast = upper.ToArray().Max();
        output.Metric("Typical weekday peak", $"{dailyPeak.Where((_, d) => d % 7 < 5).Average():F0} kW", $"busiest hour {busiestHour:00}:00");
        output.Metric("Forecast error (MAPE)", $"{mape:F1}%", "next 48 hours, scored after the fact");
        output.Metric("Band coverage", $"{covered:F0}%", "hours inside the 95% band; close to 95 means honest uncertainty");
        output.Metric("Peak to contract", $"{peakForecast:F0} kW", "top of the 95% band over the next two days");
        output.Line($"Contracting {peakForecast:F0} kW covers the forecast peak of {forecast.Max():F0} kW with a {peakForecast - forecast.Max():F0} kW margin " +
                    $"drawn from the model's own uncertainty; the real peak was {actual.Max():F0} kW.");
    }
}
