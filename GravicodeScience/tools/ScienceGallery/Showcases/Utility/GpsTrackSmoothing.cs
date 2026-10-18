using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviProb;
using Gravicode.Science.SciNet;
using Matplotlib.Net.Artists;
using Matplotlib.Net.Rendering;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;

namespace ScienceGallery.Showcases.Utility;

/// <summary>Smoothing a jittery GPS track with a Kalman smoother, and why raw GPS overstates distance.</summary>
public sealed class GpsTrackSmoothing : Showcase
{
    public override string Title => "Why does my running app say I ran further than I did?";
    public override Domain Domain => Domain.Utility;
    public override GraviLibrary[] Libraries => [GraviLibrary.GraviProb, GraviLibrary.SciNet, GraviLibrary.GraviNum];
    public override string Summary => "A constant-velocity Kalman smoother cleans a 1 Hz GPS track; summing raw fixes inflates the distance.";
    public override string Scenario =>
        "A phone logs a GPS fix every second on a 15-minute park run, each fix off by a few metres in a direction " +
        "that wanders from second to second. Joining the raw fixes zig-zags around the real path, and every zig-zag adds distance, so the " +
        "app overstates the run. A Kalman filter with a constant-velocity model, run forwards and then smoothed " +
        "backwards, weighs each fix against where the runner should be; a Savitzky–Golay filter is the quick " +
        "alternative for comparison.";

    public override void Run(ShowcaseOutput output)
    {
        // The true route: a loop through a park, run at about 3.2 m/s, one point per second (metres).
        const int seconds = 900;
        const double gpsError = 4.5;                              // metres, per axis
        var rng = new GraviRandom(16);
        var truth = new double[seconds, 2];
        double heading = 0, x = 0, y = 0;
        for (var s = 0; s < seconds; s++)
        {
            heading += 2 * Math.PI / seconds + 0.012 * Math.Sin(s / 40.0);   // one loop, with some wiggle
            var speed = 3.2 + 0.4 * Math.Sin(s / 90.0);
            x += speed * Math.Cos(heading); y += speed * Math.Sin(heading);
            (truth[s, 0], truth[s, 1]) = (x, y);
        }
        // GPS error wanders rather than jumping: each second keeps 90% of the last second's error
        // (satellite geometry and multipath change slowly), scaled so its spread stays at gpsError.
        var fixes = new double[seconds, 2];
        double ex = 0, ey = 0, keep = 0.9, fresh = gpsError * Math.Sqrt(1 - 0.9 * 0.9);
        for (var s = 0; s < seconds; s++)
        {
            (ex, ey) = (keep * ex + rng.Normal(0, fresh), keep * ey + rng.Normal(0, fresh));
            (fixes[s, 0], fixes[s, 1]) = (truth[s, 0] + ex, truth[s, 1] + ey);
        }

        // State (x, y, vx, vy): position moves by velocity each second; velocity drifts slowly.
        // The GPS observes position only.
        var transition = NdArray.FromArray(new double[,] { { 1, 0, 1, 0 }, { 0, 1, 0, 1 }, { 0, 0, 1, 0 }, { 0, 0, 0, 1 } });
        var observe = NdArray.FromArray(new double[,] { { 1, 0, 0, 0 }, { 0, 1, 0, 0 } });
        var accel = 0.15 * 0.15;                                   // (m/s²)²: how hard a runner changes pace or direction
        var processNoise = NdArray.FromArray(new double[,]
        {
            { accel / 4, 0, accel / 2, 0 }, { 0, accel / 4, 0, accel / 2 },
            { accel / 2, 0, accel, 0 }, { 0, accel / 2, 0, accel },
        });
        var gpsNoise = NdArray.FromArray(new double[,] { { gpsError * gpsError, 0 }, { 0, gpsError * gpsError } });
        var kalman = new KalmanFilter(transition, observe, processNoise, gpsNoise);
        var smoothed = kalman.Smooth(NdArray.FromArray(fixes));
        var kx = smoothed.Select(e => e.Mean.At(0)).ToArray();
        var ky = smoothed.Select(e => e.Mean.At(1)).ToArray();

        // The quick alternative: a Savitzky–Golay filter over 15 seconds on each coordinate.
        var raw = (X: Enumerable.Range(0, seconds).Select(s => fixes[s, 0]).ToArray(), Y: Enumerable.Range(0, seconds).Select(s => fixes[s, 1]).ToArray());
        var sx = Signal.SavgolFilter(raw.X, 15, 2);
        var sy = Signal.SavgolFilter(raw.Y, 15, 2);

        double Length(double[] px, double[] py) => Enumerable.Range(1, px.Length - 1).Sum(i => Math.Sqrt(Math.Pow(px[i] - px[i - 1], 2) + Math.Pow(py[i] - py[i - 1], 2)));
        double Rmse(double[] px, double[] py) => Math.Sqrt(Enumerable.Range(0, seconds).Average(i => Math.Pow(px[i] - truth[i, 0], 2) + Math.Pow(py[i] - truth[i, 1], 2)));
        var trueX = Enumerable.Range(0, seconds).Select(s => truth[s, 0]).ToArray();
        var trueY = Enumerable.Range(0, seconds).Select(s => truth[s, 1]).ToArray();
        var actual = Length(trueX, trueY);

        var figure = GalleryCharts.Figure(10, 5.6);
        var (left, right) = (figure.AddSubplot(1, 2, 1), figure.AddSubplot(1, 2, 2));
        left.Scatter(raw.X, raw.Y, "GPS fixes", Color.Parse("#8A94A3"), sizes: [6], alpha: 0.6);
        left.Plot(trueX, trueY, "true route", Color.Parse(GalleryCharts.Ink), lineWidth: 1.2, style: LineStyle.Dashed);
        left.Plot(kx, ky, "Kalman smoother", GalleryCharts.Of(GraviLibrary.GraviProb), lineWidth: 1.6);
        left.SetAspect(1.0).SetTitle("The whole run").SetXLabel("east (m)").SetYLabel("north (m)");
        left.Legend(LegendLocation.Center);

        // Zoom into one minute to see the zig-zag.
        var minute = Enumerable.Range(300, 60).ToArray();
        right.Plot(minute.Select(i => raw.X[i]).ToArray(), minute.Select(i => raw.Y[i]).ToArray(), "raw fixes joined", Color.Parse("#8A94A3"), lineWidth: 1.0, marker: MarkerStyle.Circle, markerSize: 3);
        right.Plot(minute.Select(i => trueX[i]).ToArray(), minute.Select(i => trueY[i]).ToArray(), "true route", Color.Parse(GalleryCharts.Ink), lineWidth: 1.2, style: LineStyle.Dashed);
        right.Plot(minute.Select(i => sx[i]).ToArray(), minute.Select(i => sy[i]).ToArray(), "Savitzky–Golay", Color.Parse("#E8A317"), lineWidth: 1.4);
        right.Plot(minute.Select(i => kx[i]).ToArray(), minute.Select(i => ky[i]).ToArray(), "Kalman smoother", GalleryCharts.Of(GraviLibrary.GraviProb), lineWidth: 1.8);
        right.SetTitle("One minute, close up").SetXLabel("east (m)").SetYLabel("north (m)");
        right.Legend(LegendLocation.Best);
        output.Chart(figure);

        output.Metric("True distance", $"{actual / 1000:F2} km");
        output.Metric("Raw GPS distance", $"{Length(raw.X, raw.Y) / 1000:F2} km", $"{(Length(raw.X, raw.Y) / actual - 1) * 100:+0;-0}%: every zig-zag adds length");
        output.Metric("Kalman distance", $"{Length(kx, ky) / 1000:F2} km", $"{(Length(kx, ky) / actual - 1) * 100:+0.0;-0.0}%; position error {Rmse(kx, ky):F1} m");
        output.Metric("Savitzky–Golay distance", $"{Length(sx, sy) / 1000:F2} km", $"{(Length(sx, sy) / actual - 1) * 100:+0.0;-0.0}%; position error {Rmse(sx, sy):F1} m");
        output.Line($"Raw fixes are {Rmse(raw.X, raw.Y):F1} m off on average; the smoother brings that to {Rmse(kx, ky):F1} m because it knows a runner " +
                    "cannot jump sideways in a second. That knowledge, not averaging alone, is what removes the phantom distance.");
    }
}
