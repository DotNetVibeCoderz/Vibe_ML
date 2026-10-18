using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Matplotlib.Net.Artists;
using Matplotlib.Net.Rendering;
using ScienceGallery.Models;
using ScienceGallery.Services;

namespace ScienceGallery.Showcases.Education;

/// <summary>A physics practical: fit a thrown ball's first half-second, then predict where it lands.</summary>
public sealed class ProjectileLab : Showcase
{
    public override string Title => "Where will the thrown ball land?";
    public override Domain Domain => Domain.Education;
    public override GraviLibrary[] Libraries => [GraviLibrary.SciNet, GraviLibrary.GraviNum];
    public override string Summary => "Fit a tennis ball's tracked positions, then predict its landing with and without air drag.";
    public override string Scenario =>
        "In a school physics practical, students film a tennis ball thrown across the field and track it frame by " +
        "frame for the first half-second. Fitting the points gives the launch velocity. The textbook parabola then " +
        "predicts the landing spot, and integrating the motion with quadratic air drag shows how far off the " +
        "no-drag answer is for a light, fuzzy ball.";

    public override void Run(ShowcaseOutput output)
    {
        const double g = 9.81;                                     // m/s²
        // Quadratic drag on a tennis ball: a = -k |v| v with k = ½ ρ Cd A / m.
        const double rho = 1.2, cd = 0.55, radius = 0.0335, mass = 0.057;
        var k = 0.5 * rho * cd * Math.PI * radius * radius / mass;  // per metre

        void Flight(double t, ReadOnlySpan<double> s, Span<double> ds, double drag)
        {
            // state = (x, y, vx, vy)
            var speed = Math.Sqrt(s[2] * s[2] + s[3] * s[3]);
            ds[0] = s[2];
            ds[1] = s[3];
            ds[2] = -drag * speed * s[2];
            ds[3] = -g - drag * speed * s[3];
        }

        // The "real" throw: 22 m/s at 38 degrees from shoulder height, with drag. The camera films at
        // 60 frames per second and the tracking is good to about a centimetre.
        double[] launch = [0, 1.6, 22 * Math.Cos(38 * Math.PI / 180), 22 * Math.Sin(38 * Math.PI / 180)];
        var frames = Enumerable.Range(0, 31).Select(i => i / 60.0).ToArray();
        var truth = Integrate.SolveIvp((t, s, ds) => Flight(t, s, ds, k), 0, 0.5, launch, rtol: 1e-9, atol: 1e-9, tEval: frames);
        var rng = new GraviRandom(7);
        var xMeasured = frames.Select((_, i) => truth.Y[0, i] + rng.Normal(0, 0.01)).ToArray();
        var yMeasured = frames.Select((_, i) => truth.Y[1, i] + rng.Normal(0, 0.01)).ToArray();

        // Over half a second each coordinate is close to a quadratic in time. The linear terms are the
        // launch velocity; the quadratic term in y is the apparent downward acceleration.
        var fitX = Optimize.CurveFit((t, p) => p[0] + p[1] * t + 0.5 * p[2] * t * t, frames, xMeasured, p0: [0, 10, 0]);
        var fitY = Optimize.CurveFit((t, p) => p[0] + p[1] * t + 0.5 * p[2] * t * t, frames, yMeasured, p0: [1, 10, -10]);
        double[] fittedStart = [fitX.Popt[0], fitY.Popt[0], fitX.Popt[1], fitY.Popt[1]];
        var speed0 = Math.Sqrt(fittedStart[2] * fittedStart[2] + fittedStart[3] * fittedStart[3]);
        var angle0 = Math.Atan2(fittedStart[3], fittedStart[2]) * 180 / Math.PI;

        // Predict the rest of the flight from the fitted launch, stopping when the ball reaches the ground.
        var ground = new OdeEvent((t, s) => s[1], terminal: 1, direction: -1);
        var vacuum = Integrate.SolveIvp((t, s, ds) => Flight(t, s, ds, 0), 0, 10, fittedStart, events: [ground], maxStep: 0.02);
        var withDrag = Integrate.SolveIvp((t, s, ds) => Flight(t, s, ds, k), 0, 10, fittedStart, events: [ground], maxStep: 0.02);
        var rangeVacuum = vacuum.YEvents![0][0][0];
        var rangeDrag = withDrag.YEvents![0][0][0];

        var figure = GalleryCharts.Figure(9, 4.6);
        var ax = figure.AddSubplot();
        ax.Plot(vacuum.Y.Row(0).ToArray(), vacuum.Y.Row(1).ToArray(), "textbook parabola (no drag)", style: LineStyle.Dashed);
        ax.Plot(withDrag.Y.Row(0).ToArray(), withDrag.Y.Row(1).ToArray(), "with air drag");
        ax.Scatter(xMeasured, yMeasured, "tracked frames (first 0.5 s)", sizes: [18]);
        ax.Scatter([rangeVacuum, rangeDrag], [0, 0], "predicted landing", marker: MarkerStyle.Triangle, sizes: [90]);
        ax.SetYLim(0, null);
        ax.SetTitle("Thrown tennis ball").SetXLabel("distance (m)").SetYLabel("height (m)");
        ax.Legend(LegendLocation.UpperRight);
        output.Chart(figure);

        output.Metric("Fitted launch speed", $"{speed0:F1} ± {Math.Sqrt(fitX.Perr[1] * fitX.Perr[1] + fitY.Perr[1] * fitY.Perr[1]):F1} m/s", $"at {angle0:F1}° above horizontal");
        output.Metric("Apparent downward acceleration", $"{-fitY.Popt[2]:F2} ± {fitY.Perr[2]:F2} m/s²", "more than g = 9.81: drag pulls against the rising ball");
        output.Metric("Landing, no drag", $"{rangeVacuum:F1} m");
        output.Metric("Landing, with drag", $"{rangeDrag:F1} m", $"{(1 - rangeDrag / rangeVacuum) * 100:F0}% shorter");
        output.Line($"Half a second of video pins the launch to within a few percent, but the textbook model still " +
                    $"overshoots by {rangeVacuum - rangeDrag:F1} m: for a ball this light, drag is not a small correction.");
    }
}
