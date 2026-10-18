using Gravicode.Science.SciNet;
using Matplotlib.Net.Artists;
using Matplotlib.Net.Rendering;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;

namespace ScienceGallery.Showcases.Games;

/// <summary>Exact bounce times with ODE events, against the frame-by-frame physics most games use.</summary>
public sealed class BouncingBallPhysics : Showcase
{
    public override string Title => "Does the game's bouncing ball lose the right amount of energy?";
    public override Domain Domain => Domain.Games;
    public override GraviLibrary[] Libraries => [GraviLibrary.SciNet, GraviLibrary.GraviNum];
    public override string Summary => "Integrate between bounces with terminal events, then compare with a 60 fps frame-stepped engine.";
    public override string Scenario =>
        "A physics programmer tunes a ball with a coefficient of restitution of 0.8: each bounce should keep " +
        "64% of the height. Game engines step at a fixed 60 frames per second and only notice the floor after " +
        "the ball is already inside it, so contact happens a little late and at the wrong speed. Integrating " +
        "with an ODE solver that stops exactly at each impact (a terminal event) gives the reference to tune " +
        "against.";

    public override void Run(ShowcaseOutput output)
    {
        const double g = 9.81, restitution = 0.8, drop = 2.0, drag = 0.02;   // m/s², -, m, 1/m (light air drag)

        // Flight between bounces: state (x, y, vx, vy) with gravity and quadratic drag.
        void Flight(double t, ReadOnlySpan<double> s, Span<double> ds)
        {
            var speed = Math.Sqrt(s[2] * s[2] + s[3] * s[3]);
            ds[0] = s[2]; ds[1] = s[3];
            ds[2] = -drag * speed * s[2];
            ds[3] = -g - drag * speed * s[3];
        }
        var floor = new OdeEvent((t, s) => s[1], terminal: 1, direction: -1);   // stop exactly when y reaches 0, falling

        // Exact: integrate to each impact, reverse and shrink the vertical speed, continue.
        var (xs, ys, apexes, impacts, impactX) = (new List<double>(), new List<double>(), new List<double>(), new List<double>(), new List<double>());
        double[] state = [0, drop, 1.2, 0];
        var clock = 0.0;
        for (var bounce = 0; bounce < 12; bounce++)
        {
            var leg = Integrate.SolveIvp(Flight, clock, clock + 10, state, OdeMethod.Dop853, events: [floor], rtol: 1e-10, atol: 1e-12, maxStep: 0.01);
            xs.AddRange(leg.Y.Row(0).ToArray()); ys.AddRange(leg.Y.Row(1).ToArray());
            apexes.Add(leg.Y.Row(1).ToArray().Max());
            clock = leg.TEvents![0][0];
            impacts.Add(clock);
            var hit = leg.YEvents![0][0];
            impactX.Add(hit[0]);
            state = [hit[0], 0, hit[2], -restitution * hit[3]];
        }

        // The game engine: semi-implicit Euler at 60 fps; when the ball is found below the floor,
        // it is put back on the floor and its vertical speed reversed.
        const double dt = 1 / 60.0;
        var engineApexes = new List<double>();
        double ex = 0, ey = drop, evx = 1.2, evy = 0, peak = drop;
        for (var frame = 0; frame < 60 * 30 && engineApexes.Count < 12; frame++)
        {
            var speed = Math.Sqrt(evx * evx + evy * evy);
            evx += -drag * speed * evx * dt;
            evy += (-g - drag * speed * evy) * dt;
            ex += evx * dt; ey += evy * dt;
            peak = Math.Max(peak, ey);
            if (ey < 0) { engineApexes.Add(peak); ey = 0; evy = -restitution * evy; peak = 0; }
        }

        // Without drag each apex is restitution² of the last; drag takes a little more.
        var ratio = Math.Exp(Stats.Linregress(Enumerable.Range(0, apexes.Count).Select(i => (double)i).ToArray(), apexes.Select(h => Math.Log(h)).ToArray()).Slope);
        var engineRatio = Math.Exp(Stats.Linregress(Enumerable.Range(0, engineApexes.Count).Select(i => (double)i).ToArray(), engineApexes.Select(h => Math.Log(h)).ToArray()).Slope);

        var figure = GalleryCharts.Figure(11, 4.6);
        var (left, right) = (figure.AddSubplot(1, 2, 1), figure.AddSubplot(1, 2, 2));
        left.Plot(xs.ToArray(), ys.ToArray(), "ball (exact events)", lineWidth: 1.4);
        left.Scatter(impactX.ToArray(), new double[impactX.Count], "impact, located by the event", Color.Parse("#D9363E"), sizes: [30]);
        left.Legend(LegendLocation.UpperRight);
        left.SetTitle("Trajectory").SetXLabel("distance (m)").SetYLabel("height (m)").SetYLim(0, drop * 1.08);

        var n = Enumerable.Range(1, apexes.Count).Select(i => (double)i).ToArray();
        right.Plot(n, apexes.ToArray(), "exact (ODE events)", marker: MarkerStyle.Circle);
        right.Plot(n.Take(engineApexes.Count).ToArray(), engineApexes.ToArray(), "60 fps engine", marker: MarkerStyle.Square, style: LineStyle.Dashed);
        right.Plot(n, n.Select(i => drop * Math.Pow(restitution * restitution, i - 1)).ToArray(), "e² rule, no drag", Color.Parse("#8A94A3"), style: LineStyle.Dotted);
        right.SetYScale("log").SetTitle("Height of each bounce").SetXLabel("bounce").SetYLabel("apex height (m)");
        right.Legend(LegendLocation.UpperRight);
        output.Chart(figure);

        output.Metric("Height kept per bounce", $"{ratio * 100:F1}%", $"e² = {restitution * restitution * 100:F0}% without drag");
        output.Metric("60 fps engine keeps", $"{engineRatio * 100:F1}%", "per bounce, same ball");
        output.Metric("Time to 12th bounce", $"{impacts[^1]:F2} s", $"first impact at {impacts[0]:F3} s");
        output.Line($"After 12 bounces the engine's ball is at {engineApexes[^1] * 100:F1} cm against {apexes[^1] * 100:F1} cm for the exact one: " +
                    "small per-bounce timing errors compound, which is why engines detect the contact time within the frame.");
    }
}
