using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Matplotlib.Net.Rendering;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;

namespace ScienceGallery.Showcases.Office;

/// <summary>A week's rota as a mixed-integer program: cover every shift, respect rest rules and leave, honour preferences.</summary>
public sealed class ShiftRostering : Showcase
{
    public override string Title => "Build next week's rota that covers every shift and honours leave";
    public override Domain Domain => Domain.Office;
    public override GraviLibrary[] Libraries => [GraviLibrary.SciNet, GraviLibrary.GraviNum];
    public override string Summary => "Mixed-integer linear programming schedules eight staff across day and night shifts.";
    public override string Scenario =>
        "A clinic manager spends Friday afternoon on next week's rota: two or three people on each day shift, one " +
        "or two at night, nobody on a day shift straight after a night, at most five shifts each, and approved " +
        "leave respected. Each person also says how much they dislike nights and weekends. Written as a " +
        "mixed-integer program, the solver finds the rota with the least total dislike that satisfies every rule.";

    public override void Run(ShowcaseOutput output)
    {
        string[] staff = ["Ani", "Budi", "Clara", "Dedi", "Eva", "Fikri", "Gina", "Hadi"];
        string[] days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
        const int dayShift = 0, nightShift = 1;
        int[] needDay = [3, 3, 3, 3, 3, 2, 2], needNight = [2, 2, 2, 2, 2, 1, 1];
        double[] nightDislike = [1, 3, 0.5, 2, 3, 0.5, 2, 1];       // per night shift
        double[] weekendDislike = [2, 0.5, 1, 3, 0.5, 2, 1, 1];     // per weekend shift
        var leave = new HashSet<(int Person, int Day)> { (0, 2), (0, 3), (3, 5), (3, 6), (6, 0) };

        // One binary variable per person, day and shift: x = 1 means "works it".
        int S = staff.Length, D = days.Length;
        int Var(int s, int d, int k) => (s * D + d) * 2 + k;
        var n = S * D * 2;
        var cost = new double[n];
        var upper = Enumerable.Repeat(1.0, n).ToArray();
        for (var s = 0; s < S; s++)
            for (var d = 0; d < D; d++)
                for (var k = 0; k < 2; k++)
                {
                    cost[Var(s, d, k)] = 0.01 + (k == nightShift ? nightDislike[s] : 0) + (d >= 5 ? weekendDislike[s] : 0);
                    if (leave.Contains((s, d))) upper[Var(s, d, k)] = 0;       // leave is a bound, not a penalty
                }

        // Each constraint is one row: lower ≤ (coefficients · x) ≤ upper.
        var rows = new List<(double[] A, double Lb, double Ub)>();
        void Row(IEnumerable<int> vars, double lb, double ub)
        {
            var a = new double[n];
            foreach (var v in vars) a[v] += 1;
            rows.Add((a, lb, ub));
        }
        for (var d = 0; d < D; d++)
        {
            Row(Enumerable.Range(0, S).Select(s => Var(s, d, dayShift)), needDay[d], needDay[d] + 1);     // cover the day
            Row(Enumerable.Range(0, S).Select(s => Var(s, d, nightShift)), needNight[d], needNight[d]); // cover the night
        }
        for (var s = 0; s < S; s++)
        {
            for (var d = 0; d < D; d++) Row([Var(s, d, dayShift), Var(s, d, nightShift)], 0, 1);           // one shift a day
            for (var d = 0; d + 1 < D; d++) Row([Var(s, d, nightShift), Var(s, d + 1, dayShift)], 0, 1);   // rest after nights
            Row(Enumerable.Range(0, D).SelectMany(d => new[] { Var(s, d, 0), Var(s, d, 1) }), 3, 5);      // 3–5 shifts a week
        }
        var a = new NdArray(rows.SelectMany(r => r.A).ToArray(), rows.Count, n);
        var constraint = new LinearConstraint(a, rows.Select(r => r.Lb).ToArray(), rows.Select(r => r.Ub).ToArray());

        var result = Optimize.Milp(cost, Enumerable.Repeat(1, n).ToArray(), new Bounds(new double[n], upper), [constraint],
                                   new MilpOptions { TimeLimit = 1.5 });
        var x = result.X ?? throw new InvalidOperationException("No feasible rota: " + result.Message);
        bool Works(int s, int d, int k) => x[Var(s, d, k)] > 0.5;

        // Draw the rota: one cell per person and day.
        var grid = new double[S, D];
        for (var s = 0; s < S; s++)
            for (var d = 0; d < D; d++)
                grid[s, d] = Works(s, d, dayShift) ? 1 : Works(s, d, nightShift) ? 2 : 0;
        var figure = GalleryCharts.Figure(8.5, 5.6);
        var ax = figure.AddSubplot();
        ax.Heatmap(grid, staff, days, colormap: "Blues", gridColor: Color.Parse("#FFFFFF"));
        for (var s = 0; s < S; s++)
            for (var d = 0; d < D; d++)
            {
                var label = leave.Contains((s, d)) ? "leave" : grid[s, d] switch { 1 => "day", 2 => "night", _ => "" };
                ax.Text(d, s, label, grid[s, d] == 2 ? Color.Parse("#FFFFFF") : Color.Parse(GalleryCharts.Ink),
                        fontSize: 11, align: TextAlign.Center, baseline: TextBaseline.Middle);
            }
        ax.SetTitle("Next week's rota (white: off)");
        output.Chart(figure);

        var shifts = Enumerable.Range(0, S).Select(s => Enumerable.Range(0, D).Count(d => grid[s, d] > 0)).ToArray();
        var nights = Enumerable.Range(0, S).Select(s => Enumerable.Range(0, D).Count(d => grid[s, d] == 2)).ToArray();
        output.Metric("Shifts covered", $"{shifts.Sum()}", $"{needDay.Sum() + needNight.Sum()} required, every rule satisfied");
        output.Metric("Total dislike", (result.Fun ?? 0).ToString("F1"), result.Status == 0 ? "proven optimal" : $"best found ({result.Message})");
        output.Metric("Branch-and-bound nodes", result.MipNodeCount.ToString("N0"), $"{n} yes/no decisions, {rows.Count} rules");
        var nightOwl = Enumerable.Range(0, S).MaxBy(s => nights[s]);
        output.Line($"Nights go mostly to those who mind them least: {staff[nightOwl]} (dislike {nightDislike[nightOwl]}) works {nights[nightOwl]}, " +
                    $"while {staff[Array.IndexOf(nightDislike, nightDislike.Max())]} (dislike {nightDislike.Max()}) works " +
                    $"{nights[Array.IndexOf(nightDislike, nightDislike.Max())]}. Everyone works {shifts.Min()}–{shifts.Max()} shifts.");
    }
}
