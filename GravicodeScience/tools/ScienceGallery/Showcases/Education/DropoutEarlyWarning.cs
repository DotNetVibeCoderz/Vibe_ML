using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviLearn;
using Gravicode.Science.GraviLearn.ModelSelection;
using Gravicode.Science.GraviLearn.Trees;
using Gravicode.Science.GraviNum;
using Matplotlib.Net.Artists;
using Matplotlib.Net.Rendering;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;

namespace ScienceGallery.Showcases.Education;

/// <summary>An early-warning model that flags students at risk of dropping out, while there is time to help.</summary>
public sealed class DropoutEarlyWarning : Showcase
{
    public override string Title => "Which students need a counsellor this term?";
    public override Domain Domain => Domain.Education;
    public override GraviLibrary[] Libraries => [GraviLibrary.GraviLearn, GraviLibrary.GraviFrame, GraviLibrary.GraviNum];
    public override string Summary => "A random forest on attendance, grades and workload ranks students by dropout risk.";
    public override string Scenario =>
        "A vocational school loses about one student in five before graduation, and counsellors can only see so " +
        "many students a term. A model trained on past cohorts ranks the current students by risk. What matters " +
        "is recall in the top of the list: how many of the students who will leave the counsellors actually meet. " +
        "The feature importances tell the school which warning signs to watch.";

    public override void Run(ShowcaseOutput output)
    {
        // Four past cohorts, 1,200 students. Each column is something a school already records.
        var rng = new GraviRandom(31);
        const int n = 1200;
        var attendance = new double[n]; var gpa = new double[n]; var late = new double[n];
        var travel = new double[n]; var work = new double[n]; var dropped = new double[n];
        for (var i = 0; i < n; i++)
        {
            work[i] = rng.NextBernoulli(0.35) ? Math.Round(rng.Uniform(5, 30)) : 0;          // part-time job, h/week
            travel[i] = Math.Round(rng.Gamma(2, 6), 1);                                      // km to school
            attendance[i] = Math.Clamp(rng.Normal(92 - 0.4 * work[i] - 0.15 * travel[i], 5), 40, 100); // %
            gpa[i] = Math.Clamp(rng.Normal(1.2 + 0.022 * attendance[i], 0.45), 0, 4);
            late[i] = rng.Poisson(1 + 0.12 * work[i] + 0.05 * (100 - attendance[i]));       // late assignments
            // The (unknown to the model) process that makes a student leave.
            var logit = -2.4 - 0.16 * (attendance[i] - 90) - 1.1 * (gpa[i] - 2.8) + 0.12 * late[i] + 0.03 * work[i];
            dropped[i] = rng.NextBernoulli(1 / (1 + Math.Exp(-logit))) ? 1 : 0;
        }
        var students = DataFrame.FromColumns(new Dictionary<string, double[]>
        {
            ["attendance_pct"] = attendance, ["gpa"] = gpa, ["late_assignments"] = late,
            ["travel_km"] = travel, ["work_hours"] = work, ["dropped_out"] = dropped,
        });
        string[] features = ["attendance_pct", "gpa", "late_assignments", "travel_km", "work_hours"];

        var x = students.ToNdArray(features);
        var y = students.Numeric("dropped_out").ToNdArray();
        var split = Selection.Split(x, y, testSize: 0.3, seed: 5, stratify: true);   // keep the dropout rate in both halves

        var forest = new RandomForestClassifier(nTrees: 150, maxDepth: 6, minSamplesLeaf: 5, seed: 3);
        forest.Fit(split.TrainX, split.TrainY);
        var risk = forest.PredictProbabilities(split.TestX).Column(1);              // P(dropout)
        var auc = Metrics.RocAucScore(split.TestY, risk);
        var (fpr, tpr, _) = Metrics.RocCurve(split.TestY, risk);

        // Counsellors can meet the 20% of students the model ranks riskiest. How many leavers is that?
        var testCount = split.TestY.Size;
        var order = Enumerable.Range(0, testCount).OrderByDescending(i => risk.At(i)).ToArray();
        var seen = order.Take(testCount / 5).Count(i => split.TestY.At(i) == 1);
        var leavers = (int)split.TestY.ToArray().Sum();
        var importances = forest.FeatureImportances.ToArray();

        var figure = GalleryCharts.Figure(10, 4.6);
        var (left, right) = (figure.AddSubplot(1, 2, 1), figure.AddSubplot(1, 2, 2));
        left.Plot(fpr, tpr, $"random forest (AUC {auc:F2})");
        left.Plot([0, 1], [0, 1], "coin flip", Color.Parse("#8A94A3"), style: LineStyle.Dashed, lineWidth: 1.2);
        left.SetTitle("ROC curve, held-out students").SetXLabel("false alarm rate").SetYLabel("leavers caught");
        left.Legend(LegendLocation.LowerRight);

        var ranked = Enumerable.Range(0, features.Length).OrderBy(i => importances[i]).ToArray();
        right.BarH(Enumerable.Range(0, features.Length).Select(i => (double)i).ToArray(),
                   ranked.Select(i => importances[i]).ToArray(), color: GalleryCharts.Of(GraviLibrary.GraviLearn));
        right.SetYTicks(Enumerable.Range(0, features.Length).Select(i => (double)i).ToArray(),
                        ranked.Select(i => features[i].Replace('_', ' ')).ToArray());
        right.SetTitle("What drives the risk score").SetXLabel("importance (mean impurity decrease)");
        output.Chart(figure);

        output.Metric("Dropout rate", $"{y.ToArray().Average() * 100:F0}%", $"{n:N0} past students");
        output.Metric("ROC AUC (held out)", auc.ToString("F3"), "0.5 is guessing, 1.0 is perfect ranking");
        output.Metric("Leavers in the top-20% list", $"{seen} of {leavers}", $"{100.0 * seen / leavers:F0}% recall for a fifth of the caseload");
        output.Line($"Seeing the riskiest fifth of students reaches {100.0 * seen / leavers:F0}% of those who would leave, " +
                    $"{(double)seen / leavers / 0.2:F1}× better than picking students at random. " +
                    $"The strongest signal is {features[ranked[^1]].Replace('_', ' ')}.");
    }
}
