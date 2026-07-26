using BlazorML.Core.Data;
using BlazorML.Core.Domain;
using BlazorML.ML.Evaluation;

namespace BlazorML.Tests;

public class EvaluatorTests
{
    private static TabularData Binary((string Actual, string Predicted, double Probability)[] rows)
    {
        var table = TabularData.WithColumns("Label", Evaluator.PredictedLabel, Evaluator.Probability);

        foreach (var (actual, predicted, probability) in rows)
        {
            table.AddRow(actual, predicted, probability);
        }

        return table;
    }

    [Fact]
    public void Binary_confusion_matrix_counts_land_in_the_right_cells()
    {
        // 2 true positives, 1 false positive, 3 true negatives, 1 false negative.
        var scored = Binary([
            ("1", "1", 0.9), ("1", "1", 0.8), ("1", "0", 0.3),
            ("0", "1", 0.6), ("0", "0", 0.2), ("0", "0", 0.1), ("0", "0", 0.05)
        ]);

        var result = Evaluator.Evaluate(scored, MlTask.BinaryClassification, "Label", "1");

        // Row 0 is the negative class, row 1 the positive; column order matches.
        Assert.Equal([[3, 1], [1, 2]], result.ConfusionMatrix);
        Assert.Equal(5d / 7, result.Metrics["Accuracy"], 6);
        Assert.Equal(2d / 3, result.Metrics["Precision"], 6);
        Assert.Equal(2d / 3, result.Metrics["Recall"], 6);
    }

    [Fact]
    public void Auc_is_one_when_every_positive_outranks_every_negative()
    {
        var scored = Binary([
            ("1", "1", 0.99), ("1", "1", 0.90), ("0", "0", 0.20), ("0", "0", 0.10)
        ]);

        var result = Evaluator.Evaluate(scored, MlTask.BinaryClassification, "Label", "1");

        Assert.Equal(1.0, result.Metrics["AUC"], 6);
    }

    [Fact]
    public void Auc_is_zero_when_the_ranking_is_exactly_backwards()
    {
        var scored = Binary([
            ("1", "1", 0.10), ("1", "1", 0.20), ("0", "0", 0.90), ("0", "0", 0.99)
        ]);

        var result = Evaluator.Evaluate(scored, MlTask.BinaryClassification, "Label", "1");

        Assert.Equal(0.0, result.Metrics["AUC"], 6);
    }

    /// <summary>
    /// The rank-based AUC exists to handle ties correctly. All-identical scores carry no ranking
    /// information at all, which is exactly 0.5 — a naive threshold sweep gets this wrong.
    /// </summary>
    [Fact]
    public void Auc_is_one_half_when_every_score_is_tied()
    {
        var scored = Binary([
            ("1", "1", 0.5), ("1", "1", 0.5), ("0", "1", 0.5), ("0", "1", 0.5)
        ]);

        var result = Evaluator.Evaluate(scored, MlTask.BinaryClassification, "Label", "1");

        Assert.Equal(0.5, result.Metrics["AUC"], 6);
    }

    [Fact]
    public void Auc_is_not_a_number_when_only_one_class_is_present()
    {
        // With no negatives there is nothing to rank against; reporting 1.0 here would be a lie.
        var scored = Binary([("1", "1", 0.9), ("1", "1", 0.8)]);

        var result = Evaluator.Evaluate(scored, MlTask.BinaryClassification, "Label", "1");

        Assert.True(double.IsNaN(result.Metrics["AUC"]));
    }

    [Fact]
    public void Roc_curve_starts_at_the_origin_and_ends_at_one_one()
    {
        var scored = Binary([
            ("1", "1", 0.9), ("1", "1", 0.7), ("0", "0", 0.4), ("0", "0", 0.1)
        ]);

        var result = Evaluator.Evaluate(scored, MlTask.BinaryClassification, "Label", "1");
        var curve = result.RocCurve!;

        Assert.Equal(0, curve[0].FalsePositiveRate);
        Assert.Equal(0, curve[0].TruePositiveRate);
        Assert.Equal(1, curve[^1].FalsePositiveRate);
        Assert.Equal(1, curve[^1].TruePositiveRate);

        // The curve must be monotonic in both axes or it is not a ROC curve.
        for (var i = 1; i < curve.Count; i++)
        {
            Assert.True(curve[i].FalsePositiveRate >= curve[i - 1].FalsePositiveRate);
            Assert.True(curve[i].TruePositiveRate >= curve[i - 1].TruePositiveRate);
        }
    }

    [Fact]
    public void Multiclass_macro_accuracy_weights_a_rare_class_as_heavily_as_a_common_one()
    {
        var table = TabularData.WithColumns("Label", Evaluator.PredictedLabel);

        // Nine of ten "besar" correct, but the single "kecil" row is wrong.
        for (var i = 0; i < 9; i++)
        {
            table.AddRow("besar", "besar");
        }

        table.AddRow("besar", "besar");
        table.AddRow("kecil", "besar");

        var result = Evaluator.Evaluate(table, MlTask.MulticlassClassification, "Label");

        Assert.Equal(10d / 11, result.Metrics["MicroAccuracy"], 6);
        // Macro averages per-class recall: (1.0 + 0.0) / 2.
        Assert.Equal(0.5, result.Metrics["MacroAccuracy"], 6);
    }

    [Fact]
    public void Regression_metrics_match_hand_computed_values()
    {
        var table = TabularData.WithColumns("Label", Evaluator.Score);
        table.AddRow(10.0, 12.0);   // error  2
        table.AddRow(20.0, 18.0);   // error -2
        table.AddRow(30.0, 34.0);   // error  4

        var result = Evaluator.Evaluate(table, MlTask.Regression, "Label");

        Assert.Equal(8d / 3, result.Metrics["MeanAbsoluteError"], 6);
        Assert.Equal(24d / 3, result.Metrics["MeanSquaredError"], 6);
        Assert.Equal(Math.Sqrt(8), result.Metrics["RootMeanSquaredError"], 6);
        Assert.Equal(3, result.Residuals!.Count);
    }

    /// <summary>
    /// Regression test for the split bug: a single test row makes MAE and RMSE identical, which
    /// is the tell that an evaluation is being computed from almost no data. Real data never
    /// looks like this, so the metrics must differ once there is genuine spread.
    /// </summary>
    [Fact]
    public void Regression_mae_and_rmse_differ_once_errors_vary()
    {
        var table = TabularData.WithColumns("Label", Evaluator.Score);
        table.AddRow(10.0, 11.0);
        table.AddRow(20.0, 25.0);

        var result = Evaluator.Evaluate(table, MlTask.Regression, "Label");

        Assert.NotEqual(result.Metrics["MeanAbsoluteError"], result.Metrics["RootMeanSquaredError"], 6);
    }

    [Fact]
    public void Regression_r_squared_is_one_for_a_perfect_fit()
    {
        var table = TabularData.WithColumns("Label", Evaluator.Score);
        table.AddRow(1.0, 1.0);
        table.AddRow(2.0, 2.0);
        table.AddRow(3.0, 3.0);

        var result = Evaluator.Evaluate(table, MlTask.Regression, "Label");

        Assert.Equal(1.0, result.Metrics["RSquared"], 6);
    }

    [Fact]
    public void Clustering_balance_is_one_for_evenly_sized_clusters()
    {
        var table = TabularData.WithColumns(Evaluator.ClusterId);

        foreach (var cluster in (string[])["1", "1", "2", "2", "3", "3"])
        {
            table.AddRow(cluster);
        }

        var result = Evaluator.Evaluate(table, MlTask.Clustering, "Label");

        Assert.Equal(3, result.Metrics["Clusters"]);
        Assert.Equal(1.0, result.Metrics["Balance"], 6);
    }

    [Fact]
    public void Evaluate_returns_an_empty_result_rather_than_throwing_when_columns_are_absent()
    {
        var table = TabularData.WithColumns("sesuatu");
        table.AddRow("nilai");

        var result = Evaluator.Evaluate(table, MlTask.BinaryClassification, "Label");

        Assert.Empty(result.Metrics);
    }
}
