using BlazorML.Core.Data;
using BlazorML.Core.Domain;
using BlazorML.ML.Evaluation;
using BlazorML.ML.Execution;
using BlazorML.ML.Execution.Executors;
using BlazorML.ML.Training;

namespace BlazorML.Tests;

/// <summary>
/// Trains real ML.NET models on small synthetic datasets with a signal planted in them, then
/// checks the model found it. These are the tests that would catch a broken featurisation or a
/// mis-wired trainer, which no amount of unit testing around the edges would.
/// </summary>
public class TrainingIntegrationTests
{
    private readonly AlgorithmExecutor _algorithms = new();
    private readonly TrainingExecutor _training = new();
    private readonly ScoringExecutor _scoring = new();

    /// <summary>A separable binary problem: the label follows a threshold on one feature.</summary>
    private static TabularData BinaryData(int rows = 300)
    {
        var random = new Random(11);
        var table = TabularData.WithColumns("pendapatan", "usia", "kota", "beli");

        for (var i = 0; i < rows; i++)
        {
            var income = random.Next(1, 100);
            var age = random.Next(18, 70);
            var city = (string[])["Bandung", "Jakarta", "Medan"] is var c ? c[random.Next(3)] : "?";

            // Planted signal with a little noise, so a working model should clear 0.8 AUC
            // but a perfect score would mean something is leaking.
            var buys = income > 55 ^ random.NextDouble() < 0.08;

            table.AddRow(income, age, city, buys ? "1" : "0");
        }

        return table;
    }

    private static TabularData RegressionData(int rows = 300)
    {
        var random = new Random(13);
        var table = TabularData.WithColumns("luas", "kamar", "harga");

        for (var i = 0; i < rows; i++)
        {
            var area = 40 + random.Next(0, 160);
            var bedrooms = 1 + area / 50;
            var price = area * 12 + bedrooms * 40 + (random.NextDouble() - 0.5) * 60;

            table.AddRow(area, bedrooms, Math.Round(price, 2));
        }

        return table;
    }

    private TrainedModelHandle Train(string algorithmId, TabularData data, string? label)
    {
        var algorithmContext = TestContext.For(algorithmId);
        var spec = _algorithms.Output<TrainerSpec>(algorithmContext);

        var trainModule = spec.Task == MlTask.Clustering ? "train.clustering" : "train.model";
        var trainContext = TestContext.For(trainModule,
            label is null ? null : [("labelColumn", label)], spec, data);

        return _training.Output<TrainedModelHandle>(trainContext);
    }

    private TabularData Score(TrainedModelHandle model, TabularData data)
    {
        var context = TestContext.For("score.model", [("appendOriginal", "true")], model, data);
        return _scoring.Output<TabularData>(context);
    }

    // ---------------------------------------------------------------- binary

    [Theory]
    [InlineData("algo.bin.fastTree")]
    [InlineData("algo.bin.logistic")]
    [InlineData("algo.bin.sdca")]
    [InlineData("algo.bin.fastForest")]
    [InlineData("algo.bin.perceptron")]
    public void Binary_trainers_learn_a_planted_signal(string algorithmId)
    {
        var data = BinaryData();
        var model = Train(algorithmId, data, "beli");
        var scored = Score(model, data);

        Assert.True(scored.Has(Evaluator.PredictedLabel),
            $"{algorithmId} produced no {Evaluator.PredictedLabel} column.");

        var result = Evaluator.Evaluate(scored, MlTask.BinaryClassification, "beli", "1");

        Assert.True(result.Metrics["Accuracy"] > 0.8,
            $"{algorithmId} only reached accuracy {result.Metrics["Accuracy"]:F3} on separable data.");
    }

    [Fact]
    public void Binary_scoring_produces_a_probability_the_evaluator_can_build_a_curve_from()
    {
        var data = BinaryData();
        var model = Train("algo.bin.fastTree", data, "beli");
        var scored = Score(model, data);

        var result = Evaluator.Evaluate(scored, MlTask.BinaryClassification, "beli", "1");

        Assert.True(scored.Has(Evaluator.Probability));
        Assert.NotNull(result.RocCurve);
        Assert.True(result.RocCurve!.Count > 10);
        Assert.True(result.Metrics["AUC"] > 0.8);
    }

    [Fact]
    public void Categorical_and_numeric_columns_are_both_featurised()
    {
        // "kota" is a string column; if the featuriser dropped it the trainer would still run,
        // so the check is that the model records it as a feature it used.
        var model = Train("algo.bin.fastTree", BinaryData(), "beli");

        Assert.Contains("kota", model.FeatureColumns);
        Assert.Contains("pendapatan", model.FeatureColumns);
        Assert.DoesNotContain("beli", model.FeatureColumns);
    }

    // ------------------------------------------------------------ multiclass

    [Fact]
    public void Multiclass_predictions_come_back_as_the_original_labels_not_key_numbers()
    {
        var random = new Random(17);
        var data = TabularData.WithColumns("ukuran", "kelas");

        foreach (var (name, centre) in (( string Name, int Centre )[])
                 [("kecil", 10), ("sedang", 50), ("besar", 90)])
        {
            for (var i = 0; i < 60; i++)
            {
                data.AddRow(centre + random.Next(-6, 6), name);
            }
        }

        var model = Train("algo.multi.sdca", data, "kelas");
        var scored = Score(model, data);

        var predictions = scored.Rows
            .Select(r => Convert.ToString(r[scored.IndexOf(Evaluator.PredictedLabel)]))
            .Distinct()
            .ToList();

        // A missing MapKeyToValue would leave these as "1", "2", "3".
        Assert.All(predictions, p => Assert.Contains(p, (string[])["kecil", "sedang", "besar"]));

        var result = Evaluator.Evaluate(scored, MlTask.MulticlassClassification, "kelas");
        Assert.True(result.Metrics["MacroAccuracy"] > 0.85);
    }

    // ------------------------------------------------------------- regression

    [Theory]
    [InlineData("algo.reg.fastTree")]
    [InlineData("algo.reg.sdca")]
    [InlineData("algo.reg.fastForest")]
    public void Regression_trainers_recover_a_linear_relationship(string algorithmId)
    {
        var data = RegressionData();
        var model = Train(algorithmId, data, "harga");
        var scored = Score(model, data);

        var result = Evaluator.Evaluate(scored, MlTask.Regression, "harga");

        Assert.True(result.Metrics["RSquared"] > 0.9,
            $"{algorithmId} only reached R² {result.Metrics["RSquared"]:F3} on a near-linear target.");
    }

    [Fact]
    public void Regression_residuals_carry_one_point_per_scored_row()
    {
        // Regression test for the split bug, from the other side: a residual count far below the
        // row count means rows were lost somewhere between scoring and evaluation.
        var data = RegressionData(200);
        var model = Train("algo.reg.fastTree", data, "harga");
        var scored = Score(model, data);

        var result = Evaluator.Evaluate(scored, MlTask.Regression, "harga");

        Assert.Equal(200, scored.RowCount);
        Assert.Equal(200, result.Residuals!.Count);
    }

    // ------------------------------------------------------------- clustering

    [Fact]
    public void Clustering_separates_two_obvious_groups()
    {
        var random = new Random(23);
        var data = TabularData.WithColumns("x", "y");

        for (var i = 0; i < 80; i++)
        {
            data.AddRow(random.Next(0, 10), random.Next(0, 10));
            data.AddRow(random.Next(90, 100), random.Next(90, 100));
        }

        var algorithmContext = TestContext.For("algo.cluster.kmeans", [("numberOfClusters", "2")]);
        var spec = _algorithms.Output<TrainerSpec>(algorithmContext);
        var trainContext = TestContext.For("train.clustering", null, spec, data);
        var model = _training.Output<TrainedModelHandle>(trainContext);

        var scored = Score(model, data);

        // The cluster index must not arrive under the classifier's column name.
        Assert.True(scored.Has(Evaluator.ClusterId), "Scoring produced no cluster column.");
        Assert.False(scored.Has(Evaluator.PredictedLabel));

        var result = Evaluator.Evaluate(scored, MlTask.Clustering, "x");

        Assert.Equal(2, result.Metrics["Clusters"]);
        Assert.True(result.Metrics["Balance"] > 0.9, "The two planted groups came out lopsided.");
    }

    // ------------------------------------------------------------ round trip

    [Fact]
    public void A_saved_model_scores_identically_after_being_reloaded()
    {
        // This is the path the published REST endpoint takes, so a break here would only show
        // up in production.
        var data = BinaryData(200);
        var model = Train("algo.bin.fastTree", data, "beli");
        var before = Score(model, data);

        var ml = new Microsoft.ML.MLContext(seed: 42);
        using var buffer = new MemoryStream();
        ml.Model.Save(model.Transformer, model.InputSchema, buffer);
        buffer.Position = 0;

        var reloaded = ml.Model.Load(buffer, out _);

        using var bridge = new MlDataBridge();
        var view = bridge.ToDataView(ml, data, "beli", MlTask.BinaryClassification, out _);
        var after = MlDataBridge.FromDataView(reloaded.Transform(view));

        var beforeColumn = before.IndexOf(Evaluator.PredictedLabel);
        var afterColumn = after.IndexOf(Evaluator.PredictedLabel);

        Assert.Equal(before.RowCount, after.RowCount);

        for (var r = 0; r < before.RowCount; r++)
        {
            Assert.Equal(
                Convert.ToString(before.Rows[r][beforeColumn]),
                Convert.ToString(after.Rows[r][afterColumn]));
        }
    }

    /// <summary>
    /// Regression test. The fitted pipeline copies the label into a "Label" column, so its input
    /// schema demands that column — including at inference time, when the label is the thing
    /// being predicted. Scoring unlabelled rows is the normal production case and must work.
    /// </summary>
    [Fact]
    public void Scoring_works_on_rows_that_carry_no_label_column()
    {
        var training = BinaryData(200);
        var model = Train("algo.bin.fastTree", training, "beli");

        var unlabelled = training.Clone();
        unlabelled.RemoveColumn("beli");
        Assert.False(unlabelled.Has("beli"));

        var scored = Score(model, unlabelled);

        Assert.Equal(unlabelled.RowCount, scored.RowCount);
        Assert.True(scored.Has(Evaluator.PredictedLabel));
        Assert.True(scored.Has(Evaluator.Probability));
    }

    [Fact]
    public void Scoring_unlabelled_rows_gives_the_same_answers_as_scoring_labelled_ones()
    {
        // The placeholder label the bridge inserts must not influence the prediction; if it did,
        // the API would quietly disagree with what the designer showed.
        var data = BinaryData(150);
        var model = Train("algo.bin.fastTree", data, "beli");

        var withLabel = Score(model, data);

        var stripped = data.Clone();
        stripped.RemoveColumn("beli");
        var withoutLabel = Score(model, stripped);

        var a = withLabel.IndexOf(Evaluator.PredictedLabel);
        var b = withoutLabel.IndexOf(Evaluator.PredictedLabel);

        for (var r = 0; r < data.RowCount; r++)
        {
            Assert.Equal(
                Convert.ToString(withLabel.Rows[r][a]),
                Convert.ToString(withoutLabel.Rows[r][b]));
        }
    }

    // -------------------------------------------------------------- failures

    [Fact]
    public void Train_model_refuses_a_supervised_trainer_with_no_label()
    {
        var spec = _algorithms.Output<TrainerSpec>(TestContext.For("algo.bin.fastTree"));
        var context = TestContext.For("train.model", [("labelColumn", null)], spec, BinaryData(50));

        var error = Assert.Throws<InvalidOperationException>(() =>
            _training.ExecuteAsync(context).GetAwaiter().GetResult());

        Assert.Contains("label column", error.Message);
    }

    [Fact]
    public void Train_model_says_which_input_is_missing_rather_than_dereferencing_null()
    {
        var context = TestContext.For("train.model", [("labelColumn", "beli")], null, null);

        var error = Assert.Throws<InvalidOperationException>(() =>
            _training.ExecuteAsync(context).GetAwaiter().GetResult());

        Assert.Contains("Algorithm", error.Message);
    }
}
