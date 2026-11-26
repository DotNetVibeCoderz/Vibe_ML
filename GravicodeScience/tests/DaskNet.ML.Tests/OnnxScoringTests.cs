using Gravicode.Science.DaskNet;
using Gravicode.Science.DaskNet.Distributed;
using Gravicode.Science.DaskNet.ML;
using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviLearn;
using Gravicode.Science.GraviLearn.Linear;
using Gravicode.Science.GraviLearn.Preprocessing;
using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.Science.Tests.DaskNet.ML;

/// <summary>ONNX scoring is pinned against the model it was exported from, and against a model
/// written by Python's onnx package whose output is known in closed form.</summary>
public sealed class OnnxScoringTests
{
    private static string Fixture(string name) => Path.Combine(AppContext.BaseDirectory, "Fixtures", name);

    [Fact]
    public async Task An_exported_regression_pipeline_scores_like_the_pipeline_locally_and_on_a_cluster()
    {
        var (x, y, _) = Datasets.MakeRegression(2_000, features: 4, noise: 0.3, seed: 8);
        var pipeline = new Pipeline().Add(new StandardScaler()).Add(new RidgeRegression(alpha: 0.5));
        pipeline.Fit(x, y);
        var model = OnnxExport.Build(pipeline, 4);
        string[] features = ["x0", "x1", "x2", "x3"];
        var frame = DaskFrame.FromFrame(DataFrame.FromMatrix(x, features), 5);
        var expected = pipeline.Predict(x).ToArray();

        void Check(DataFrame scored)
        {
            var got = scored.Numeric("onnx").Values.ToArray();
            Assert.Equal(expected.Length, got.Length);
            for (var i = 0; i < got.Length; i++)            // the exported graph is float32
                Assert.True(Math.Abs(got[i] - expected[i]) <= 1e-4 * Math.Max(1, Math.Abs(expected[i])), $"row {i}: {got[i]} vs {expected[i]}");
            Assert.Equal(features.Append("onnx"), scored.ColumnNames);
        }

        Check(frame.ScoreOnnx(model, features, columns: ["onnx"]).Compute());
        await using var cluster = await LocalCluster.StartAsync(workers: 2, threadsPerWorker: 2);
        Check(frame.ScoreOnnx(model, features, columns: ["onnx"]).Compute(cluster.Client));
    }

    [Fact]
    public void An_exported_classifier_returns_the_winning_class_index()
    {
        var iris = Datasets.LoadIris();
        var pipeline = new Pipeline().Add(new StandardScaler()).Add(new LogisticRegression(maxIterations: 300));
        pipeline.Fit(iris.Features, iris.Target);
        string[] features = ["a", "b", "c", "d"];
        var frame = DaskFrame.FromFrame(DataFrame.FromMatrix(iris.Features, features), 3);
        var scored = frame.ScoreOnnx(OnnxExport.Build(pipeline, 4), features).Compute();
        var predicted = pipeline.Predict(iris.Features).ToArray();
        var classes = ((LogisticRegression)pipeline.Steps[1]).Classes;
        var labels = scored.Numeric("label").Values.ToArray().Select(i => classes[(int)i]).ToArray();
        var agree = predicted.Zip(labels).Count(p => p.First == p.Second);
        Assert.True(agree >= predicted.Length - 1, $"{agree}/{predicted.Length} agree");   // float32 may flip a tie
    }

    [Fact]
    public void A_python_written_double_model_gives_its_closed_form_and_names_wide_outputs()
    {
        // Y = X W + b with W = [[1, -2], [0.5, 0.25], [3, 0]], b = [0.1, -0.3]; written by onnx.helper (onnx 1.22, opset 13).
        var table = new DataFrame([new NumericSeries("p", [1, 0, -2]), new NumericSeries("q", [2, 1, 0.5]), new NumericSeries("r", [3, 0, 1]),
            new TextSeries("id", ["a", "b", "c"])]);
        var frame = DaskFrame.FromFrame(table, 2);
        var scored = frame.ScoreOnnx(Fixture("affine_double.onnx"), ["p", "q", "r"]).Compute();
        Assert.Equal(["p", "q", "r", "id", "Y_0", "Y_1"], scored.ColumnNames);
        double[][] rows = [[1, 2, 3], [0, 1, 0], [-2, 0.5, 1]];
        for (var i = 0; i < 3; i++)
        {
            Assert.Equal(rows[i][0] * 1 + rows[i][1] * 0.5 + rows[i][2] * 3 + 0.1, scored.Numeric("Y_0")[i], 12);
            Assert.Equal(rows[i][0] * -2 + rows[i][1] * 0.25 - 0.3, scored.Numeric("Y_1")[i], 12);
        }

        Assert.Throws<ArgumentException>(() => frame.ScoreOnnx(Fixture("affine_double.onnx"), ["p", "q"]));
        Assert.Throws<ArgumentException>(() => frame.ScoreOnnx(Fixture("affine_double.onnx"), ["p", "q", "r"], output: "Z"));
        Assert.Throws<ArgumentException>(() => frame.ScoreOnnx(Fixture("affine_double.onnx"), ["p", "q", "r"], columns: ["only-one"]));
    }
}
