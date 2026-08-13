using Gravicode.Science.GraviLearn;
using Gravicode.Science.GraviLearn.Decomposition;
using Gravicode.Science.GraviLearn.Linear;
using Gravicode.Science.GraviLearn.Preprocessing;
using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviNum.Io;
using Xunit;

namespace Gravicode.Science.Tests.GraviLearn;

/// <summary>
/// Tests for exporting a fitted pipeline to ONNX.
/// </summary>
/// <remarks>
/// These check the file this library produces can be read back and holds the right weights. The
/// stronger check — that <c>onnx.checker</c> calls the graph valid and Python's onnxruntime
/// reproduces the predictions — was run separately against the real tooling, because pulling
/// onnxruntime into the test suite would add a large native dependency for a check that only needs
/// running when the writer changes.
/// </remarks>
public class OnnxExportTests
{
    private static (NdArray X, NdArray Y) Fixture(int rows = 60, int features = 4)
    {
        var rng = new GraviRandom(11);
        var x = rng.StandardNormal(rows, features);
        var y = NdArray.Zeros(rows);

        // A linearly separable target, so the fitted model is well determined.
        for (var i = 0; i < rows; i++) y.SetAt(i, x[i, 0] + 0.5 * x[i, 1] > 0 ? 1 : 0);
        return (x, y);
    }

    [Fact]
    public void AFittedPipelineExportsToAReadableModel()
    {
        var (x, y) = Fixture();
        var pipeline = new Pipeline()
            .Add(new StandardScaler())
            .Add(new LogisticRegression(maxIterations: 200));
        pipeline.Fit(x, y);

        Assert.True(OnnxExport.Supports(pipeline));

        var tensors = OnnxReader.ReadWeightsByName(Save(pipeline, x.Shape[1]));

        // The scaler's fitted statistics must survive the round trip, as float32.
        var scaler = (StandardScaler)pipeline.Steps[0];
        Assert.Contains("scaler_mean", tensors.Keys);
        for (var j = 0; j < scaler.Mean.Size; j++)
            Assert.Equal(scaler.Mean.At(j), tensors["scaler_mean"].Values[j], 5);

        Assert.Contains("scaler_scale", tensors.Keys);
        Assert.Contains("logit_coef", tensors.Keys);
        Assert.Contains("logit_intercept", tensors.Keys);
    }

    [Fact]
    public void PcaComponentsAreExportedTransposedForTheMatMul()
    {
        var (x, y) = Fixture();
        var pipeline = new Pipeline()
            .Add(new StandardScaler())
            .Add(new PCA(components: 2))
            .Add(new LogisticRegression(maxIterations: 200));
        pipeline.Fit(x, y);

        var tensors = OnnxReader.ReadWeightsByName(Save(pipeline, x.Shape[1]));
        var pca = (PCA)pipeline.Steps[1];

        // ComponentVectors is (components x features); the graph multiplies rows of x by it, so
        // the stored constant must be its transpose. Getting this backwards produces a model that
        // loads cleanly and computes nonsense.
        var exported = tensors["pca_components"];
        Assert.Equal([pca.FeatureCount, 2], exported.Shape);

        for (var i = 0; i < pca.FeatureCount; i++)
            for (var j = 0; j < 2; j++)
                Assert.Equal(pca.ComponentVectors[j, i], exported.Values[i * 2 + j], 5);
    }

    [Fact]
    public void ARegressionPipelineExportsItsCoefficients()
    {
        var rng = new GraviRandom(13);
        var x = rng.StandardNormal(50, 3);
        var y = NdArray.Zeros(50);
        for (var i = 0; i < 50; i++) y.SetAt(i, 2 * x[i, 0] - x[i, 1] + 0.5 * x[i, 2] + 1);

        var pipeline = new Pipeline()
            .Add(new MinMaxScaler())
            .Add(new RidgeRegression(alpha: 0.1));
        pipeline.Fit(x, y);

        var tensors = OnnxReader.ReadWeightsByName(Save(pipeline, 3));

        // MinMax is folded into one multiply and add rather than the four-operation form it is
        // defined by, so what should appear is a slope and an offset, not a min and a range.
        Assert.Contains("minmax_slope", tensors.Keys);
        Assert.Contains("minmax_offset", tensors.Keys);
        Assert.Contains("linear_coef", tensors.Keys);
        Assert.Contains("linear_intercept", tensors.Keys);

        Assert.Equal([3, 1], tensors["linear_coef"].Shape);
    }

    [Fact]
    public void AnUnfittedPipelineIsRefused()
    {
        var pipeline = new Pipeline().Add(new StandardScaler()).Add(new LogisticRegression());
        Assert.Throws<InvalidOperationException>(() => OnnxExport.Build(pipeline, 4));
    }

    [Fact]
    public void APipelineWithNoAffineFormIsRefusedRatherThanApproximated()
    {
        // A decision tree is not an affine map. Emitting *something* for it would produce a model
        // that loads and predicts wrongly, which is far worse than refusing.
        var (x, y) = Fixture();
        var pipeline = new Pipeline()
            .Add(new StandardScaler())
            .Add(new Gravicode.Science.GraviLearn.Trees.DecisionTree(maxDepth: 3));
        pipeline.Fit(x, y);

        Assert.False(OnnxExport.Supports(pipeline));
        Assert.Contains("DecisionTree", OnnxExport.UnsupportedSteps(pipeline));

        var error = Assert.Throws<NotSupportedException>(() => OnnxExport.Build(pipeline, x.Shape[1]));
        Assert.Contains("DecisionTree", error.Message);
    }

    [Fact]
    public void TheExportedGraphDeclaresASymbolicBatchDimension()
    {
        // Pinning the row count to whatever the training set had is a classic export bug: the
        // model then refuses the single row a serving endpoint actually sends. The dimension is
        // written as a name rather than a number, which shows up as "batch" in the bytes.
        var (x, y) = Fixture();
        var pipeline = new Pipeline().Add(new StandardScaler()).Add(new LogisticRegression(maxIterations: 100));
        pipeline.Fit(x, y);

        var bytes = OnnxExport.Build(pipeline, x.Shape[1]);
        Assert.Contains("batch", System.Text.Encoding.UTF8.GetString(bytes), StringComparison.Ordinal);
    }

    /// <summary>Writes the model to a temporary file and returns the path.</summary>
    private static string Save(Pipeline pipeline, int features)
    {
        var path = Path.Combine(Path.GetTempPath(), $"gravicode-{Guid.NewGuid():N}.onnx");
        OnnxExport.Save(pipeline, path, features);
        return path;
    }
}
